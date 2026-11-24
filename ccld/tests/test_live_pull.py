import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ccld.analyzer.ccld_analyze import Config, analyze, comparable_groups
from ccld.analyzer.live import Endpoint, LiveAnalysis, SlowWindow, make_analysis
from ccld.tools.decode_async_wait import Event, WaitDecoder
from ccld.tools.live_transport import NodeBuffer, Pull, rank_stream_lost_record, receive_rank

ROOT = Path(__file__).resolve().parents[2]


def event(kind, rank, op, when, duration=0):
    e = Event()
    e.magic, e.version, e.type = 0x444c4343, 1, kind
    e.hash, e.rank, e.nranks = 42, rank, 2
    e.seq, e.nowNs, e.windowNs = op, when, 1000
    if kind in (3, 4):
        e.startNs = when - duration
    e.op, e.datatype, e.algorithm, e.protocol = b'AllReduce', b'float32', b'RING', b'SIMPLE'
    e.count, e.bytes, e.nChannels = (0 if kind == 1 else 4096), 16384, 1
    e.channels[0].send = e.channels[0].recv = 40
    return bytes(e)


class RankStreamLossTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_exit_is_scoped_and_propagates_with_last_operation(self):
        buffer = NodeBuffer(16)
        stream = asyncio.StreamReader()
        stream.feed_data(event(1, 1, 0, 0) + event(2, 1, 4, 100))
        stream.feed_eof()
        writer = Mock(wait_closed=AsyncMock())
        with patch('ccld.tools.live_transport.peer_process', return_value=(42, '100')), \
                patch('ccld.tools.live_transport.process_exited', return_value=True):
            await receive_rank(stream, writer, buffer)
        self.assertEqual((buffer.failed, buffer.end), (0, 3))

        results = []
        analysis, decoder = LiveAnalysis(Config(baseline_ms=1.), results.append), WaitDecoder()
        analysis.add(decoder.consume(event(1, 0, 0, 0)))
        for raw in buffer.records[:buffer.end]:
            analysis.add(decoder.consume(raw))
        analysis.evaluate()
        self.assertEqual([(x['kind'], x['roots']) for x in results], [('rank_exit', [1])])
        self.assertEqual(results[0]['rank_streams'],
                         [{'rank': 1, 'last_op_count': 4, 'process_exited': True}])
        # A later peer teardown belongs to the already disrupted communicator.
        analysis.add(decoder.consume(rank_stream_lost_record((42, 0, 2), 5, True)))
        analysis.evaluate()
        self.assertEqual(len(results), 1)

    async def test_socket_eof_without_exit_proof_is_a_rank_scoped_gap(self):
        results = []
        analysis, decoder = LiveAnalysis(Config(baseline_ms=1.), results.append), WaitDecoder()
        analysis.add(decoder.consume(event(1, 1, 0, 0)))
        analysis.add(decoder.consume(rank_stream_lost_record((42, 1, 2), 2, False)))
        analysis.evaluate()
        self.assertEqual([(x['category'], x['roots']) for x in results], [('coverage_gap', [])])
        self.assertEqual(results[0]['rank_streams'][0]['rank'], 1)


class LiveDecisionTests(unittest.TestCase):
    def test_mixed_ring_requires_bracketed_waits_on_the_same_clock(self):
        for reason in ('valid', 'fast_late_rank', 'pure_s1', 'other_host', 'missing_sample', 'counter_regressed'):
            with self.subTest(reason=reason):
                cfg = Config(baseline_ms=1, repeats=1, window_s=.1, ring_wait_phases=True)
                reports, traces = [], []
                analysis = LiveAnalysis(cfg, reports.append)
                for rank in range(4):
                    traces.append(dict(event='comm_init', comm_hash='fixture', world_size=4, rank=rank,
                                       op_count=0, mono_ns=1, host='different' if reason == 'other_host'
                                       and rank == 3 else 'host'))
                for op in range(1, 6):
                    t = op * 100_000_000
                    for rank in range(4):
                        row = dict(comm_hash='fixture', op_count=op, world_size=4, rank=rank,
                                   op='AllReduce', count=4096, bytes=16384, datatype='float32',
                                   algorithm='Ring', protocol='Simple', wait_probe=True,
                                   start_mono_ns=t + (5_000_000 if rank == 2 else 0))
                        channel = dict(id=0, ring_prev=(rank-1) % 4, ring_next=(rank+1) % 4,
                                       wait_data_spins=4, wait_credit_spins=4)
                        if rank != 2:
                            for stamp in (t+4_500_000, t+5_500_000):
                                if reason == 'missing_sample' and rank == 3 and stamp < t+5_000_000:
                                    continue
                                progress = dict(channel, wait_credit_spins=500 if reason == 'counter_regressed'
                                                and rank == 0 else 4)
                                traces.append(dict(row, event='snapshot', mono_ns=stamp, duration_ns=stamp-t,
                                                   channels=[progress]))
                        final = dict(channel, wait_credit_spins=200 if rank == 0 and reason != 'pure_s1' else 4)
                        finish = t + (6_300_000 if rank == 2 and reason in ('fast_late_rank', 'pure_s1')
                                      else 12_000_000)
                        traces.append(dict(row, event='complete', mono_ns=finish,
                                           duration_ns=finish-row['start_mono_ns'], channels=[final]))
                for trace in traces:
                    analysis.add(trace)
                analysis.evaluate()
                for decisions in (reports, analyze(traces, cfg)):
                    decision = [e for e in decisions if e.get('category') == 'slow'][-1]
                    if reason in ('valid', 'fast_late_rank'):
                        self.assertEqual((decision['kind'], decision['compute_roots'], decision['communication_roots']),
                                         ('S3', [2], [1]))
                        self.assertEqual(decision['method'], 'post_entry_ring_wait')
                    elif reason == 'pure_s1':
                        self.assertEqual((decision['kind'], decision['roots']), ('S1', [2]))
                    else:
                        self.assertEqual((decision['kind'], decision['roots'], decision['communication_roots']),
                                         ('undetermined', [], []))

    def test_grouped_exclusion_retires_only_after_all_ranks_agree(self):
        for peer in ('matching', 'different_count', 'complete', 'missing'):
            with self.subTest(peer=peer):
                reports, decoder = [], WaitDecoder()
                analysis = LiveAnalysis(Config(hang_s=.1), reports.append)
                for rank in range(2):
                    analysis.add(decoder.consume(event(1, rank, 0, 0)))
                    entered = Event.from_buffer_copy(event(2, rank, 1, 1_000))
                    if rank == 1 and peer == 'different_count':
                        entered.count += 64
                    analysis.add(decoder.consume(bytes(entered)))
                for rank in range(2):
                    if rank == 1 and peer in ('complete', 'missing'):
                        if peer == 'complete':
                            analysis.add(decoder.consume(event(4, 1, 1, 5_000, 4_000)))
                        continue
                    dropped = Event.from_buffer_copy(event(6, rank, 1, 3_000))
                    dropped.reason = b'multiple_collectives'
                    analysis.add(decoder.consume(bytes(dropped)))
                    if rank == 0:
                        analysis.evaluate()
                        self.assertFalse(reports)
                analysis.add(decoder.consume(event(3, 0, 2, 200_010_000, 200_000_000)))
                analysis.evaluate()
                self.assertEqual([(r['category'], r['kind'], r['roots']) for r in reports], {
                    'matching': [('unsupported', 'skipped', []), ('hang', 'H1', [1])],
                    'different_count': [('coverage_gap', 'undetermined', [])],
                    'complete': [('coverage_gap', 'undetermined', [])],
                    'missing': [],
                }[peer])

    def test_completed_collective_reports_a_minority_signature(self):
        reports, decoder = [], WaitDecoder()
        analysis = LiveAnalysis(Config(slow_warmup_ops=100), reports.append)
        for rank in range(4):
            raw = Event.from_buffer_copy(event(1, rank, 0, 0))
            raw.nranks = 4
            analysis.add(decoder.consume(bytes(raw)))
        for rank in range(4):
            raw = Event.from_buffer_copy(event(4, rank, 1, 200, 100))
            raw.nranks = 4
            if rank == 2:
                raw.count -= 4
            analysis.add(decoder.consume(bytes(raw)))
        analysis.evaluate()
        self.assertEqual([(r['category'], r['kind'], r['roots']) for r in reports],
                         [('collective_mismatch', 'H2', [2])])

    def test_frame_busy_keeps_known_alignment_with_completed_peers(self):
        results, decoder = [], WaitDecoder()
        analysis = LiveAnalysis(Config(hang_s=.1), results.append)
        for rank in range(2):
            analysis.add(decoder.consume(event(1, rank, 0, 0)))
            analysis.add(decoder.consume(event(2, rank, 1, 1_000)))
        raw = Event.from_buffer_copy(event(6, 1, 1, 3_000))
        raw.reason = b'frame_slot_busy'
        analysis.add(decoder.consume(bytes(raw)))
        analysis.add(decoder.consume(event(4, 0, 1, 5_000, 4_000)))
        analysis.add(decoder.consume(event(3, 0, 2, 200_010_000, 200_000_000)))
        analysis.evaluate()
        self.assertEqual([(r['category'], r['kind'], r['roots']) for r in results],
                         [('coverage', 'skipped', []), ('hang', 'H1', [1])])

    def test_completed_collective_distinguishes_builtin_reductions_at_the_same_shape(self):
        reports, decoder = [], WaitDecoder()
        analysis = LiveAnalysis(Config(), reports.append)
        for rank in range(4):
            raw = Event.from_buffer_copy(event(1, rank, 0, 0))
            raw.nranks = 4
            analysis.add(decoder.consume(bytes(raw)))
        for rank in range(4):
            raw = Event.from_buffer_copy(event(4, rank, 1, 200, 100))
            raw.nranks, raw.mode, raw.frameGroups = 4, 1 << 30, 3 if rank == 2 else 1
            analysis.add(decoder.consume(bytes(raw)))
        analysis.evaluate()
        self.assertEqual([(r['category'], r['kind'], r['roots']) for r in reports],
                         [('collective_mismatch', 'H2', [2])])

    def test_same_host_linear_tree_reads_identity_from_initialization(self):
        for different_hosts in (False, True):
            with self.subTest(different_hosts=different_hosts):
                reports, decoder = [], WaitDecoder()
                analysis = LiveAnalysis(Config(baseline_ms=1., repeats=1, window_s=.1),
                                        reports.append)
                for rank in range(4):
                    raw = Event.from_buffer_copy(event(1, rank, 0, 0))
                    raw.nranks = 4
                    raw.host = f'host-{rank if different_hosts else 0}'.encode()
                    analysis.add(decoder.consume(bytes(raw)))
                for op in range(1, 4):
                    for rank in range(4):
                        duration = 2 if rank == 2 or op != 1 else 60
                        raw = Event.from_buffer_copy(event(4, rank, op, op * 200_000_000,
                                                          duration * 1_000_000))
                        raw.nranks, raw.algorithm, raw.channels[0].parent = 4, b'TREE', rank - 1
                        analysis.add(decoder.consume(bytes(raw)))
                    analysis.evaluate()
                diagnoses = [(r['kind'], r['roots']) for r in reports if r['category'] == 'slow']
                self.assertEqual(diagnoses, [] if different_hosts else [('S1', [2])])

    def tree_states(self, op, durations):
        states = {}
        decoder = WaitDecoder(defer_channels=True)
        for rank, parent in enumerate((-1, 0, 0, 1, 1, 2, 2, 3)):
            raw = Event.from_buffer_copy(event(4, rank, op, op*100*10**6, durations[rank]*10**6))
            raw.nranks, raw.algorithm, raw.nChannels = 8, b'TREE', 2
            for cid in (0,1):
                raw.channels[cid].id, raw.channels[cid].parent = cid, parent
                n = 10 if rank == 4 else 2
                raw.channels[cid].sendChanges = raw.channels[cid].recvChanges = n
            states[rank] = decoder.consume(bytes(raw))
        return states

    def test_tree_slow_publishes_layer_candidates_and_one_corroborated_root(self):
        results = []
        window = SlowWindow('000000000000002a', Config(baseline_ms=1., repeats=1), results.append)
        # Rank 1 is last to enter the first layer. Rank 4 appears slow by
        # communication rate in the affected lower layer.
        window.add(1, self.tree_states(1, (15, 4, 15, 15, 15, 15, 15, 15)), 8)
        window.flush()
        layer = [r for r in results if r['root_scope'] == 'tree_layer']
        self.assertEqual(len(layer), 4)
        self.assertTrue(all(not r['roots'] for r in layer))
        self.assertEqual({r['group_kind']: r['candidate_ranks'] for r in layer},
                         {'S1': [1], 'S2': [4]})
        final = [r for r in results if r['root_scope'] == 'communicator']
        self.assertEqual([(r['status'], r['kind'], r['roots']) for r in final],
                         [('candidate', 'S1', [1])])

    def test_tree_vote_abstains_when_layers_pick_different_operations(self):
        results = []
        window = SlowWindow('000000000000002a', Config(baseline_ms=1., repeats=1), results.append)
        window.add(1, self.tree_states(1, (15, 4, 15, 15, 15, 15, 15, 15)), 8)
        window.add(2, self.tree_states(2, (20, 19, 19, 4, 20, 20, 20, 20)), 8)
        window.flush()
        final = [r for r in results if r['root_scope'] == 'communicator']
        self.assertEqual([(r['status'], r['roots'], r['operation_counts']) for r in final],
                         [('incomplete', [], [1, 2])])

    def test_tree_grouping_reads_parent_fields_without_expanding_metrics(self):
        decoder = WaitDecoder(defer_channels=True)
        states = {}
        for rank, parent in enumerate((-1, 0, 0)):
            raw = Event.from_buffer_copy(event(4, rank, 1, 100, 10))
            raw.nranks, raw.algorithm, raw.channels[0].parent = 3, b'TREE', parent
            states[rank] = decoder.consume(bytes(raw))
        groups, warnings = comparable_groups(states, 3)
        self.assertEqual(groups, [('tree:channel=0:depth=1', [1, 2], 0)])
        self.assertFalse(warnings)
        self.assertTrue(all(e['channels']._decoded is None for e in states.values()))

    def test_slow_candidate_with_deferred_channels_has_same_s2_root(self):
        cfg = Config(window_s=.1, baseline_ms=1., repeats=1)

        def run(defer):
            results = []
            decoder = WaitDecoder(defer_channels=defer)
            analysis = LiveAnalysis(cfg, results.append)
            for rank in range(2):
                analysis.add(decoder.consume(event(1, rank, 0, 0)))
            for op, when, durations in [(1, 100, (5, 5)), (2, 150, (10, 11)), (3, 205, (2, 2)),
                                        (4, 310, (1, 1))]:
                for rank, duration in enumerate(durations):
                    raw = Event.from_buffer_copy(event(4, rank, op, when*10**6, duration*10**6))
                    raw.channels[0].sendChanges = raw.channels[0].recvChanges = (10 if rank == 0 else 2)
                    analysis.add(decoder.consume(bytes(raw)))
                analysis.evaluate()
            for rank in range(2):
                analysis.add(decoder.consume(event(8, rank, 4, 320*10**6)))
            analysis.evaluate()
            return [item for item in results if item['category'] == 'slow']

        eager, deferred = run(False), run(True)
        self.assertEqual([(item['op_count'], item['kind'], item['roots']) for item in deferred],
                         [(2, 'S2', [0])])
        for item in eager + deferred:
            item.pop('location_latency_ms')
        self.assertEqual(deferred, eager)

    def test_streaming_slow_windows_keep_baseline_and_repeat_count(self):
        cfg = Config(window_s=.1, baseline_rounds=2, baseline_ms=1., repeats=2)
        results, history = [], []
        analysis, decoder = LiveAnalysis(cfg, results.append), WaitDecoder()
        for rank in range(2):
            analysis.add(decoder.consume(event(1, rank, 0, 0)))
        for op, when, durations in [(1, 10, (1, 1)), (2, 40, (1, 1)),
                                     (3, 80, (10, 2)), (4, 160, (16, 2)),
                                     (5, 220, (1, 1))]:
            for rank, duration in enumerate(durations):
                record = decoder.consume(event(4, rank, op, when*10**6, duration*10**6))
                history.append(record)
                analysis.add(record)
            analysis.evaluate(allow_partial_hang=False)
            self.assertFalse(analysis.comms['000000000000002a'].pending)
        for rank in range(2):
            analysis.add(decoder.consume(event(8, rank, 5, 230*10**6)))
        analysis.evaluate()
        signature = lambda items: [(r['op_count'], r['kind'], r['roots'], r['baseline'])
                                   for r in items if r['category'] == 'slow']
        self.assertEqual(signature(results), [(3, 'detected', [], 'dynamic'), (4, 'S1', [1], 'dynamic')])
        self.assertEqual(signature(results), signature(analyze(history, cfg)))

    def test_slow_warmup_excludes_startup_and_preserves_later_diagnoses(self):
        cfg = Config(window_s=.1, baseline_rounds=1, baseline_ms=1.,
                     repeats=1, slow_warmup_ops=2)
        reports, history = [], []
        analysis, decoder = LiveAnalysis(cfg, reports.append), WaitDecoder()
        for rank in range(2):
            analysis.add(decoder.consume(event(1, rank, 0, 0)))
        for op, durations in enumerate(((16, 1), (1, 22), (1, 1), (15, 2), (1, 1)), 1):
            for rank, duration in enumerate(durations):
                record = decoder.consume(event(4, rank, op, op * 200_000_000,
                                               duration * 1_000_000))
                history.append(record)
                analysis.add(record)
            analysis.evaluate()
        for rank in range(2):
            analysis.add(decoder.consume(event(8, rank, 5, 1_002_000_000)))
        analysis.evaluate()
        result = lambda rows: [(r['op_count'], r['kind'], r['roots'], r['baseline'])
                               for r in rows if r['category'] == 'slow']
        self.assertEqual(result(reports), [(4, 'S1', [1], 'dynamic')])
        self.assertEqual(result(analyze(history, cfg)), result(reports))

    def test_completed_prefix_retires_while_backlogged_and_tolerates_slot_order(self):
        cfg = Config(window_s=.05, baseline_ms=1., baseline_rounds=100, repeats=1)
        results, history = [], []
        analysis, decoder = LiveAnalysis(cfg, results.append), WaitDecoder()
        for rank in range(2):
            analysis.add(decoder.consume(event(1, rank, 0, 0)))
        state = analysis.comms['000000000000002a']
        state.MAX_PENDING = 4
        # A later slot can be observed completing before the previous slot.
        # Keep monotonic entry times while reversing two completion times.
        for op, start, done0, done1 in [(1, 10, 15, 15), (2, 20, 90, 24),
                                       (3, 30, 52, 31), (4, 100, 105, 105),
                                       (5, 200, 201, 201), (6, 310, 311, 311)]:
            for rank, end in enumerate((done0, done1)):
                record = decoder.consume(event(4, rank, op, end*10**6, (end-start)*10**6))
                history.append(record)
                analysis.add(record)
            analysis.evaluate(allow_partial_hang=False)
            self.assertEqual(state.next_op, op + 1)
            self.assertLessEqual(len(state.completed_by_time), 2)
        for rank in range(2):
            analysis.add(decoder.consume(event(8, rank, 6, 312*10**6)))
        analysis.evaluate()
        self.assertFalse(any(item['category'] == 'coverage_gap' for item in results))
        decisions = lambda items: [(r['op_count'], r['kind'], r['roots'], r['window'])
                                   for r in items if r['category'] == 'slow']
        self.assertEqual(decisions(results), decisions(analyze(history, cfg)))

    def test_overwritten_remote_cursor_does_not_become_rank_evidence(self):
        ring = NodeBuffer(2)
        for seq in range(3):
            ring.append(event(5, 0, seq, seq))
        _, data = ring.batch(0)
        self.assertEqual(data, b'')
        reader = Endpoint('127.0.0.1', 1)
        with self.assertRaisesRegex(ValueError, 'outside the retained buffer'):
            reader.accept(Pull(ring.epoch, 1, ring.end, 0, 0, data), LiveAnalysis(Config(), self.fail))


class LiveProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_coalescing_preserves_progress_and_incomplete_hang(self):
        endpoint = Endpoint('127.0.0.1', 1)
        ms = 10**6
        batches = [
            event(1, 0, 0, 0) + event(1, 1, 0, 0)
            + event(2, 0, 1, ms) + event(2, 1, 1, ms)
            + event(2, 0, 2, 12*ms) + event(3, 0, 1, 8*ms, 7*ms)
            + event(4, 1, 1, 10*ms, 9*ms) + event(4, 0, 1, 11*ms, 10*ms),
            event(3, 0, 2, 1000*ms, 988*ms) + event(5, 1, 1, 1000*ms),
            event(4, 0, 2, 1201*ms, 1189*ms) + event(2, 1, 2, 1200*ms)
            + event(4, 1, 2, 1201*ms, ms)
            + event(8, 0, 2, 1202*ms) + event(8, 1, 2, 1202*ms),
        ]

        async def run(coalesce):
            decisions = []
            cfg = Config(hang_s=.5, window_s=.1, baseline_ms=1., repeats=1)
            analysis = make_analysis(cfg, decisions.append, 2, coalesce)
            try:
                for data in batches:
                    analysis.receive(endpoint, data)
                    await analysis.process(True)
                await analysis.finish([endpoint])
                for item in decisions:
                    item.pop('location_latency_ms', None)
                self.assertEqual(analysis.gaps, 0)
                self.assertFalse(analysis.comms)
                self.assertTrue(any(d['kind'] == 'H1' and d['roots'] == [1] for d in decisions))
                return dict(analysis.events), dict(analysis.completed), decisions, dict(analysis.coalesced)
            finally:
                await analysis.close()

        baseline, folded = await run(False), await run(True)
        self.assertEqual(baseline[:3], folded[:3])
        self.assertEqual(folded[3], {'entered': 3, 'snapshot': 1})

    async def test_parallel_decoder_keeps_a_wait_pair_across_batches(self):
        analysis = make_analysis(Config(), lambda _: None, 8, coalesce_completed=True)
        endpoint = Endpoint('127.0.0.1', 1)
        wait, done = [Event.from_buffer_copy(event(kind, 0, 1, 100, 10))
                      for kind in (7, 4)]
        wait.reserved = 4
        wait.channels[0].parent = wait.channels[0].sendChanges = 1
        done.reason = b'ring_wait_v1'
        second_comm = Event.from_buffer_copy(event(1, 0, 0, 1))
        second_comm.hash = 44
        try:
            analysis.receive(endpoint, event(1, 0, 0, 1) + event(1, 1, 0, 1) +
                             bytes(second_comm) + bytes(wait))
            await analysis.process(False)
            analysis.receive(endpoint, bytes(done))
            await analysis.process(False)
            await analysis.finish([endpoint])
            self.assertEqual(analysis.completed, {'AllReduce': 1})
            self.assertEqual(len(analysis.pools), 3)
            observed = analysis.comms['000000000000002a'].pending[1][0]
            self.assertEqual(observed['channels'][0]['ring_next'], 1)
        finally:
            await analysis.close()

    async def test_controller_pulls_two_agents_and_reports_hang_while_rank_streams_are_open(self):
        with tempfile.TemporaryDirectory(prefix='ccld-pull-', dir='/tmp') as temp:
            env = dict(os.environ, CCLD_PULL_TOKEN='32' * 32)
            children, writers = [], []

            async def launch(module, *args):
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, '-m', module, *args, cwd=ROOT, env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                children.append(proc)
                return proc

            try:
                endpoints, paths = [], []
                for rank in range(2):
                    path = Path(temp)/f'node{rank}.sock'
                    agent = await launch('ccld.tools.live_transport', '--socket', str(path),
                                         '--bind', '127.0.0.1', '--port', '0')
                    ready = await asyncio.wait_for(agent.stdout.readline(), 5)
                    self.assertTrue(ready.startswith(b'CCLD_NODE_READY '), ready)
                    endpoints.extend(['--node', ready.split()[1].decode()])
                    paths.append(path)
                observer = await launch('ccld.analyzer.live', *endpoints, '--poll-s', '.01',
                                        '--baseline-ms', '1', '--hang-s', '1', '--decode-workers', '2')
                self.assertEqual(await asyncio.wait_for(observer.stderr.readline(), 5), b'CCLD_ANALYZER_READY\n')
                for rank, path in enumerate(paths):
                    _, writer = await asyncio.open_unix_connection(str(path))
                    writers.append(writer)
                    writer.write(event(1, rank, 0, 1))
                    await writer.drain()
                writers[0].write(event(2, 0, 1, 100) + event(3, 0, 1, 2_000_000_100, 2_000_000_000))
                await writers[0].drain()
                # A poll can precede the other node's init. Wait for the
                # eventual H1 decision while both producer sockets stay open.
                async def result():
                    seen = []
                    while line := await observer.stdout.readline():
                        item = json.loads(line)
                        if item.get('kind') == 'H1':
                            return item
                        seen.append(item)
                    self.fail(f'analyzer stopped before locating H1: {seen}')
                item = await asyncio.wait_for(result(), 5)
                self.assertEqual(item['roots'], [1])
                self.assertEqual(item['op_count'], 1)
                for rank, writer in enumerate(writers):
                    writer.write(event(4, rank, 1, 2_000_001_000, 2_000_000_000)
                                 + event(8, rank, 1, 2_000_002_000))
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                observer.send_signal(signal.SIGTERM)
                stdout, stderr = await asyncio.wait_for(observer.communicate(), 5)
                self.assertEqual(observer.returncode, 0, stderr)
                summary = json.loads(stdout.splitlines()[-1])
                self.assertEqual(summary['events']['comm_end'], 2)
                self.assertEqual(summary['communicators_with_gaps'], 0)
                self.assertEqual(summary['completed_operations'], {'AllReduce': 2})
                self.assertTrue(all(p.is_socket() for p in Path(temp).iterdir()))
            finally:
                for writer in writers:
                    writer.close()
                for proc in reversed(children):
                    if proc.returncode is None:
                        proc.terminate()
                    await asyncio.wait_for(proc.wait(), 5)
