import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analyzer'))
from ccld_analyze import Config, analyze, comparable_groups, annotate_live_latency


def ev(rank, kind, op=1, duration=1, t=1_000_000_000, world=4, **fields):
    e = dict(event=kind, comm_hash='c', rank=rank, world_size=world, mono_ns=t,
             op_count=op, op='AllReduce', bytes=1024, count=256, datatype='float32',
             algorithm='Ring', protocol='Simple')
    if kind in ('snapshot', 'complete'):
        e['duration_ns'] = int(duration * 1e6)
        e['channels'] = [dict(id=0, send_count=10, recv_count=10, send_rate=.5,
                              recv_rate=.5, send_delta_per_ms=1, recv_delta_per_ms=1)]
    e.update(fields)
    return e


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(hang_s=1, baseline_ms=10, slow_threshold=3,
                          window_s=1, repeats=2, flush_last_window=True)

    def test_h1_needs_known_rank(self):
        rows = [ev(i, 'snapshot', duration=2000) for i in (0, 1, 2)]
        rows.append(ev(3, 'heartbeat', op=0))
        out = analyze(rows, self.cfg)
        self.assertEqual((out[0]['kind'], out[0]['roots']), ('H1', [3]))
        self.assertEqual(analyze(rows[:-1], self.cfg)[0]['kind'], 'undetermined')

    def test_h2_checks_complete_signatures_without_a_timeout(self):
        rows = [ev(rank, 'complete', count=64 if rank == 2 else 256) for rank in range(4)]
        out = analyze(rows, self.cfg)[0]
        self.assertEqual((out['category'], out['kind'], out['roots']), ('collective_mismatch', 'H2', [2]))
        rows[3]['count'] = 64
        out = analyze(rows, self.cfg)[0]
        self.assertEqual((out['kind'], out['roots']), ('undetermined', []))

    def test_h2_reports_completed_rank(self):
        rows = [ev(i, 'snapshot', duration=2000) for i in (0, 1, 2)]
        rows.append(ev(3, 'complete', duration=500, count=255))
        out = analyze(rows, self.cfg)[0]
        self.assertEqual((out['kind'], out['roots']), ('H2', [3]))
        self.assertEqual(out['operation_signatures'][3][1], 255)

    def test_later_h1_is_not_attributed_when_previous_round_is_hung(self):
        rows = [ev(i, 'snapshot', op=4, duration=2000) for i in (0, 1, 3)]
        rows.append(ev(2, 'complete', op=4, duration=500, count=64))
        rows.append(ev(2, 'snapshot', op=5, duration=2000, t=2_000_000_000))
        out = analyze(rows, self.cfg)
        self.assertEqual((out[0]['kind'], out[0]['roots']), ('H2', [2]))
        self.assertEqual((out[1]['kind'], out[1]['roots'], out[1]['blocked_by_op_count']), ('undetermined', [], 4))

    def test_h2_does_not_treat_late_callbacks_as_unhung_ranks(self):
        rows = [ev(0, 'complete', op=4, duration=20_000),
                ev(1, 'snapshot', op=4, duration=21_000),
                ev(2, 'complete', op=4, duration=500, count=64),
                ev(3, 'complete', op=4, duration=20_000)]
        out = analyze(rows, self.cfg)
        self.assertEqual((out[0]['kind'], out[0]['roots']), ('H2', [2]))

    def test_h3_counts_and_ties(self):
        rows = [ev(i, 'snapshot', duration=2000) for i in range(4)]
        rows[2]['channels'][0].update(send_count=1, recv_count=2)
        rows[3]['channels'][0].update(send_count=1, recv_count=2)
        out = analyze(rows, self.cfg)[0]
        self.assertEqual((out['kind'], out['roots']), ('H3', [2, 3]))

    def test_p2p_coverage_gap_is_explicit(self):
        gap = ev(1, 'coverage_gap', op=1, reason='p2p_group_not_traced')
        out = analyze([gap], self.cfg)
        self.assertEqual(out[0]['category'], 'coverage_gap')
        self.assertEqual(out[0]['kind'], 'undetermined')
        self.assertEqual(out[0]['rank'], 1)

    def test_coverage_gap_blocks_later_cross_rank_attribution(self):
        rows = [ev(i, 'complete', op=1) for i in range(4)]
        rows.append(ev(1, 'coverage_gap', op=1, reason='p2p_group_not_traced'))
        rows.extend(ev(i, 'snapshot', op=2, duration=2000) for i in (0, 2, 3))
        out = analyze(rows, self.cfg)
        later = [x for x in out if x.get('op_count') == 2]
        self.assertEqual(len(later), 1)
        self.assertEqual((later[0]['category'], later[0]['kind'], later[0]['roots']),
                         ('invalid', 'undetermined', []))
        self.assertEqual(later[0]['coverage_gap_after_op_count'], 1)

    def test_p2p_progress_is_not_diagnosed_with_ring_tree_rules(self):
        rows = [ev(i, 'snapshot', duration=2000, algorithm='P2P') for i in range(4)]
        for event in rows:
            event['op'] = 'P2PAllPeers'
        out = analyze(rows, self.cfg)
        self.assertEqual((out[0]['category'], out[0]['kind'], out[0]['roots']),
                         ('unsupported', 'undetermined', []))

    def test_dropped_frame_is_not_diagnosed(self):
        rows = [ev(i, 'snapshot', duration=2000) for i in range(4)]
        rows.append(ev(1, 'dropped', reason='frame_slot_busy'))
        out = analyze(rows, self.cfg)[0]
        self.assertEqual((out['category'], out['kind'], out['roots']), ('invalid', 'undetermined', []))

    def test_latency_uses_only_same_host_clock_and_unambiguous_comm(self):
        events = [ev(1, 'entered', op=1, t=100)]
        marker = {'rank': 1, 'host': 'node-a', 'mono_ns': 200, 'kind': 'S1'}
        item = {'comm_hash': 'c', 'op_count': 2, 'detection_latency_ms': None}
        same = annotate_live_latency(item, events, [marker], 'node-a', 1_000_200)
        self.assertEqual(same['detection_latency_ms'], 1.0)
        other = annotate_live_latency(item, events, [marker], 'node-b', 1_000_200)
        self.assertIsNone(other['detection_latency_ms'])
        ambiguous = events + [dict(events[0], comm_hash='other')]
        self.assertIsNone(annotate_live_latency(item, ambiguous, [marker], 'node-a', 1_000_200)['detection_latency_ms'])

    def test_live_h1_emission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = root / 'traces'; traces.mkdir()
            faults = root / 'faults'; faults.mkdir()
            out = root / 'out.ndjson'
            base = max(0, time.monotonic_ns() - 1_000_000)
            for rank in range(4):
                prior = [ev(rank, 'comm_init', op=0, t=base),
                         ev(rank, 'entered', op=1, t=base+1),
                         ev(rank, 'complete', op=1, t=base+2)]
                (traces / f'{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prior))
            marker_ns = time.monotonic_ns()
            marker = {'rank': 2, 'host': socket.gethostname(), 'mono_ns': marker_ns, 'kind': 'H1'}
            (faults / 'rank-2.ndjson').write_text(json.dumps(marker)+'\n')
            cmd = [sys.executable, str(Path(__file__).resolve().parents[1]/'analyzer/ccld_analyze.py'),
                   str(traces), '--baseline-ms', '1', '--hang-s', '.001', '--poll-s', '.05',
                   '--follow', '--fault-events', str(faults)]
            with out.open('w') as output:
                proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.PIPE)
                try:
                    for rank in (0, 1, 3):
                        sample = ev(rank, 'snapshot', op=2, duration=2, t=time.monotonic_ns())
                        with (traces / f'{rank}.jsonl').open('a') as stream:
                            stream.write(json.dumps(sample)+'\n')
                    deadline = time.monotonic()+3
                    found = None
                    while time.monotonic() < deadline:
                        text = out.read_text()
                        for line in text.splitlines():
                            record = json.loads(line)
                            if record.get('kind') == 'H1':
                                found = record
                        if found:
                            break
                        time.sleep(.05)
                    self.assertIsNotNone(found, 'live analyzer did not emit H1')
                    self.assertEqual(found['roots'], [2])
                    # This local sandbox may give child processes different clock origins.
                    # Same-host latency mapping is checked separately with explicit clocks.
                finally:
                    proc.terminate()
                    proc.communicate(timeout=3)

    def test_filter_small_barrier(self):
        rows = [ev(i, 'snapshot', duration=2000, bytes=4) for i in range(4)]
        self.assertEqual(analyze(rows, self.cfg), [])

    def test_tree_requires_topology_and_only_compares_peers(self):
        events = {i: ev(i, 'snapshot', duration=2000, algorithm='Tree') for i in range(4)}
        self.assertEqual(comparable_groups(events, 4)[0], [])
        for i, parent in enumerate([-1, 0, 0, 1]):
            events[i]['channels'][0]['tree_parent'] = parent
        groups, errors = comparable_groups(events, 4)
        self.assertFalse(errors)
        self.assertEqual(groups, [('tree:channel=0:depth=1', [1, 2], 0)])
        events[1]['channels'][0]['send_count'] = 1
        events[1]['channels'][0]['recv_count'] = 1
        unreferenced = analyze(list(events.values()), self.cfg)[0]
        self.assertEqual((unreferenced['kind'], unreferenced['roots']), ('undetermined', []))
        baseline = [ev(i, 'complete', algorithm='Tree') for i in range(4)]
        for rank, event in events.items():
            baseline[rank]['channels'][0]['tree_parent'] = event['channels'][0]['tree_parent']
            event['op_count'] = 2
        out = analyze(baseline + list(events.values()), self.cfg)[0]
        self.assertEqual((out['kind'], out['roots'], out['reference_op_count']), ('H3', [1], 1))
        events[3]['channels'][0].update(send_count=1, recv_count=1)
        tied = analyze(baseline + list(events.values()), self.cfg)[0]
        self.assertEqual((tied['kind'], tied['roots'], tied['candidate_ranks']),
                         ('undetermined', [], [1, 3]))

    def test_chain_tree_reports_slow_coverage_unavailable_without_alarm(self):
        rows = [ev(i, 'complete', duration=100, algorithm='Tree') for i in range(4)]
        for rank, row in enumerate(rows):
            row['channels'][0]['tree_parent'] = rank-1
        out = analyze(rows, self.cfg)
        self.assertFalse(any(e['category'] == 'slow' for e in out))
        warning = next(e for e in out if e['category'] == 'coverage')
        self.assertEqual((warning['kind'], warning['roots']), ('undetermined', []))
        self.assertIn('tree has no layer with at least two ranks', warning['warnings'])

    def test_slow_requires_repetition_and_finds_s1(self):
        rows = []
        for op, t in enumerate((1_000_000_000, 2_100_000_000), 1):
            for rank in range(4):
                rows.append(ev(rank, 'complete', op=op, t=t+rank, duration=10 if rank == 2 else 100))
        out = [x for x in analyze(rows, self.cfg) if x['category'] == 'slow']
        self.assertEqual([x['kind'] for x in out], ['detected', 'S1'])
        self.assertEqual(out[-1]['roots'], [2])
        self.assertAlmostEqual(out[-1]['p'], 1)

    def test_s2_uses_requested_minimum_metric(self):
        rows = []
        for rank in range(4):
            e = ev(rank, 'complete', duration=100, t=1_000_000_000+rank)
            if rank == 1:
                e['channels'][0].update(send_rate=.1, recv_rate=.1)
            rows.append(e)
        cfg = Config(baseline_ms=10, window_s=1, repeats=1, flush_last_window=True)
        out = analyze(rows, cfg)[0]
        self.assertEqual((out['kind'], out['roots']), ('S2', [1]))

    def test_s2_missing_rank_rate_is_indeterminate(self):
        rows = [ev(r, 'complete', duration=100, t=1_000_000_000+r) for r in range(4)]
        rows[2]['channels'][0]['recv_rate'] = None
        cfg = Config(baseline_ms=10, window_s=1, repeats=1, flush_last_window=True)
        item = analyze(rows,cfg)[0]
        self.assertEqual((item['kind'],item['roots']),('undetermined',[]))

    def test_h3_unequal_channels_is_indeterminate(self):
        rows = [ev(r, 'snapshot', duration=2000) for r in range(4)]
        rows[2]['channels'].append(dict(id=1,send_count=1,recv_count=1))
        item = analyze(rows,self.cfg)[0]
        self.assertEqual((item['kind'],item['roots']),('undetermined',[]))
        self.assertEqual(item['reason'], 'no comparable per-rank channel counts')
        self.assertEqual(item['operation_signatures'][2][1], 256)
        self.assertEqual(item['operation_signatures'][0][1], 256)

    def test_all_hung_signature_minority_is_h2_with_explicit_assumption(self):
        rows = [ev(r, 'snapshot', duration=2000) for r in range(4)]
        rows[2]['count'] = 64
        rows[2]['channels'].append(dict(id=1, send_count=1, recv_count=1))
        item = analyze(rows, self.cfg)[0]
        self.assertEqual((item['kind'], item['roots']), ('H2', [2]))
        self.assertEqual(item['signature_majority_ranks'], [0, 1, 3])
        self.assertEqual(item['signature_heuristic'], 'strict_majority_assumed_correct')
        self.assertEqual(item['operation_signatures'][2][1], 64)

    def test_all_hung_balanced_or_incomplete_signatures_are_indeterminate(self):
        rows = [ev(r, 'snapshot', duration=2000) for r in range(4)]
        rows[2]['count'] = rows[3]['count'] = 64
        item = analyze(rows, self.cfg)[0]
        self.assertEqual((item['kind'], item['roots']), ('undetermined', []))
        self.assertIn('no strict majority', item['reason'])
        rows[3].pop('datatype')
        item = analyze(rows, self.cfg)[0]
        self.assertEqual((item['kind'], item['roots']), ('undetermined', []))
        self.assertEqual(item['reason'], 'incomplete operation signatures')

    def test_all_hung_multiple_minority_ranks_are_not_forced_to_one(self):
        rows = [ev(r, 'snapshot', duration=2000, world=5) for r in range(5)]
        rows[3]['count'] = 64
        rows[4]['datatype'] = 'float16'
        item = analyze(rows, self.cfg)[0]
        self.assertEqual((item['kind'], item['roots']), ('H2', [3, 4]))

    def test_warmup_skip_and_baseline_range_warning(self):
        rows = []
        for op, (t, durations) in enumerate(((1_000_000_000, [1000]*4),
                                             (2_000_000_000, [10]*4),
                                             (3_000_000_000, [20]*4),
                                             (4_100_000_000, [100, 100, 1, 100])), 1):
            rows.extend(ev(r, 'complete', op=op, t=t+r, duration=durations[r]) for r in range(4))
        cfg = Config(baseline_ms=500, baseline_rounds=2, baseline_skip=1,
                     window_s=1, repeats=1, flush_last_window=True)
        out = [x for x in analyze(rows, cfg) if x.get('op_count') == 4 and x['category']=='slow'][0]
        self.assertEqual(out['t_base_ms'], 15)
        self.assertEqual(out['kind'], 'S1')
        self.assertIn('baseline_warning', out)

    def test_dynamic_baseline_waits_for_cutoff(self):
        rows = []
        for op, (t, d) in enumerate(((1_000_000_000, 10), (2_000_000_000, 20), (3_100_000_000, 100)), 1):
            rows.extend(ev(r, 'complete', op=op, t=t+r, duration=d) for r in range(4))
        cfg = Config(baseline_ms=5, baseline_rounds=2, window_s=1, repeats=1, flush_last_window=True)
        out = [x for x in analyze(rows, cfg) if x['category'] == 'slow']
        self.assertEqual(out[-1]['baseline'], 'dynamic')
        self.assertEqual(out[-1]['t_base_ms'], 15)


if __name__ == '__main__':
    unittest.main()
