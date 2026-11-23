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

    def test_offline_group_exclusions_preserve_later_coverage(self):
        entries = [ev(rank, 'entered') for rank in range(4)]
        drops = [ev(rank, 'dropped', reason='multiple_collectives') for rank in range(4)]
        following = [ev(rank, 'snapshot', op=2, duration=2000) for rank in (0, 1, 2)]
        out = analyze(entries + drops + following, self.cfg)
        self.assertEqual([(row['kind'], row['roots']) for row in out], [('skipped', []), ('H1', [3])])
        drops[0]['reason'] = 'frame_slot_busy'
        out = analyze(entries[1:] + drops + following, self.cfg)
        self.assertEqual([(row['kind'], row['roots']) for row in out],
                         [('undetermined', []), ('undetermined', [])])
        known = [entries[0], drops[0]] + [ev(rank, 'complete') for rank in (1, 2, 3)]
        out = analyze(known + following, self.cfg)
        self.assertEqual([(row['kind'], row['roots']) for row in out],
                         [('skipped', []), ('H1', [3])])

    def test_h2_reports_completed_rank(self):
        rows = [ev(i, 'snapshot', duration=2000) for i in (0, 1, 2)]
        rows.append(ev(3, 'complete', duration=500, count=255))
        out = analyze(rows, self.cfg)[0]
        self.assertEqual((out['kind'], out['roots']), ('H2', [3]))
        self.assertEqual(out['operation_signatures'][3][1], 255)

    def test_completed_collective_reports_conflicting_counts(self):
        majority = [ev(rank, 'complete', count=256) for rank in range(3)]
        out = analyze(majority + [ev(3, 'complete', count=255)], self.cfg)
        self.assertEqual([(e['category'], e['kind'], e['roots']) for e in out],
                         [('collective_mismatch', 'H2', [3])])
        majority[2]['count'] = 255
        out = analyze(majority + [ev(3, 'complete', count=255)], self.cfg)
        self.assertEqual([(e['kind'], e['roots']) for e in out], [('undetermined', [])])

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
        self.assertEqual(analyze(list(events.values()), self.cfg)[0]['roots'], [1])

    def test_tree_validates_each_channel_and_does_not_share_mutable_groups(self):
        events = {i: ev(i, 'snapshot', duration=2000, algorithm='Tree') for i in range(4)}
        for rank, event in events.items():
            event['channels'] = [
                {'id': 0, 'tree_parent': [3, 3, 0, -1][rank]},
                {'id': 1, 'tree_parent': [-1, 2, 1, 0][rank]},
                {'id': 2, 'tree_parent': [-1, 0, 0, -1][rank]},
                {'id': 3, 'tree_parent': [-1, 0, False, 1][rank]},
            ]
        groups, errors = comparable_groups(events, 4)
        self.assertEqual(groups, [('tree:channel=0:depth=1', [0, 1], 0)])
        self.assertEqual(errors, ['channel 1: cycle in tree topology', 'channel 2: expected one root',
                                  'channel 3: invalid or absent tree_parent'])
        groups[0][1].clear()
        self.assertEqual(comparable_groups(events, 4)[0], [('tree:channel=0:depth=1', [0, 1], 0)])

    def test_chain_tree_reports_slow_coverage_unavailable_without_alarm(self):
        rows = [ev(i, 'complete', duration=100, algorithm='Tree') for i in range(4)]
        for rank, row in enumerate(rows):
            row['channels'][0]['tree_parent'] = rank-1
        out = analyze(rows, self.cfg)
        self.assertFalse(any(e['category'] == 'slow' for e in out))
        warning = next(e for e in out if e['category'] == 'coverage')
        self.assertEqual((warning['kind'], warning['roots']), ('undetermined', []))
        self.assertIn('tree has no layer with at least two ranks', warning['warnings'])

    def test_chain_tree_can_detect_uniform_slowdown_after_signature_calibration(self):
        rows = []
        for op in range(1, 6):
            finished = op * 2_000_000_000
            for rank in range(4):
                duration = 10 if op < 3 else 75
                row = ev(rank, 'complete', op=op, t=finished, duration=duration, algorithm='Tree',
                         host='shared-host', start_mono_ns=finished-int(duration*1e6))
                row['channels'][0]['tree_parent'] = rank-1
                rows.append(row)
        cfg = Config(baseline_ms=10, baseline_rounds=2, baseline_scope='signature', window_s=1,
                     repeats=2, flush_last_window=True)
        observed = [(e['kind'], e['roots']) for e in analyze(rows, cfg)]
        self.assertEqual(observed, [('detected', []), ('undetermined', []), ('undetermined', [])])
        self.assertEqual(analyze(rows, self.cfg), [])

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

    def test_signature_baselines_ignore_healthy_large_ops_and_keep_small_fault(self):
        from dataclasses import replace
        events = []
        rounds = [(1024, (1, 1, 1, 1)), (1 << 20, (20, 20, 20, 20)),
                  (1024, (20, 20, 1, 20)), (1 << 20, (20, 20, 20, 20))]
        for op, (count, durations) in enumerate(rounds, 1):
            events.extend(ev(rank, 'complete', op=op, t=op * 100_000_000,
                             count=count, duration=duration)
                          for rank, duration in enumerate(durations))
        old = Config(baseline_rounds=1, baseline_ms=1, window_s=.08,
                     repeats=1, flush_last_window=True)
        self.assertTrue(any(r['op_count'] == 4 and r['category'] == 'slow'
                            for r in analyze(events, old)))
        qualified = analyze(events, replace(old, baseline_scope='signature'))
        self.assertEqual([(x['op_count'], x['kind'], x['roots']) for x in qualified if x['category'] == 'slow'],
                         [(3, 'S1', [2])])

    def test_linear_tree_uses_a_same_host_late_entry(self):
        inits = [ev(rank, 'comm_init', op=0, host='one-host') for rank in range(4)]
        rows = list(inits)
        for op, entered in ((1, 1_000_000_000), (2, 3_000_000_000)):
            completed = entered + 62_000_000
            for rank in range(4):
                start = entered + (60_000_000 if rank == 2 else 0)
                rows.append(ev(rank, 'complete', op=op, t=completed,
                               duration=(completed - start) / 1e6, algorithm='Tree',
                               start_mono_ns=start,
                               channels=[{'id': 0, 'tree_parent': rank-1}]))
        reports = analyze(rows, self.cfg)
        self.assertEqual([(r['kind'], r['roots']) for r in reports if r['kind'] == 'S1'], [('S1', [2])])
        self.assertTrue(any(r.get('method') == 'same_host_tree_entry_gap' for r in reports))
        for row in inits:
            row['host'] = f"node-{row['rank'] // 2}"
        self.assertFalse(any(r['kind'] == 'S1' for r in analyze(rows, self.cfg)))


if __name__ == '__main__':
    unittest.main()
