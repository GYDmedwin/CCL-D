import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analyzer'))
from ring_wait import RingWaitAttribution, completed_rounds, decide, graph_wait_scores


def sample_states():
    return {rank: {'wait_probe': True, 'channels': [
        {'id': 0, 'ring_prev': (rank - 1) % 8, 'ring_next': (rank + 1) % 8,
         'wait_data_spins': 1, 'wait_credit_spins': 100 if rank == 0 else 1,
         'send_count': 10, 'recv_count': 10, 'send_rate': 0.5, 'recv_rate': 0.5}]}
        for rank in range(8)}


def write_traces(root, states):
    for rank in range(8):
        init = {'event': 'comm_init', 'rank': rank, 'world_size': 8, 'comm_hash': 'c',
                'host': 'fixture', 'poll_us': 200, 'mono_ns': 0}
        records = [init]
        for op in (4, 5, 6):
            records.append({'event': 'complete', 'rank': rank, 'world_size': 8, 'comm_hash': 'c',
                            'op_count': op, 'op': 'AllReduce', 'algorithm': 'RING', 'protocol': 'LL',
                            'count': 256, 'bytes': 1024, 'datatype': 'ncclFloat32', 'n_channels': 1,
                            'mono_ns': 1_000_000_000 + (op - 4) * 1_100_000_000 + rank,
                            'duration_ns': 10_000_000 if rank == 1 else 100_000_000, **states[rank]})
        (root / f'{rank}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))


class RingWaitTests(unittest.TestCase):
    def test_directional_wait_requires_repeatable_rank_and_recovers_mixed_fault(self):
        states = sample_states()
        for base_kind, expected_kind, expected_ranks in (('S2', 'S2', [1]), ('S1', 'S3', [1, 2])):
            attribution = RingWaitAttribution(gap=0.5, slow_threshold=2)
            item = {'kind': base_kind, 'roots': [2], 'compute_roots': [2],
                    't_min_ms': 6, 't_base_ms': 1}
            for window in (0, 1):
                pending = attribution.refine(item, states, 4+window, window)
                self.assertEqual((pending['kind'], pending['roots']), ('detected', []))
            final = attribution.refine(item, states, 6, 2)
            self.assertEqual((final['kind'], final['roots']), (expected_kind, expected_ranks))
            self.assertEqual(final['communication_roots'], [1])

    def test_wait_score_and_majority_require_a_slow_gate(self):
        states = sample_states()
        metrics = graph_wait_scores(states)
        self.assertEqual(metrics['forced_top_ranks'], [1])
        self.assertAlmostEqual(sum(metrics['scores'].values()), 0)
        rounds = {op: states for op in (4, 5, 6)}
        self.assertIsNone(decide(rounds, 0.5, False)['detected'])
        self.assertEqual(decide(rounds, 0.5, True)['candidate_ranks'], [1])
        self.assertEqual(decide(rounds, 10, True)['candidate_ranks'], [])

    def test_zero_volume_and_invalid_topology_are_never_a_candidate(self):
        states = sample_states()
        for state in states.values():
            state['channels'][0].update(wait_credit_spins=0, wait_data_spins=0)
        self.assertEqual(graph_wait_scores(states)['status'], 'unavailable')
        self.assertIsNone(decide({op: states for op in (4, 5, 6)}, 0.5, True)['detected'])
        broken = copy.deepcopy(sample_states())
        broken[0]['channels'][0]['ring_next'] = 2
        with self.assertRaisesRegex(ValueError, 'Ring links'):
            graph_wait_scores(broken)

    def test_complete_input_and_dropped_record_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_traces(root, sample_states())
            rounds, comm = completed_rounds(root, 'LL')
            self.assertEqual(comm, 'c')
            self.assertEqual(decide(rounds, 0.5, True)['candidate_ranks'], [1])
            with (root / '0.jsonl').open('a') as out:
                out.write(json.dumps({'event': 'dropped'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'dropped'):
                completed_rounds(root, 'LL')

    def test_cli_does_not_infer_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_traces(root, sample_states())
            result = subprocess.run([sys.executable, str(ROOT / 'analyzer/ring_wait.py'), str(root),
                                     '--protocol', 'LL', '--threshold', '0.5'],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['status'], 'indeterminate')

    def test_online_observer_waits_for_a_real_base_slow_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_traces(root, sample_states())
            result = subprocess.run([sys.executable, str(ROOT / 'analyzer/online_ring_wait.py'), str(root),
                                     '--protocol', 'LL', '--threshold', '0.5', '--baseline-ms', '10',
                                     '--window-s', '1', '--repeats', '2', '--poll-s', '0.02'],
                                    check=True, capture_output=True, text=True, timeout=5)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertTrue(any(e.get('category') == 'slow' and e.get('kind') == 'S1' for e in events))
            wait = next(e for e in events if e.get('category') == 'online_wait')
            self.assertEqual(wait['candidate_ranks'], [1])
            self.assertEqual(wait['base_slow_kind'], 'S1')


if __name__ == '__main__':
    unittest.main()
