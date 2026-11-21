import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/tests'))
import ccld_gfx936_online_ring_wait as adapter


def write_trace(root, world_size, hosts):
    for rank in range(world_size):
        records = [{'event': 'comm_init', 'rank': rank, 'world_size': world_size,
                    'comm_hash': 'c', 'host': hosts[rank], 'poll_us': 200}]
        state = {'wait_probe': True, 'channels': [
            {'id': 0, 'ring_prev': (rank - 1) % world_size,
             'ring_next': (rank + 1) % world_size,
             'wait_data_spins': 1, 'wait_credit_spins': 100 if rank == 0 else 1,
             'send_count': 10, 'recv_count': 10}]}
        for op in (4, 5, 6):
            completed_ns = 1_000_000_000 + (op - 4) * 1_100_000_000
            records.append({'event': 'complete', 'rank': rank, 'world_size': world_size,
                            'comm_hash': 'c', 'op_count': op, 'op': 'AllReduce',
                            'algorithm': 'RING', 'protocol': 'LL', 'count': 256,
                            'datatype': 'ncclFloat32', 'n_channels': 1,
                            'mono_ns': completed_ns + rank, 'start_mono_ns': completed_ns - 10_000_000,
                            'duration_ns': 10_000_000 if rank == 1 else 100_000_000,
                            **state})
        (root / f'rank-{rank}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))


class Gfx936OnlineRingWaitTests(unittest.TestCase):
    def test_single_node_four_rank_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, 4, ['node0'] * 4)
            states, comm = adapter.completed_rounds(root, 'LL', 4)
            self.assertEqual(comm, 'c')
            result = adapter.decide(states, 0.5, True)
            self.assertEqual(result['candidate_ranks'], [1])

    def test_dual_node_eight_rank_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, 8, ['node0'] * 4 + ['node1'] * 4)
            states, comm = adapter.completed_rounds(root, 'LL', 8)
            self.assertEqual(comm, 'c')
            self.assertEqual(adapter.decide(states, 0.5, True)['candidate_ranks'], [1])

    def test_live_adapter_runs_nv_gate_and_wait_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, 4, ['node0'] * 4)
            import subprocess
            result = subprocess.run([
                sys.executable, str(ROOT / 'tools/tests/ccld_gfx936_online_ring_wait.py'),
                str(root), '--protocol', 'LL', '--world-size', '4', '--threshold', '0.5',
                '--baseline-ms', '10', '--window-s', '1', '--repeats', '2', '--flush-last-window',
                '--poll-s', '0.02'], check=True, capture_output=True, text=True, timeout=5)
            outputs = [json.loads(line) for line in result.stdout.splitlines()]
            base = [x for x in outputs if x.get('category') == 'slow']
            wait = [x for x in outputs if x.get('category') == 'online_wait']
            self.assertTrue(base)
            self.assertTrue(wait)
            self.assertEqual(wait[-1]['candidate_ranks'], [1])

    def test_layout_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_trace(root, 8, ['node0'] * 8)
            with self.assertRaisesRegex(ValueError, 'rank-to-node layout'):
                adapter.completed_rounds(root, 'LL', 8)


if __name__ == '__main__':
    unittest.main()
