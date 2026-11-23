import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / 'tools/tests/ccld_completed_mismatch.py'
SINGLE = ROOT / 'tools/tests/run_ccld_gfx936_docker_single_node_matrix.sh'
DUAL = ROOT / 'tools/tests/run_ccld_gfx936_docker_2node.sh'
RING_LIVE = ROOT / 'tools/tests/run_ccld_gfx936_ring_wait_live.sh'
EVAL51 = ROOT / 'tools/tests/run_antccl_eval51_gfx936.sh'


def event(rank, count):
    return {
        'event': 'complete', 'comm_hash': 'testcomm', 'op_count': 4,
        'rank': rank, 'world_size': 4, 'op': 'AllReduce',
        'count': count, 'bytes': count * 4, 'datatype': 'ncclFloat32',
        'algorithm': 'Ring', 'protocol': 'Simple', 'duration_ns': 1000,
    }


class CompletedMismatchTests(unittest.TestCase):
    def run_detector(self, events, world=4):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'rank.jsonl'
            trace.write_text(''.join(json.dumps(item) + '\n' for item in events))
            result = subprocess.run(
                ['python3', str(DETECTOR), str(trace), '--world-size', str(world)],
                check=True, capture_output=True, text=True)
            return [json.loads(line) for line in result.stdout.splitlines() if line]

    def test_completed_1024_byte_mismatch_identifies_minority_rank(self):
        rows = self.run_detector([event(rank, 261888 if rank == 2 else 262144)
                                  for rank in range(4)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['kind'], 'H2')
        self.assertTrue(rows[0]['detected'])
        self.assertFalse(rows[0]['hang_observed'])
        self.assertEqual(rows[0]['roots'], [2])
        self.assertEqual(rows[0]['mismatches'][0]['count_delta_bytes_from_majority'], -1024)

    def test_consistent_completed_operation_has_no_mismatch(self):
        rows = self.run_detector([event(rank, 262144) for rank in range(4)])
        self.assertEqual(rows, [])

    def test_explicit_h2_expectation_fails_when_the_round_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'rank.jsonl'
            rows = [event(rank, 261888 if rank == 2 else 262144) for rank in range(4)]
            trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            command = ['python3', str(DETECTOR), str(trace), '--world-size', '4',
                       '--expect-rank', '2', '--expect-op-count']
            self.assertEqual(subprocess.run(command + ['4'], capture_output=True).returncode, 0)
            rejected = subprocess.run(command + ['5'], capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn('missing expected', rejected.stderr)

    def test_dual_command_line_selects_dual_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / 'tools/tests'
            tests.mkdir(parents=True)
            entry = tests / EVAL51.name
            entry.write_text(EVAL51.read_text())
            for source, status in ((SINGLE, 17), (DUAL, 23)):
                stub = tests / source.name
                stub.write_text(f'#!/bin/sh\nexit {status}\n')
                stub.chmod(0o755)
            library = root / 'librccl.so'
            library.touch()
            env = dict(os.environ, CCLD_TOPOLOGY='single', RCCL_LIB=str(library),
                       CCLD_RUN=str(root / 'result'))
            result = subprocess.run(
                ['bash', str(entry), '--topology', 'dual', '--case', 'H2'],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 23, result.stderr)

    def test_mismatch_detector_is_wired_into_decoded_matrix_runners(self):
        source = EVAL51.read_text()
        self.assertIn('ccld_completed_mismatch.py', source)
        self.assertIn('H2)', source)
        self.assertTrue(SINGLE.exists() and DUAL.exists())

    def test_s2_live_acceptance_requires_detection_not_rank_vote(self):
        source = RING_LIVE.read_text()
        self.assertIn('validate_ccld_s2_detected.py', source)
        self.assertIn('analysis-final.jsonl', source)
        self.assertNotIn("candidate_ranks')==[target]", source)

    def test_s2_detection_gate_accepts_detected_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / 'analysis.jsonl'
            manifest = root / 'fault-config.json'
            analysis.write_text(json.dumps({'category': 'slow', 'kind': 'detected',
                                            'roots': [], 'op_count': 4}) + '\n')
            manifest.write_text(json.dumps({'fault_kind': 'S2', 'fault_rank': 2,
                                            'fault_cycles': 115000}))
            result = subprocess.run(['python3', str(ROOT / 'tools/tests/validate_ccld_s2_detected.py'),
                                     str(analysis), str(manifest), '--target-rank', '2'],
                                    check=True, capture_output=True, text=True)
            finding = json.loads(result.stdout)
            self.assertTrue(finding['detected'])
            self.assertFalse(finding['confirmed_classification'])


if __name__ == '__main__':
    unittest.main()
