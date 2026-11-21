import json
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / 'tools/tests/run_ccld_gfx936_docker_node.sh'
WRITER = ROOT / 'tools/tests/write_ccld_fault_config.py'


class FaultProvenanceTests(unittest.TestCase):
    def test_node_launcher_writes_fault_config(self):
        text = NODE.read_text()
        self.assertIn('fault-config.json', text)
        self.assertIn('NCCL_CCLD_FAULT_RANK', text)
        self.assertIn('NCCL_CCLD_FAULT_OP', text)
        self.assertIn('NCCL_CCLD_FAULT_CYCLES', text)
        self.assertIn('write_ccld_fault_config.py', text)
        self.assertIn('--h2-delta-bytes', text)


    def test_writer_records_explicit_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fault-config.json'
            env = os.environ.copy()
            env.update({
                'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'LL', 'CCLD_NNODES': '2',
                'CCLD_NPROC_PER_NODE': '4', 'NCCL_CCLD_FAULT_KIND': 'S2',
                'NCCL_CCLD_FAULT_RANK': '3', 'NCCL_CCLD_FAULT_OP': '4',
                'NCCL_CCLD_FAULT_REPEATS': '3', 'NCCL_CCLD_FAULT_CYCLES': '50000000',
                'RCCL_LIB': '/tmp/librccl.so', 'RCCL_BUILD_DIR': '/tmp/build',
            })
            subprocess.run(['python3', str(WRITER), str(path)], env=env, check=True)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded['world_size'], 8)
            self.assertEqual(loaded['fault_rank'], 3)
            self.assertEqual(loaded['fault_first_op'], 4)


    def test_h2_manifest_records_workload_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fault-config.json'
            env = os.environ.copy()
            env.update({
                'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple', 'CCLD_NNODES': '1',
                'CCLD_NPROC_PER_NODE': '4', 'CCLD_WORKLOAD_FAULT': 'H2',
                'CCLD_WORKLOAD_FAULT_RANK': '1', 'CCLD_WORKLOAD_FAULT_ITERATION': '3',
                'CCLD_WORKLOAD_H2_DELTA_BYTES': '1024', 'RCCL_LIB': '/tmp/librccl.so',
                'RCCL_BUILD_DIR': '/tmp/build',
            })
            subprocess.run(['python3', str(WRITER), str(path)], env=env, check=True)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded['workload_fault_kind'], 'H2')
            self.assertEqual(loaded['workload_fault_rank'], 1)
            self.assertEqual(loaded['workload_h2_delta_bytes'], 1024)

    def test_fault_config_schema_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fault-config.json'
            data = {
                'algorithm': 'Ring', 'protocol': 'Simple', 'world_size': 4,
                'fault_kind': 'S2', 'fault_rank': 3, 'fault_first_op': 4,
                'fault_repeats': 3, 'fault_cycles': 50000000,
            }
            path.write_text(json.dumps(data))
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded['fault_rank'], 3)
            self.assertEqual(loaded['fault_first_op'], 4)
            self.assertEqual(loaded['fault_repeats'], 3)


if __name__ == '__main__':
    unittest.main()
