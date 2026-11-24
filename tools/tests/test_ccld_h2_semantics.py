#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

WORKLOAD = Path(__file__).with_name('ccld_torch_collectives_rccl.py')
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('PyTorch is required for the workload tests')
spec = importlib.util.spec_from_file_location('ccld_torch_collectives_rccl', WORKLOAD)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class H2SemanticsTests(unittest.TestCase):
    def test_default_matches_nv_quarter_count(self):
        self.assertEqual(module.h2_injected_count(262144), 65536)

    def test_small_count_matches_nv_minimum_count(self):
        self.assertEqual(module.h2_injected_count(3), 1)

    def test_divisor_is_applied_to_actual_numel(self):
        self.assertEqual(module.h2_injected_count(100, 5), 20)

    def test_operation_mismatch_targets_only_selected_rank(self):
        self.assertEqual(module.h2_operation_for_rank('allreduce', True, 1, 1, 'operation', 'reducescatter'), 'reducescatter')
        self.assertEqual(module.h2_operation_for_rank('allreduce', True, 0, 1, 'operation', 'reducescatter'), 'allreduce')


if __name__ == '__main__':
    unittest.main()
