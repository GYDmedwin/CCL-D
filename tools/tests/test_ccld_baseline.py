import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('calibrate_ccld_baseline', ROOT / 'calibrate_ccld_baseline.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BaselineCalibrationTests(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertAlmostEqual(module.quantile([1.0, 2.0, 4.0, 8.0], 0.5), 3.0)

    def test_control_trace_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rank.jsonl'
            path.write_text('{"event":"complete","duration_ns":1000000}\n'
                            '{"event":"snapshot","duration_ns":9000000}\n'
                            '{"event":"complete","duration_ns":2000000}\n')
            self.assertEqual(module.durations(Path(directory)), [1.0, 2.0])

    def test_margin_keeps_baseline_under_control_lower_envelope(self):
        self.assertLess(module.quantile([0.3, 0.4, 0.5], 0.05) * 0.95, 0.3)


if __name__ == '__main__':
    unittest.main()
