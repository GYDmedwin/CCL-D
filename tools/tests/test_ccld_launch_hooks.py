import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ENQUEUE = (ROOT / "src/enqueue.cc").read_text()
CCLD = (ROOT / "src/misc/ccld.cc").read_text()


class CcldLaunchHookTest(unittest.TestCase):
    def test_each_successful_launch_path_enqueues_completion(self):
        start = ENQUEUE.index("ncclResult_t ncclLaunchKernel(struct ncclComm* comm")
        end = ENQUEUE.index("ncclResult_t ncclLaunchKernelAfter_NoCuda", start)
        launch = ENQUEUE[start:end]
        paths = (
            "hipExtLaunchKernel",
            "cuLaunchKernelEx",
            "cudaLaunchKernel(sym",
        )
        for path in paths:
            with self.subTest(path=path):
                pos = launch.index(path)
                hook = launch.index("ccldCallbackEnqueue(comm, plan, launchStream)", pos)
                success_return = launch.index("return ncclSuccess;", hook)
                self.assertLess(pos, hook)
                self.assertLess(hook, success_return)

    def test_event_and_hostfunc_completion_are_implemented(self):
        self.assertIn("hipEventRecord(slot.completionEvent, stream)", CCLD)
        self.assertIn("hipEventQuery(slot.completionEvent)", CCLD)
        self.assertIn("hipLaunchHostFunc(stream, ccldDone, callback)", CCLD)


if __name__ == "__main__":
    unittest.main()
