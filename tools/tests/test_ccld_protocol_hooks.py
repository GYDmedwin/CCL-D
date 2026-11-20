#!/usr/bin/env python3
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
NV = pathlib.Path("/workspace/nv/CCL-D")


class LlCounterHooksTest(unittest.TestCase):
    def test_send_counter_follows_credit_wait(self):
        source = (ROOT / "src/device/prims_ll.h").read_text()
        wait = "if (SEND) waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine));"
        count = "if (SEND) ccldCount(ccldChannel, tid, group, true, false);"
        self.assertLess(source.index(wait), source.index(count))
        self.assertIn("ccldChannel(ccldOutput(e, tid, group))", source)

    def test_receive_counter_follows_ll_operation_and_post(self):
        source = (ROOT / "src/device/prims_ll.h").read_text()
        receive = source.index("if (RECV) {\n      for (int i=0; i < MaxRecv; i++) incRecv(i);")
        post = source.index("postRecv();", receive)
        count = source.index("ccldCount(ccldChannel, tid, group, false, true);", receive)
        self.assertLess(post, count)

    def test_plan_keeps_ll128_out_of_gfx936_counters(self):
        source = (ROOT / "src/misc/ccld.cc").read_text()
        self.assertIn('ccldDropped(state, plan->ccldSeq, "ll128_unsupported_gfx936");', source)
        self.assertIn("plan->ccldProtocol != NCCL_PROTO_SIMPLE && plan->ccldProtocol != NCCL_PROTO_LL", source)
        self.assertIn("const bool ringCollective = plan->ccldAlgorithm == NCCL_ALGO_RING", source)
        self.assertIn("const bool treeAllReduce = plan->ccldAlgorithm == NCCL_ALGO_TREE", source)

    def test_reference_module_is_ncccl_ccld_ll(self):
        nv = (NV / "src/device/prims_ll.h").read_text()
        self.assertIn("if (SEND) waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine));", nv)
        self.assertIn("if (RECV) {\n      for (int i=0; i < MaxRecv; i++) incRecv(i);", nv)


if __name__ == "__main__":
    unittest.main()
