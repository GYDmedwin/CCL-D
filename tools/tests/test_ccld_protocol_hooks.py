#!/usr/bin/env python3
import os
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
NV_ROOT = os.environ.get("CCLD_NV_ROOT")


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

    @unittest.skipUnless(NV_ROOT, "set CCLD_NV_ROOT to the NCCL checkout for source parity checks")
    def test_nccl_reference_waits_and_tree_bindings(self):
        root = pathlib.Path(NV_ROOT)
        nv = (root / "src/device/prims_ll.h").read_text()
        self.assertIn("if (SEND) waitSend(divUp(nelem, EltPerLine)*sizeof(ncclLLFifoLine));", nv)
        self.assertIn("if (RECV) {\n      for (int i=0; i < MaxRecv; i++) incRecv(i);", nv)
        self.assertIn("const bool track = tid == 0 && ccldWaitChannel;", nv)
        simple = (root / "src/device/prims_simple.h").read_text()
        self.assertRegex(simple, r"const bool track = [^;]*ccldWaitChannel[^;]*index == 0")
        tree = (root / "src/device/all_reduce.h").read_text()
        start = tree.index("void runTreeUpDown")
        updown = tree[start:tree.index("template<typename T, typename RedOp>", start)]
        self.assertGreaterEqual(updown.count("work->redOpArg, 0, 0, 0, work"), 2)
        start = tree.index("void runTreeSplit")
        split = tree[start:tree.index("template<typename T, typename RedOp>", start)]
        self.assertGreaterEqual(split.count("work->redOpArg"), 3)
        self.assertGreaterEqual(split.count("work)"), 3)

    def test_async_writer_is_only_sink_for_telemetry(self):
        source = (ROOT / "src/misc/ccld.cc").read_text()
        log = source[source.index("static void ccldWrite"):source.index("static void ccldWriter")]
        self.assertIn("logQueue->pushBatch(events, count)", log)
        self.assertNotIn("fwrite", log)
        self.assertIn("CcldQueue<CcldLogEvent>", source)
        self.assertIn("state->writer = std::thread(ccldWriter, state)", source)
        self.assertIn("state->writer.join()", source)

    def test_socket_and_file_are_writer_sinks(self):
        source = (ROOT / "src/misc/ccld.cc").read_text()
        writer = source[source.index("static void ccldWriter"):source.index("static uint64_t ccldLoad")]
        self.assertIn("fwrite(batch, sizeof(CcldLogEvent), count, state->output)", writer)
        self.assertIn("state->nodeStream.write(batch, count * sizeof(CcldLogEvent))", writer)
        self.assertIn('"NCCL_CCLD_NODE_SOCKET"', source)
        self.assertIn('"binary_queue_overflow"', writer)

    def test_gfx936_full_socket_roundtrip_gate_exists(self):
        script = (ROOT / "tools/tests/run_ccld_gfx936_socket_sink.sh").read_text()
        self.assertIn("NCCL_CCLD_NODE_SOCKET", script)
        self.assertIn("decode_ccld_binary.py", script)
        self.assertIn("comm_end", script)

    def test_wait_probe_hooks_use_lead_threads(self):
        simple = (ROOT / "src/device/prims_simple.h").read_text()
        ll = (ROOT / "src/device/prims_ll.h").read_text()
        self.assertIn("const bool track = ccldWaitChannel != nullptr && index == 0;", simple)
        self.assertIn("const bool track = tid == 0 && ccldWaitChannel != nullptr;", ll)
        self.assertIn("NCCL_CCLD_WAIT_ENABLE", (ROOT / "src/misc/ccld.cc").read_text())

    def test_tree_primitive_work_bindings(self):
        rccl = (ROOT / "src/device/all_reduce.h").read_text()
        rccl_updown = rccl[rccl.index("void runTreeUpDown"):rccl.index("void runTreeSplit")]
        self.assertGreaterEqual(rccl_updown.count("work->redOpArg, 0, 0, 0, work"), 2)
        rccl_split = rccl[rccl.index("void runTreeSplit"):rccl.index("template<typename T, typename RedOp>", rccl.index("void runTreeSplit"))]
        self.assertGreaterEqual(rccl_split.count("work->redOpArg"), 3)
        self.assertGreaterEqual(rccl_split.count("work)"), 3)

    def test_fault_injection_isolated_and_nv_mapped(self):
        primitive = (ROOT / "src/device/primitives.h").read_text()
        simple = (ROOT / "src/device/prims_simple.h").read_text()
        ll = (ROOT / "src/device/prims_ll.h").read_text()
        work = (ROOT / "src/include/device.h").read_text()
        host = (ROOT / "src/misc/ccld.cc").read_text()
        cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn("#ifdef CCLD_FAULT_INJECTION", primitive)
        self.assertIn("ccldInject(ccldFaultKind, ccldFaultCycles, tid, ccldHangDone)", simple)
        self.assertIn("ccldInject(ccldFaultKind, ccldFaultCycles, tid, ccldHangDone)", ll)
        self.assertIn("uint64_t ccldFaultCycles", work)
        self.assertIn('"NCCL_CCLD_FAULT_KIND"', host)
        self.assertIn("CCLD_FAULT_INJECTION", cmake)
        self.assertIn('option(CCLD_FAULT_INJECTION', cmake)
        self.assertIn('"Enable isolated CCL-D H3/S2 fault simulation"  OFF)', cmake)
        fault_script = (ROOT / "tools/tests/run_ccld_gfx936_fault_injection.sh").read_text()
        self.assertIn("CCLD_FAULT_INJECTION=ON", fault_script)
        self.assertIn("NCCL_CCLD_FAULT_KIND=S2", fault_script)


if __name__ == "__main__":
    unittest.main()
