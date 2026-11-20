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

    def test_async_writer_is_only_sink_for_telemetry(self):
        source = (ROOT / "src/misc/ccld.cc").read_text()
        log = source[source.index("static void ccldWrite"):source.index("static void ccldWriter")]
        self.assertIn("logQueue->push(event)", log)
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


if __name__ == "__main__":
    unittest.main()
