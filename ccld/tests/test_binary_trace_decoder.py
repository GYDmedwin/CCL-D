import ctypes
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from decode_ccld_binary import Event, RECORD_BYTES, decode_file, decode_record


def event(kind):
    e = Event()
    e.magic, e.version, e.type = 0x444c4343, 1, kind
    e.hash, e.seq, e.nowNs, e.rank, e.nranks = 123, 4, 1000, 0, 8
    return e


class BinaryTraceDecoderTests(unittest.TestCase):
    def test_init_decodes_new_frame_capacity_and_old_records(self):
        e = event(1)
        self.assertEqual(decode_record(bytes(e))['frame_blocks'], 8)
        self.assertEqual(decode_record(bytes(e))['queue_slots'], 128)
        e.reserved, e.bytes, e.windowNs = 512, 2048, 4292608
        self.assertEqual(decode_record(bytes(e))['frame_blocks'], 512)
        self.assertEqual(decode_record(bytes(e))['queue_slots'], 2048)
        self.assertEqual(decode_record(bytes(e))['queue_bytes'], 4292608)

    def test_complete_record_rates_and_corrupt_data(self):
        e = event(4)
        e.startNs, e.count, e.bytes, e.windowNs = 100, 256, 1024, 500
        e.op, e.algorithm, e.protocol, e.datatype = b'AllReduce', b'RING', b'SIMPLE', b'ncclFloat32'
        e.nChannels = 1
        c = e.channels[0]
        c.id, c.parent, c.send, c.recv, c.sendChanges, c.recvChanges = 3, 1, 14, 7, 2, 0
        c.sendDelta, c.recvDelta = 4, 2
        decoded = decode_record(bytes(e))
        self.assertEqual(decoded['duration_ns'], 900)
        self.assertEqual(decoded['channels'][0]['send_rate'], 0.5)
        self.assertIsNone(decoded['channels'][0]['recv_rate'])
        self.assertEqual(decoded['channels'][0]['send_delta_per_ms'], 8000)
        self.assertAlmostEqual(decoded['busbw_gbps'], 1024 * 2 * 7 / 8 / 900)
        e.magic = 0
        with self.assertRaises(ValueError):
            decode_record(bytes(e))
        with self.assertRaises(ValueError):
            decode_record(bytes(e)[:-1])

    def test_p2p_progress_and_coverage_gap_are_explicit(self):
        gap = event(7)
        gap.seq = 0
        gap.reason = b'p2p_not_equal_all_peers'
        decoded = decode_record(bytes(gap))
        self.assertEqual((decoded['event'], decoded['op_count'], decoded['reason']),
                         ('coverage_gap', 0, 'p2p_not_equal_all_peers'))
        complete = event(4)
        complete.startNs, complete.windowNs, complete.nChannels = 100, 50, 1
        complete.op, complete.algorithm, complete.protocol = b'P2PAllPeers', b'P2P', b'LL'
        complete.reason = b'plan_to_event_observation'
        decoded = decode_record(bytes(complete))
        self.assertEqual(decoded['pattern'], 'p2p_all_peers')
        self.assertEqual(decoded['duration_scope'], 'plan_to_event_observation')

    def test_offline_rejects_partial_and_follow_waits_for_complete_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, out = root / 'binary', root / 'json'
            src.mkdir()
            path = src / 'ccld-abc.ccldbin'
            first, second = bytes(event(1)), bytes(event(5))
            path.write_bytes(first + second[:100])
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                decode_file(path, root / 'bad.jsonl')
            proc = subprocess.Popen([sys.executable, str(ROOT / 'tools/decode_ccld_binary.py'),
                                     str(src), str(out), '--follow', '--interval-s', '0.01'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                target = out / 'ccld-abc.jsonl'
                while (not target.exists() or not target.read_text()) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(target.exists())
                with path.open('ab') as stream:
                    stream.write(second[100:])
                deadline = time.monotonic() + 5
                while len(target.read_text().splitlines()) < 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
                proc.send_signal(signal.SIGTERM)
                _, stderr = proc.communicate(timeout=5)
                self.assertEqual(proc.returncode, 0, stderr)
                self.assertEqual([json.loads(line)['event'] for line in target.read_text().splitlines()],
                                 ['comm_init', 'heartbeat'])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        self.assertEqual(ctypes.sizeof(Event), RECORD_BYTES)


if __name__ == '__main__':
    unittest.main()
