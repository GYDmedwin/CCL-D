import sys
import json
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from decode_async_wait import Event, WaitDecoder


def event(kind):
    e = Event()
    e.magic, e.version, e.type = 0x444c4343, 1, kind
    e.hash, e.seq, e.nowNs, e.rank, e.nranks, e.nChannels = 123, 4, 1000, 0, 8, 1
    e.channels[0].id = 2
    return e


class WaitDecoderTests(unittest.TestCase):
    def test_wait_pair_join_and_missing_or_mismatched_pairs_fail(self):
        wait, done = event(7), event(4)
        wait.reserved = 4
        wait.channels[0].parent, wait.channels[0].sendChanges = 7, 1
        wait.channels[0].send, wait.channels[0].recv = 37, 42
        done.reason, done.windowNs = b'ring_wait_v1', 100
        done.op, done.algorithm, done.protocol = b'AllReduce', b'RING', b'SIMPLE'
        decoder = WaitDecoder()
        self.assertIsNone(decoder.consume(bytes(wait)))
        result = decoder.consume(bytes(done))
        self.assertTrue(result['wait_probe'])
        self.assertEqual(result['channels'][0]['ring_prev'], 7)
        self.assertEqual(result['channels'][0]['ring_next'], 1)
        self.assertEqual(result['channels'][0]['wait_data_spins'], 37)
        self.assertEqual(result['channels'][0]['wait_credit_spins'], 42)
        decoder.finish()
        with self.assertRaisesRegex(ValueError, 'missing'):
            decoder.consume(bytes(done))
        decoder.consume(bytes(wait))
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            decoder.finish()

    def test_invalid_neighbors_and_ordinary_record(self):
        wait = event(7)
        wait.reserved = 3
        wait.channels[0].parent, wait.channels[0].sendChanges = 8, 1
        with self.assertRaisesRegex(ValueError, 'neighbor'):
            WaitDecoder().consume(bytes(wait))
        init = event(1)
        init.count = 3072
        init.nChannels = 0
        data = WaitDecoder().consume(bytes(init))
        self.assertEqual(data['wait_frame_bytes'], 3072)
        self.assertTrue(data['wait_requested'])

    def test_follow_waits_for_the_paired_record(self):
        wait, done = event(7), event(4)
        wait.reserved = 4
        wait.channels[0].parent, wait.channels[0].sendChanges = 7, 1
        wait.channels[0].send, wait.channels[0].recv = 37, 42
        done.reason, done.windowNs = b'ring_wait_v1', 100
        done.op, done.algorithm, done.protocol = b'AllReduce', b'RING', b'SIMPLE'
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'bin', Path(tmp) / 'json'
            src.mkdir()
            path = src / 'ccld-fixture.ccldbin'
            path.write_bytes(bytes(wait))
            proc = subprocess.Popen([sys.executable, str(ROOT / 'tools/decode_async_wait.py'),
                                     str(src), str(dst), '--follow', '--interval-s', '0.01'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                target = dst / 'ccld-fixture.jsonl'
                deadline = time.monotonic() + 5
                while not target.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(target.read_text(), '')
                with path.open('ab') as out:
                    out.write(bytes(done))
                deadline = time.monotonic() + 5
                while not target.read_text() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(json.loads(target.read_text())['channels'][0]['wait_data_spins'], 37)
                proc.send_signal(signal.SIGTERM)
                _, stderr = proc.communicate(timeout=5)
                self.assertEqual(proc.returncode, 0, stderr)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    unittest.main()
