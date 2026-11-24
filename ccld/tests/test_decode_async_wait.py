import sys
import json
import pickle
import struct
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from decode_async_wait import Event, WaitDecoder, follow


def event(kind):
    e = Event()
    e.magic, e.version, e.type = 0x444c4343, 1, kind
    e.hash, e.seq, e.nowNs, e.rank, e.nranks, e.nChannels = 123, 4, 1000, 0, 8, 1
    e.channels[0].id = 2
    return e


def fragments(channel_count, wait=False):
    total = (channel_count + 31) // 32
    for index in range(total):
        progress = event(4)
        progress.version = 2
        progress.reserved = (total << 16) | (index << 8) | 4
        progress.op, progress.algorithm, progress.protocol = b'AllReduce', b'RING', b'SIMPLE'
        progress.windowNs = 200
        progress.nChannels = min(32, channel_count - index * 32)
        for offset in range(progress.nChannels):
            cid = index * 32 + offset
            progress.channels[offset].id = cid
            progress.channels[offset].send = cid + 1
        if wait:
            waiting = Event.from_buffer_copy(bytes(progress))
            waiting.type = 7
            progress.reason = b'ring_wait_v1'
            for offset in range(waiting.nChannels):
                waiting.channels[offset].parent = 7
                waiting.channels[offset].sendChanges = 1
                waiting.channels[offset].recv = 40
            yield waiting
        yield progress


class WaitDecoderTests(unittest.TestCase):
    def test_tree_wait_pairs_preserve_independent_child_counters(self):
        progress = event(4)
        progress.algorithm, progress.protocol = b'Tree', b'Simple'
        progress.reason, progress.windowNs = b'tree_wait_v1', 100
        progress.channels[0].parent = -1
        side = event(10)
        side.reserved = 4
        payload = bytearray(bytes(side))
        row = struct.Struct('<5i9I')
        row.pack_into(payload, Event.channels.offset, 2, -1, 1, 2, -1, 0,
                      0, 37, 50, 0, 0, 3, 4, 0)
        decoder = WaitDecoder(defer_channels=True)
        self.assertIsNone(decoder.consume(payload))
        output = decoder.consume(bytes(progress))
        decoder.finish()
        self.assertTrue(output['tree_wait_probe'])
        self.assertFalse(output['wait_probe'])
        metrics = pickle.loads(pickle.dumps(output['channels']))
        self.assertEqual(metrics[0]['tree_wait_peers'], [
            {'rank': 1, 'data_spins': 37, 'credit_spins': 3},
            {'rank': 2, 'data_spins': 50, 'credit_spins': 4},
        ])

        decoder.consume(payload)
        progress.channels[0].parent = 3
        with self.assertRaisesRegex(ValueError, 'inconsistent Tree-wait pair'):
            decoder.consume(bytes(progress))

    def test_fragmented_tree_progress_waits_for_the_last_peer_record(self):
        decoder = WaitDecoder(defer_channels=True)
        row = struct.Struct('<5i9I')
        for progress in fragments(33):
            progress.algorithm, progress.reason = b'Tree', b'tree_wait_v1'
            wait = Event.from_buffer_copy(bytes(progress))
            wait.type = 10
            payload = bytearray(bytes(wait))
            for index in range(progress.nChannels):
                cid = progress.channels[index].id
                progress.channels[index].parent = 4
                row.pack_into(payload, Event.channels.offset + index * row.size,
                              cid, 4, 1, 2, -1, 0, cid, 1, 2, 0, 0, 3, 4, 0)
            self.assertIsNone(decoder.consume(payload))
            result = decoder.consume(bytes(progress))
            if progress.nChannels == 32:
                self.assertIsNone(result)
        decoder.finish()
        self.assertEqual(result['n_channels'], 33)
        self.assertEqual(result['channels'][-1]['tree_wait_peers'][0],
                         {'rank': 4, 'data_spins': 32, 'credit_spins': 0})

    def test_progress_timestamp_requires_the_feature_flag_and_a_valid_interval(self):
        progress = event(4)
        progress.startNs = (1 << 32) + 20
        progress.nowNs = progress.startNs + 500
        progress.windowNs = 500
        progress.frameBytes, progress.frameChannels = 300, 1
        key = 'first_progress_observed_mono_ns'
        self.assertNotIn(key, WaitDecoder().consume(bytes(progress)))
        progress.mode = 1 << 31
        self.assertEqual(WaitDecoder().consume(bytes(progress))[key], (1 << 32) + 300)
        progress.frameBytes = 10
        with self.assertRaisesRegex(ValueError, 'outside the observed operation'):
            WaitDecoder().consume(bytes(progress))

    def test_wide_progress_requires_all_parts_and_preserves_wait_pairs(self):
        for count, waits in ((33, False), (65, True), (128, True)):
            with self.subTest(count=count, waits=waits):
                records = list(fragments(count, waits))
                decoder = WaitDecoder(defer_channels=True)
                for record in records[:-1]:
                    self.assertIsNone(decoder.consume(bytes(record)))
                data = decoder.consume(bytes(records[-1]))
                decoder.finish()
                self.assertEqual(data['n_channels'], count)
                self.assertIsNone(data['channels']._decoded)
                self.assertEqual(list(data['channels'].parents()), list(range(count)))
                self.assertEqual(data['channels'][-1]['send_count'], count)
                if waits:
                    self.assertEqual(data['channels'][-1]['wait_credit_spins'], 40)

    def test_partial_or_inconsistent_fragments_have_no_completion(self):
        first, last = fragments(33)
        decoder = WaitDecoder()
        self.assertIsNone(decoder.consume(bytes(first)))
        with self.assertRaisesRegex(ValueError, 'incomplete final progress'):
            decoder.finish()
        last.channels[0].id = 31
        with self.assertRaisesRegex(ValueError, 'invalid channels'):
            decoder.consume(bytes(last))
        with self.assertRaisesRegex(ValueError, 'out of order'):
            WaitDecoder().consume(bytes(last))
        decoder = WaitDecoder()
        decoder.consume(bytes(first))
        last.count += 1
        with self.assertRaisesRegex(ValueError, 'inconsistent progress'):
            decoder.consume(bytes(last))

    def test_deferred_ring_wait_metrics_match_eager_after_pickling(self):
        wait, done = event(7), event(4)
        wait.reserved = 4
        wait.channels[0].parent, wait.channels[0].sendChanges = 7, 1
        wait.channels[0].send, wait.channels[0].recv = 37, 42
        done.reason, done.windowNs = b'ring_wait_v1', 100
        done.op, done.algorithm, done.protocol = b'AllReduce', b'RING', b'SIMPLE'
        done.channels[0].recvChanges = 4

        def joined(defer):
            decoder = WaitDecoder(defer_channels=defer)
            decoder.consume(bytes(wait))
            result = decoder.consume(bytes(done))
            decoder.finish()
            return result

        deferred = pickle.loads(pickle.dumps(joined(True)))
        self.assertEqual(deferred['channels'].parents(), {2: 0})
        self.assertIsNone(deferred['channels']._decoded)
        deferred['channels'] = list(deferred['channels'])
        self.assertEqual(deferred, joined(False))

        decoder = WaitDecoder(defer_channels=True)
        decoder.consume(bytes(wait))
        done.channels[0].id = 3
        with self.assertRaisesRegex(ValueError, 'inconsistent Ring-wait pair'):
            decoder.consume(bytes(done))

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

    def test_follow_drains_files_already_visited_when_stop_arrives(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'bin', Path(tmp) / 'json'
            src.mkdir()
            first = src / 'ccld-a.ccldbin'
            last = src / 'ccld-b.ccldbin'
            first.write_bytes(bytes(event(2)))
            marker = event(5)
            marker.rank = 1
            last.write_bytes(bytes(marker))
            called = False
            stop = None

            def signal_handler(_signum, handler):
                nonlocal stop
                stop = handler

            class EndingDecoder(WaitDecoder):
                def consume(self, raw):
                    nonlocal called
                    result = super().consume(raw)
                    if result['rank'] == 1 and not called:
                        called = True
                        entry = event(2)
                        entry.seq = 5
                        with first.open('ab') as out:
                            out.write(bytes(entry))
                        stop(signal.SIGTERM, None)
                    return result

            with patch('decode_async_wait.signal.signal', side_effect=signal_handler):
                follow(src, dst, 0, EndingDecoder)
            self.assertEqual([json.loads(line)['op_count'] for line in
                              (dst/'ccld-a.jsonl').read_text().splitlines()], [4, 5])


if __name__ == '__main__':
    unittest.main()
