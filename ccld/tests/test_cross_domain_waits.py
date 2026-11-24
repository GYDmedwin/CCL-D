import asyncio
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from ccld.analyzer.cross_domain import CrossDomainWaits
from ccld.tools.decode_async_wait import WaitDecoder
from ccld.tools.live_transport import NodeBuffer, receive_rank
from ccld.tools.wait_memory import WAIT_ROW, WaitPage, WaitWriter, new_page_fd
from ccld.tools.wait_observation import evidence_record, receive_wait


def group(op, missing=(), size=2):
    remaining = {rank: {'event': 'snapshot'} for rank in range(size) if rank not in missing}
    return SimpleNamespace(next_op=op, world=size, gap=False, ended=set(),
                           observed={rank: op - (rank in missing) for rank in range(size)},
                           last_hang=(op, 'H1', tuple(missing)) if missing else None,
                           pending={op: remaining})


def observed(consumer, producer, size, wait_id=1, active=True):
    return WaitDecoder().consume(evidence_record(consumer, producer, size, wait_id, 1 if active else 2))


class CrossDomainWaitTests(unittest.TestCase):
    def setUp(self):
        self.analysis = CrossDomainWaits()
        self.out = []

    def test_cross_step_links_follow_the_known_wait_chain(self):
        comms = {'0000000000000001': group(2),
                 '0000000000000002': group(4, [1]),
                 '0000000000000003': group(5, [1]),
                 '0000000000000004': group(5, [1])}
        self.analysis.add(observed((2,4,1), (1,2,0), 2))
        self.analysis.add(observed((3,5,1), (2,4,0), 2))
        self.analysis.evaluate(comms, self.out.append)
        self.assertEqual([item['comm_hash'] for item in self.out],
                         ['0000000000000002','0000000000000003'])
        self.assertEqual(self.out[1]['upstream_candidates'],
                         [{'comm_hash': '0000000000000001', 'op_count': 2}])

    def test_multiple_supported_predecessors_are_all_reported(self):
        comms = {'0000000000000001': group(4),
                 '0000000000000002': group(6),
                 '0000000000000003': group(9, [0,1,2], size=4)}
        self.analysis.add(observed((3,9,0), (1,4,0), 4))
        self.analysis.add(observed((3,9,1), (2,6,0), 4))
        self.analysis.evaluate(comms, self.out.append)
        self.assertEqual(len(self.out[0]['upstream_candidates']), 2)
        self.assertEqual(self.out[0]['unresolved_ranks'], [2])

    def test_completed_or_uncovered_producer_clears_the_attribution(self):
        for ended in ('completed', 'gap', 'closed_wait'):
            with self.subTest(ended=ended):
                analysis = CrossDomainWaits()
                out = []
                comms = {'0000000000000001': group(2), '0000000000000002': group(4, [1])}
                analysis.add(observed((2,4,1), (1,2,0), 2))
                analysis.evaluate(comms, out.append)
                if ended == 'completed':
                    comms['0000000000000001'].pending[2][0]['event'] = 'complete'
                elif ended == 'gap':
                    comms['0000000000000001'].gap = True
                else:
                    analysis.add(observed((2,4,1), (1,2,0), 2, active=False))
                analysis.evaluate(comms, out.append)
                self.assertEqual([item['kind'] for item in out], ['witnessed_wait', 'attribution_cleared'])

    def test_pull_exports_active_scopes_and_retires_closed_rank_bindings(self):
        buffer = NodeBuffer(16)
        process = (1, 'birth')
        identity = (1, 0, 2)
        buffer.bind_rank(process, identity)
        buffer.waits[1] = (process, [((2,4,1), (1,2,0), 2, 1)])
        buffer.waits.pop(1)
        buffer.capture_waits()
        self.assertEqual(buffer.end, 0)
        buffer.waits[2] = (process, [((2,4,1), (1,2,0), 2, 2)])
        buffer.capture_waits()
        buffer.capture_waits()
        self.assertEqual(buffer.end, 1)
        self.assertTrue(WaitDecoder().consume(buffer.records[0])['active'])
        buffer.unbind_rank(process, identity)
        buffer.capture_waits()
        self.assertEqual(buffer.end, 2)
        self.assertFalse(WaitDecoder().consume(buffer.records[1])['active'])
        self.assertFalse(buffer.waits)

    def test_shared_wait_requires_current_same_process_bindings(self):
        buffer = NodeBuffer(16)
        owner = (11, 'birth')
        producer = (1, 0, 2)
        consumer = (2, 1, 2)
        with patch('ccld.tools.live_transport.time.monotonic_ns', return_value=10):
            buffer.bind_rank(owner, producer)
            buffer.bind_rank((12, 'another'), consumer)
        page = Mock(snapshot=lambda: WAIT_ROW.pack(1, 11, 1, 2, 0, 2, 3, 1))
        buffer.wait_pages[owner] = page
        buffer.capture_waits()
        self.assertEqual(buffer.end, 0)
        with patch('ccld.tools.live_transport.time.monotonic_ns', return_value=12):
            buffer.bind_rank(owner, consumer)
        buffer.capture_waits()
        self.assertEqual(buffer.end, 0)
        page.snapshot = lambda: WAIT_ROW.pack(2, 13, 1, 2, 0, 2, 3, 1)
        buffer.capture_waits()
        self.assertEqual(buffer.end, 1)
        with patch('ccld.tools.live_transport.time.monotonic_ns', return_value=14):
            buffer.unbind_rank(owner, producer)
            buffer.bind_rank(owner, producer)
        buffer.capture_waits()
        buffer.capture_waits()
        self.assertEqual(buffer.end, 2)
        self.assertFalse(WaitDecoder().consume(buffer.records[1])['active'])


class LocalWaitRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_endpoints_from_different_processes(self):
        buffer = NodeBuffer(16)
        buffer.bind_rank((11, 'started'), (1, 0, 2))
        buffer.bind_rank((12, 'started'), (2, 0, 2))
        reader = asyncio.StreamReader()
        reader.feed_data(json.dumps({'command': 'wait', 'producer': ['0000000000000001', 3, 0],
                                     'consumers': [['0000000000000002', 4, 0]]}).encode() + b'\n')
        reader.feed_eof()
        responses = []
        writer = SimpleNamespace(write=responses.append, drain=AsyncMock(),
                                 close=Mock(), wait_closed=AsyncMock())
        with patch('ccld.tools.wait_observation.peer_process', return_value=(11, 'started')):
            await receive_wait(reader, writer, buffer)
        self.assertIn('error', json.loads(responses[0]))
        self.assertFalse(buffer.waits)
        self.assertEqual(buffer.active, 0)


@unittest.skipUnless(sys.platform == 'linux', 'shared wait sockets use Linux peer credentials')
class SharedWaitIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_cleared_page_can_be_reused_without_exposing_old_scope(self):
        fd = new_page_fd()
        self.addCleanup(os.close, fd)
        page, reader = WaitPage(fd), WaitPage(fd)
        self.addCleanup(page.close)
        self.addCleanup(reader.close)
        writer = WaitWriter(page, None)
        producer = (1, 2, 0)
        first = writer.begin(producer, ((2, 3, 0),))
        self.assertEqual(WAIT_ROW.unpack(reader.snapshot())[0], first)
        writer.end(first)
        self.assertEqual(reader.snapshot(), b'')
        second = writer.begin(producer, ((2, 4, 0),))
        row = WAIT_ROW.unpack(reader.snapshot())
        self.assertEqual((row[0], row[6]), (second, 4))

    async def test_scopes_are_local_and_process_exit_retires_evidence(self):
        child_code = """
import os, socket, sys, time
from ccld.tools.decode_ccld_binary import Event
from ccld.tools.wait_observation import Operation, local_communicators, observe_wait

path = sys.argv[1]
streams = [socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) for _ in range(3)]
for index, stream in enumerate(streams, 1):
    stream.connect(path)
    event = Event()
    event.magic, event.version, event.type = 0x444c4343, 1, 1
    event.hash, event.rank, event.nranks = index, 0, 2
    stream.sendall(bytes(event))
for attempt in range(50):
    if len(local_communicators(path)) == 3:
        break
    time.sleep(.01)
else:
    raise RuntimeError('missing rank bindings')
producer = Operation('0000000000000001', 2, 0)
first = Operation('0000000000000002', 3, 0)
second = Operation('0000000000000003', 4, 0)
with observe_wait(producer, [first], path) as recorded:
    assert recorded
    forked = os.fork()
if forked == 0:
    os._exit(0)
assert os.waitpid(forked, 0)[1] == 0
with observe_wait(producer, [first], path) as recorded:
    assert recorded
    try:
        with observe_wait(producer, [second], path) as nested:
            assert nested
            print('two waits', flush=True)
            assert sys.stdin.readline()
            raise ValueError('stop nested wait')
    except ValueError:
        pass
    print('one wait', flush=True)
    sys.stdin.readline()
"""
        buffer = NodeBuffer(16)
        with TemporaryDirectory(prefix='ccld-shared-test-') as directory:
            path = str(Path(directory) / 'rank')
            ranks = await asyncio.start_unix_server(lambda r, w: receive_rank(r, w, buffer), path)
            waits = await asyncio.start_unix_server(lambda r, w: receive_wait(r, w, buffer), path+'.wait')
            async with ranks, waits:
                child = await asyncio.create_subprocess_exec(
                    sys.executable, '-c', child_code, path,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
                try:
                    self.assertEqual(await asyncio.wait_for(child.stdout.readline(), 5), b'two waits\n')
                    buffer.capture_waits()
                    self.assertEqual(buffer.end, 5)
                    child.stdin.write(b'close nested\n')
                    await child.stdin.drain()
                    self.assertEqual(await asyncio.wait_for(child.stdout.readline(), 5), b'one wait\n')
                    buffer.capture_waits()
                    self.assertEqual(buffer.end, 6)
                finally:
                    if child.returncode is None:
                        child.kill()
                    await asyncio.wait_for(child.wait(), 5)
                for _ in range(100):
                    if not buffer.wait_pages and not buffer.active:
                        break
                    await asyncio.sleep(.01)
                self.assertFalse(buffer.wait_pages)
                buffer.capture_waits()
                decoder = WaitDecoder()
                records = [decoder.consume(record) for record in buffer.records[:buffer.end]]
                events = [event for event in records if event['event'] == 'wait_observation']
                exits = [event for event in records if event['event'] == 'rank_stream_lost']
                self.assertEqual(len(exits), 3)
                self.assertTrue(all(event['process_exited'] for event in exits))
                self.assertEqual([event['active'] for event in events], [True, True, False, False])
                self.assertEqual(events[0]['wait_id'], events[-1]['wait_id'])
                self.assertEqual(events[1]['wait_id'], events[2]['wait_id'])


if __name__ == '__main__':
    unittest.main()
