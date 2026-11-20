import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from ccld.analyzer.cross_domain import CrossDomainWaits
from ccld.tools.decode_async_wait import WaitDecoder
from ccld.tools.live_transport import NodeBuffer
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
        buffer.process_streams[process][identity] = 1
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


class LocalWaitRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_endpoints_from_different_processes(self):
        buffer = NodeBuffer(16)
        buffer.process_streams[(11, 'started')][(1, 0, 2)] = 1
        buffer.process_streams[(12, 'started')][(2, 0, 2)] = 1
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


if __name__ == '__main__':
    unittest.main()
