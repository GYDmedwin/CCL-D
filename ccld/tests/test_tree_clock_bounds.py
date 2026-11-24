import asyncio
from functools import partial
import unittest

from ccld.analyzer.clock_bounds import ClockBounds
from ccld.analyzer.tree_wait import tree_phase
from ccld.tools.live_transport import ClockSample, NodeBuffer, sample_clock, serve_pull


class ClockExchangeTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_exchange_contains_the_shared_clock_offset(self):
        buffer = NodeBuffer(1)
        token = b't' * 32
        async with await asyncio.start_server(
                partial(serve_pull, buffer=buffer, token=token), '127.0.0.1', 0) as server:
            host, port = server.sockets[0].getsockname()
            sample = await sample_clock(host, port, token, 1)
            self.assertEqual(sample.epoch, buffer.epoch)
            self.assertEqual(sample.host, buffer.hostname.decode())
            self.assertLessEqual(sample.lower_ns, 0)
            self.assertGreaterEqual(sample.upper_ns, 0)
            with self.assertRaises(asyncio.IncompleteReadError):
                await sample_clock(host, port, b'x' * 32, 1)


class TreeClockTests(unittest.TestCase):
    def setUp(self):
        self.t0 = 10_000_000_000
        self.skew = 80_000_000_000
        self.hosts = {0: 'a', 1: 'b', 2: 'b'}
        self.states = {
            r: {'start_mono_ns': self.t0 + (self.skew if r else 0) + (6_000_000 if r == 2 else 0),
                'mono_ns': self.t0 + 13_000_000 + (self.skew if r else 0)}
            for r in self.hosts
        }
        self.samples = {
            r: [{**self.states[r], 'mono_ns': self.t0 + step + (self.skew if r else 0)}
                for step in (5_000_000, 7_000_000)]
            for r in (0, 1)
        }
        self.clocks = ClockBounds()
        self.clocks.add('node-a', ClockSample('a', b'a', self.t0, -50_000, 50_000))
        self.clocks.add('node-b', ClockSample('b', b'b', self.t0 + self.skew,
                                             -self.skew - 60_000, -self.skew + 60_000))

    def brackets(self):
        return tree_phase(self.states, self.samples, self.hosts, 2, self.clocks)

    def test_cross_host_phase_uses_certain_before_and_after_samples(self):
        boundary = self.brackets()
        self.assertEqual(boundary.limits, {r: tuple(rows) for r, rows in self.samples.items()})
        self.assertEqual(boundary.uncertainty_ns, 272_000)
        self.assertGreater(boundary.tail_ns, 6_000_000)

    def test_cross_host_apparent_boundary_overlap_abstains(self):
        self.samples[0][0]['mono_ns'] = self.t0 + 5_999_999
        boundary = self.brackets()
        self.assertEqual(boundary.admission_rank, 2)
        self.assertTrue(boundary.confirmed_admission)
        self.assertIsNone(boundary.limits)

    def test_cross_host_apparent_entry_order_overlap_does_not_confirm_admission(self):
        self.states[0]['start_mono_ns'] = self.t0 + 4_950_000
        self.samples[0][0]['start_mono_ns'] = self.states[0]['start_mono_ns']
        self.samples[0][1]['start_mono_ns'] = self.states[0]['start_mono_ns']
        boundary = self.brackets()
        self.assertFalse(boundary.confirmed_admission)
        self.assertIsNotNone(boundary.limits)

    def test_all_rank_boundary_can_follow_the_delayed_admission_candidate(self):
        self.states[1]['start_mono_ns'] = self.t0 + self.skew + 6_500_000
        self.samples[2] = [{**self.states[2], 'mono_ns': self.t0 + self.skew + 7_500_000}]
        boundary = self.brackets()
        self.assertFalse(boundary.confirmed_admission)
        self.assertNotIn(1, boundary.limits)
        self.assertIsNone(boundary.limits[2][0])
        self.assertEqual(boundary.limits[2][1], self.samples[2][0])

    def test_missing_or_stale_calibration_abstains(self):
        self.clocks.samples['a'].clear()
        self.clocks.add('node-a', ClockSample('a', b'a', self.t0 - 3_000_000_000, -50_000, 50_000))
        self.assertIsNone(self.brackets())

    def test_conflicting_agent_identity_invalidates_host_clock(self):
        self.clocks.add('other-endpoint', ClockSample('a', b'a', self.t0, -50_000, 50_000))
        self.assertIsNone(self.brackets())
