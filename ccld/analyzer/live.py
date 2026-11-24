"""Pull CCL-D events from training nodes and diagnose collectives in memory."""
import argparse
import asyncio
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
import heapq
import json
import math
import signal
import sys
import time

from .ccld_analyze import (acknowledged_exclusion, add_decision_arguments, aligned_exclusion, analyze_comm, barrier,
                           completed_mismatch, decision_config, exclusion_report,
                           result_common, SlowWindows)
from .clock_bounds import ClockBounds
from .cross_domain import CrossDomainWaits
from .tree_s1_vote import CLASSIFIED, vote
from ..tools.decode_async_wait import RECORD_BYTES, WaitDecoder
from ..tools.live_transport import job_token, pull, sample_clock


class SlowWindow(SlowWindows):
    def flush(self):
        tree_states, tree_decisions = defaultdict(dict), defaultdict(list)
        for key, label, states, item in self.decisions():
            if label.startswith('tree:') and item['kind'] in CLASSIFIED:
                tree_states[key][item['op_count']] = states
                tree_decisions[key].append(item)
                item = dict(item, kind='group_candidate', group_kind=item['kind'],
                            root_scope='tree_layer', candidate_ranks=item['roots'], roots=[])
            self.emit(item)
        for key, decisions in tree_decisions.items():
            outcome = self.tree_decision(tree_states[key], decisions)
            if 'operation_profile' in decisions[0]:
                outcome['operation_profile'] = decisions[0]['operation_profile']
            self.emit(outcome)

    def tree_decision(self, states_by_op, decisions):
        ops = sorted(states_by_op)
        op = ops[0] if len(ops) == 1 else None
        item = result_common(self.comm, op, 'slow')
        item.update(kind='undetermined', roots=[], root_scope='communicator', window=self.window,
                    method='tree_s1_channel_vote')
        if op is None:
            item.update(operation_counts=ops, status='incomplete', reason='Tree groups selected different operations')
            return item
        started = time.perf_counter_ns()
        try:
            outcome = vote(states_by_op[op], decisions)
        except ValueError as exc:
            item.update(status='incomplete', reason=str(exc))
            return item
        item.update(outcome)
        item['location_latency_ms'] = (time.perf_counter_ns() - started) / 1e6
        if outcome['status'] == 'candidate':
            item['kind'] = 'S1'
        return item


class Communicator:
    MAX_PENDING = 2048
    MAX_PHASE_OPERATIONS = 16
    MAX_PHASE_SNAPSHOTS = 32

    def __init__(self, comm, world, cfg, emit, clocks=None):
        self.comm, self.world, self.cfg, self.emit = comm, world, cfg, emit
        self.clocks = clocks
        self.observed = {}
        self.rank_hosts = {}
        self.ended = set()
        self.rank_losses = {}
        self.pending = defaultdict(dict)
        self.phase_samples = OrderedDict() if cfg.ring_wait_phases or cfg.tree_cross_host_phases else None
        self.completed_by_time = []
        self.tree_references = deque(maxlen=4)
        self.rank0_entry_ns = 0
        self.next_op = 1
        self.last_hang = None
        self.window = SlowWindow(comm, cfg, emit)
        self.gap = False

    def mark_gap(self, op, reason, **fields):
        if self.gap:
            return
        self.gap = True
        self.pending.clear()
        if self.phase_samples is not None:
            self.phase_samples.clear()
        self.completed_by_time.clear()
        self.tree_references.clear()
        self.window.profiles.clear()
        item = {'category': 'coverage_gap', 'kind': 'undetermined', 'roots': [],
                'comm_hash': self.comm, 'op_count': op, 'reason': reason}
        self.emit(dict(item, **fields))

    def add(self, event):
        rank, op, kind = event['rank'], event['op_count'], event['event']
        if event['world_size'] != self.world:
            self.mark_gap(op, 'inconsistent communicator size')
            return
        if kind == 'comm_init':
            if rank in self.observed:
                self.mark_gap(op, 'duplicate rank stream for a communicator')
            self.observed[rank] = 0
            self.rank_hosts[rank] = event.get('host')
        elif rank not in self.observed or rank in self.ended:
            self.mark_gap(op, 'event outside an initialized rank stream')
            return
        if kind == 'rank_stream_lost':
            self.ended.add(rank)
            if not self.gap:
                self.rank_losses[rank] = event
            return
        if kind == 'comm_end':
            self.ended.add(rank)
        self.observed[rank] = max(self.observed[rank], op)
        if self.gap:
            return
        if kind == 'dropped':
            states = self.pending.get(op)
            skipped = acknowledged_exclusion(states.get(rank) if states is not None else None, event)
            if skipped is not None:
                states[rank] = skipped
                return
            self.mark_gap(op, event.get('reason', 'missing telemetry'))
        elif kind == 'coverage_gap':
            self.mark_gap(op, event.get('reason', 'missing telemetry'))
        elif kind in ('entered', 'snapshot', 'complete'):
            if op < self.next_op:
                self.mark_gap(op, 'telemetry arrived after the operation closed')
                return
            states = self.pending[op]
            previous = states.get(rank)
            if previous is not None and previous['event'] == 'skipped':
                self.mark_gap(op, 'progress after an explicit exclusion')
                return
            if (previous is None or kind == 'complete' or
                    (previous['event'] != 'complete' and event['mono_ns'] >= previous['mono_ns'])):
                states[rank] = event
            if (self.phase_samples is not None and kind == 'snapshot' and
                    (event.get('wait_probe') is True or event.get('tree_wait_probe') is True)):
                if op not in self.phase_samples and len(self.phase_samples) == self.MAX_PHASE_OPERATIONS:
                    self.phase_samples.popitem(last=False)
                samples = self.phase_samples.setdefault(op, {})
                self.phase_samples.move_to_end(op)
                samples.setdefault(rank, deque(maxlen=self.MAX_PHASE_SNAPSHOTS)).append(event)
            if len(self.pending) + len(self.completed_by_time) > self.MAX_PENDING:
                self.mark_gap(op, 'too many operations waiting for complete rank coverage')

    def diagnose_hang(self, states):
        if not any(e.get('duration_ns', 0) >= self.cfg.hang_s * 1e9 for e in states.values()):
            return
        progress = [{'comm_hash': self.comm, 'rank': rank, 'world_size': self.world,
                     'op_count': op, 'mono_ns': 0, 'event': 'heartbeat'}
                    for rank, op in self.observed.items()]
        for item in analyze_comm(self.comm, progress + list(states.values()), self.cfg,
                                 tree_references=self.tree_references):
            if item['category'] != 'hang':
                continue
            if (self.last_hang is not None and self.last_hang[:2] == (item['op_count'], 'H1')
                    and item['kind'] == 'H2'):
                known_signatures = {
                    tuple(signature) for signature in item.get('operation_signatures', {}).values()
                    if all(value is not None and value != '' for value in signature)
                }
                if len(known_signatures) <= 1:
                    continue
            key = (item['op_count'], item['kind'], tuple(item['roots']))
            if key != self.last_hang:
                self.last_hang = key
                self.emit(item)

    def evaluate(self, allow_partial_hang):
        if self.rank_losses:
            losses = [{'rank': rank, 'last_op_count': item['op_count'],
                       'process_exited': item['process_exited']}
                      for rank, item in sorted(self.rank_losses.items())]
            self.rank_losses.clear()
            roots = [item['rank'] for item in losses if item['process_exited']]
            diagnosed = (self.last_hang is not None and self.last_hang[0] >= self.next_op and
                         self.last_hang[1] in ('H2', 'H3'))
            if roots and not diagnosed and len(roots) < self.world:
                self.mark_gap(None, 'local process exited before the communicator ended',
                              category='rank_liveness', kind='rank_exit', roots=roots, rank_streams=losses)
            elif roots:
                self.mark_gap(None, 'rank exits observed after a diagnosis or during communicator-wide shutdown',
                              rank_streams=losses)
            else:
                self.mark_gap(None, 'rank stream closed without verified process exit', rank_streams=losses)
        if self.gap:
            return
        while (states := self.pending.get(self.next_op)) is not None:
            kinds = {e['event'] for e in states.values()}
            if len(states) != self.world or not kinds <= {'complete', 'skipped'}:
                if allow_partial_hang and 'skipped' not in kinds:
                    self.diagnose_hang(states)
                break
            op = self.next_op
            reason = aligned_exclusion(states, self.world) if 'skipped' in kinds else None
            if 'skipped' in kinds and reason is None:
                self.mark_gap(op, 'operation exclusions did not align across ranks')
                return
            self.next_op += 1
            del self.pending[op]
            start = states[0]['mono_ns' if states[0]['event'] == 'skipped' else 'start_mono_ns']
            if start < self.rank0_entry_ns:
                self.mark_gap(op, 'rank-0 entry times were out of order')
                return
            self.rank0_entry_ns = start
            if reason:
                self.emit(exclusion_report(self.comm, op, reason))
                continue
            if all(barrier(e) for e in states.values()):
                continue
            mismatch = completed_mismatch(self.comm, op, states, self.world)
            if mismatch:
                self.emit(mismatch)
            else:
                self.diagnose_hang(states)
                heapq.heappush(self.completed_by_time, (states[0]['mono_ns'], op, states))
                if str(states[0].get('algorithm', '')).lower() == 'tree':
                    self.tree_references.append(states)
        # Sampling cyclic slots can observe a later op's completion first.
        # A future rank-0 completion cannot precede its own entry, and the
        # finalized entries above advance in operation order.
        watermark = math.inf if len(self.ended) == self.world else self.rank0_entry_ns
        while self.completed_by_time and self.completed_by_time[0][0] <= watermark:
            _, op, states = heapq.heappop(self.completed_by_time)
            snapshots = self.phase_samples.pop(op, None) if self.phase_samples is not None else None
            try:
                self.window.add(op, states, self.world, self.rank_hosts, snapshots, self.clocks)
            except ValueError as exc:
                self.mark_gap(op, str(exc))
                break


class LiveAnalysis:
    def __init__(self, cfg, emit):
        self.cfg, self.emit = cfg, emit
        self.comms = {}
        self.cross_domain = CrossDomainWaits()
        self.clocks = ClockBounds() if cfg.tree_cross_host_phases else None
        self.events = Counter()
        self.completed = Counter()
        self.first_complete_ns = None
        self.gaps = 0

    def add(self, event):
        self.events[event['event']] += 1
        if event['event'] == 'wait_observation':
            self.cross_domain.add(event)
            return
        if event['event'] == 'complete':
            self.completed[event['op']] += 1
            if self.first_complete_ns is None:
                self.first_complete_ns = time.monotonic_ns()
        comm = event['comm_hash']
        if comm not in self.comms:
            self.comms[comm] = Communicator(comm, event['world_size'], self.cfg, self.emit, self.clocks)
        self.comms[comm].add(event)

    def receive(self, endpoint, records):
        for offset in range(0, len(records), RECORD_BYTES):
            event = endpoint.decoder.consume(records[offset:offset + RECORD_BYTES])
            if event is not None:
                self.add(event)

    def evaluate(self, allow_partial_hang=True):
        for comm, state in list(self.comms.items()):
            state.evaluate(allow_partial_hang)
            if len(state.ended) == state.world:
                if state.pending:
                    state.mark_gap(state.next_op, 'rank streams closed with incomplete operations')
                self.gaps += int(state.gap)
                del self.comms[comm]
        if allow_partial_hang:
            self.cross_domain.evaluate(self.comms, self.emit)

    async def process(self, allow_partial_hang):
        self.evaluate(allow_partial_hang)

    async def finish(self, endpoints):
        for endpoint in endpoints:
            endpoint.decoder.finish()

    async def close(self):
        pass


@dataclass
class Endpoint:
    host: str
    port: int
    cursor: int = 0
    epoch: bytes = b''
    decoder: WaitDecoder = field(default_factory=lambda: WaitDecoder(defer_channels=True))

    def accept(self, response, analysis):
        name = f'{self.host}:{self.port}'
        if self.epoch and self.epoch != response.epoch:
            raise ValueError(f'{name} agent restarted')
        if response.failed:
            raise ValueError(f'{name} reported an incomplete rank stream')
        if not response.oldest <= self.cursor <= response.end:
            raise ValueError(f'{name} telemetry cursor fell outside the retained buffer')
        self.epoch = response.epoch
        analysis.receive(self, response.records)
        self.cursor += response.count
        return self.cursor == response.end


async def monitor(endpoints, token, analysis, poll_s, timeout, stop):
    startup_deadline = time.monotonic() + 15
    shutdown_deadline = None
    ready = False
    previous_unreachable = []
    while True:
        if stop.is_set() and shutdown_deadline is None:
            shutdown_deadline = time.monotonic() + 10
        results = await asyncio.gather(*(pull(e.host, e.port, token, e.cursor, timeout) for e in endpoints),
                                       return_exceptions=True)
        complete, active, unreachable = True, 0, []
        for endpoint, result in zip(endpoints, results):
            if isinstance(result, (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError)):
                unreachable.append(f'{endpoint.host}:{endpoint.port}')
                continue
            if isinstance(result, BaseException):
                raise result
            complete &= endpoint.accept(result, analysis)
            active += result.active
        if ready and unreachable != previous_unreachable:
            if unreachable:
                print('CCL-D temporarily missing nodes: ' + ', '.join(unreachable), file=sys.stderr, flush=True)
            else:
                print('CCL-D reached all node agents again', file=sys.stderr, flush=True)
        previous_unreachable = unreachable
        caught_up = complete and not unreachable
        await analysis.process(allow_partial_hang=caught_up)
        if not unreachable:
            if not ready:
                print('CCLD_ANALYZER_READY', file=sys.stderr, flush=True)
                ready = True
            if caught_up and stop.is_set() and not active:
                await analysis.finish(endpoints)
                return
        if (not ready and time.monotonic() > startup_deadline) or (
                shutdown_deadline is not None and time.monotonic() > shutdown_deadline):
            pending = ', '.join(unreachable) if unreachable else 'undrained rank streams'
            raise ValueError(f'could not establish complete telemetry from {pending}')
        delay = poll_s if unreachable or complete else 0
        if delay and not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), delay)
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(min(delay, 0.05))


def address(value):
    host, sep, port = value.rpartition(':')
    if not sep or not host or not port.isascii() or not port.isdecimal() or not 1 <= int(port) < 65536:
        raise argparse.ArgumentTypeError('node endpoint must be HOST:PORT')
    return Endpoint(host.strip('[]'), int(port))


def emit(item):
    now = time.monotonic_ns()
    if item.get('category') in ('rank_liveness', 'hang', 'slow', 'cross_domain', 'collective_mismatch'):
        item = dict(item, observer_mono_ns=now, timing_note='analyzer local observation time')
    print(json.dumps(item, sort_keys=True), flush=True)


async def refresh_clocks(endpoints, token, clocks, stop):
    while not stop.is_set():
        results = await asyncio.gather(*(sample_clock(e.host, e.port, token, .25) for e in endpoints),
                                       return_exceptions=True)
        for endpoint, sample in zip(endpoints, results):
            if isinstance(sample, Exception):
                continue
            if not endpoint.epoch or endpoint.epoch == sample.epoch:
                clocks.add((endpoint.host, endpoint.port), sample)
        try:
            await asyncio.wait_for(stop.wait(), .1)
        except asyncio.TimeoutError:
            pass


async def run(args, cfg):
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(signum, stop.set)
    analysis = make_analysis(cfg, emit, args.decode_workers, args.coalesce_completed)
    token = job_token()
    sampler = None
    if analysis.clocks is not None and len(args.node) > 1:
        sampler = asyncio.create_task(refresh_clocks(args.node, token, analysis.clocks, stop))
    try:
        await monitor(args.node, token, analysis, args.poll_s, args.timeout_s, stop)
    finally:
        if sampler is not None:
            stop.set()
            await sampler
        await analysis.close()
    emit({'category': 'summary', 'nodes': len(args.node), 'events': dict(analysis.events),
          'completed_operations': dict(analysis.completed),
          'first_complete_observer_mono_ns': analysis.first_complete_ns,
          'incomplete_communicators': len(analysis.comms),
          'communicators_with_gaps': analysis.gaps + sum(comm.gap for comm in analysis.comms.values())})


def make_analysis(cfg, emit, workers, coalesce_completed=False):
    if workers == 1:
        return LiveAnalysis(cfg, emit)
    from .parallel import ParallelAnalysis
    return ParallelAnalysis(cfg, emit, workers, coalesce_completed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node', type=address, action='append', required=True, help='one HOST:PORT per training node')
    p.add_argument('--poll-s', type=float, default=.2)
    p.add_argument('--timeout-s', type=float, default=2.)
    p.add_argument('--decode-workers', type=int, default=1,
                   help='decode in this process (1) or use multiple rank-sharded CPU workers')
    p.add_argument('--coalesce-completed', action='store_true',
                   help='fold progress superseded by completion within a decoder batch')
    add_decision_arguments(p)
    p.add_argument('--tree-cross-host-phases', action='store_true',
                   help='bound node clock offsets for phased Tree slowdown analysis')
    args = p.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in (args.poll_s, args.timeout_s)):
        p.error('poll and timeout intervals must be positive')
    if len({(e.host, e.port) for e in args.node}) != len(args.node):
        p.error('node endpoints must be distinct')
    if args.decode_workers < 1:
        p.error('--decode-workers must be positive')
    if args.coalesce_completed and args.decode_workers == 1:
        p.error('--coalesce-completed requires multiple decode workers')
    try:
        asyncio.run(run(args, decision_config(p, args)))
    except (ValueError, OSError) as exc:
        emit({'category': 'coverage_gap', 'kind': 'undetermined', 'roots': [], 'reason': str(exc)})
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
