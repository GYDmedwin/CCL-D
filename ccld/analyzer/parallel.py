"""Decode rank streams on dedicated CPU processes and join decisions centrally."""
import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import struct

from .live import LiveAnalysis
from ..tools.decode_async_wait import Event, RECORD_BYTES, WaitDecoder

_stream_fields = struct.Struct('<Q48xI')
_decoder = None


def _start_decoder():
    global _decoder
    _decoder = WaitDecoder(defer_channels=True)


def _decode_events(records, indices):
    for offset, index in zip(range(0, len(records), RECORD_BYTES), indices):
        event = _decoder.consume(records[offset:offset + RECORD_BYTES])
        if event is not None:
            yield index, event


def _decode(records, indices):
    return list(_decode_events(records, indices)), {}


def _decode_coalesced(records, indices):
    """Retain progress updates until the same rank and operation completes."""
    pending = {}
    updates = []
    folded = Counter()
    for index, event in _decode_events(records, indices):
        kind = event['event']
        stream = (event['comm_hash'], event['rank'])
        key = (event['op_count'], event['world_size'])
        if kind in ('entered', 'snapshot'):
            pending.setdefault(stream, {}).setdefault(key, []).append((index, event))
            continue
        if kind == 'complete':
            earlier = pending.get(stream, {}).pop(key, ())
            if earlier:
                # The coordinator evaluates only after the whole batch. Keep
                # the first position so the pending-operation limit sees the
                # same operation creation order.
                index = earlier[0][0]
                for _, previous in earlier:
                    folded[previous['event']] += 1
        elif kind != 'heartbeat':
            # Apply progress on its original side of any stream boundary.
            for events in pending.pop(stream, {}).values():
                updates.extend(events)
        updates.append((index, event))
    for operations in pending.values():
        for events in operations.values():
            updates.extend(events)
    return updates, folded


def _finish_decoder():
    _decoder.finish()


class ParallelAnalysis(LiveAnalysis):
    """Keep one batch in flight while the network collects the next batch."""
    def __init__(self, cfg, emit, workers, coalesce_completed=False):
        super().__init__(cfg, emit)
        self.context = multiprocessing.get_context('spawn')
        self.workers = workers
        self.decode = _decode_coalesced if coalesce_completed else _decode
        self.coalesced = Counter()
        # Each shard always uses the same process. A Ring-wait pair may span
        # batches, so the worker must retain its decoder between calls.
        self.pools = {}
        self.chunks = []
        self.inflight = None

    def receive(self, endpoint, records):
        if records:
            self.chunks.append(records)

    async def _apply_previous(self):
        if self.inflight is None:
            return
        tasks, allow_partial_hang = self.inflight
        self.inflight = None
        try:
            results = await asyncio.gather(*tasks)
        except BrokenProcessPool as exc:
            raise ValueError('a parallel decode worker exited early') from exc
        for _, folded in results:
            self.events.update(folded)
            self.coalesced.update(folded)
        # Merge the remaining updates in their original pulled order.
        for _, event in sorted(record for updates, _ in results for record in updates):
            self.add(event)
        self.evaluate(allow_partial_hang)

    async def process(self, allow_partial_hang):
        await self._apply_previous()
        if not self.chunks:
            self.evaluate(allow_partial_hang)
            return
        groups = {}
        index = 0
        for chunk in self.chunks:
            view = memoryview(chunk)
            for offset in range(0, len(chunk), RECORD_BYTES):
                communicator, rank = _stream_fields.unpack_from(chunk, offset + Event.hash.offset)
                shard = (communicator + rank) % self.workers
                if shard not in groups:
                    groups[shard] = (bytearray(), [])
                payload, indices = groups[shard]
                payload.extend(view[offset:offset + RECORD_BYTES])
                indices.append(index)
                index += 1
        self.chunks.clear()
        loop = asyncio.get_running_loop()
        tasks = []
        for shard, (data, indices) in groups.items():
            if shard not in self.pools:
                self.pools[shard] = ProcessPoolExecutor(
                    max_workers=1, mp_context=self.context, initializer=_start_decoder)
            tasks.append(loop.run_in_executor(self.pools[shard], self.decode, bytes(data), indices))
        self.inflight = tasks, allow_partial_hang

    async def finish(self, endpoints):
        await self._apply_previous()
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(loop.run_in_executor(pool, _finish_decoder) for pool in self.pools.values()))

    async def close(self):
        if self.inflight is not None:
            tasks, _ = self.inflight
            await asyncio.gather(*tasks, return_exceptions=True)
            self.inflight = None
        for pool in self.pools.values():
            pool.shutdown()
