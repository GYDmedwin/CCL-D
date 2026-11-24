"""Node-local telemetry buffer and remote, cursor-based binary pulls."""
import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
import hmac
from itertools import groupby
from operator import itemgetter
import os
from pathlib import Path
import secrets
import signal
import stat
import struct
import time
import zlib

from .decode_ccld_binary import FRAGMENT_KINDS, FRAGMENT_VERSION, RECORD_BYTES, TREE_WAIT
from .wait_memory import WAIT_ROW
from .wait_observation import LIMIT, evidence_record, peer_process, process_groups, receive_wait, wait_socket

PROTOCOL = b'CCLDP002'
REQUEST = struct.Struct('!8s32sQ')
RESPONSE = struct.Struct('!8s16sQQIIQQ')
RECORD_HEADER = struct.Struct('<III4xQ48xII')
MAX_PULL = 4096


def job_token():
    """Read the job credential without exposing it in command-line arguments."""
    try:
        token = bytes.fromhex(os.environ['CCLD_PULL_TOKEN'])
    except (KeyError, ValueError):
        raise ValueError('CCLD_PULL_TOKEN must be a 32-byte hexadecimal job token') from None
    if len(token) != 32:
        raise ValueError('CCLD_PULL_TOKEN must be a 32-byte hexadecimal job token')
    return token


class NodeBuffer:
    """Single-writer event loop, with a fixed number of resident records."""
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('node buffer capacity must be positive')
        self.records = [None] * capacity
        self.epoch = secrets.token_bytes(16)
        self.end = 0
        self.failed = 0
        self.active = 0
        self.process_streams = defaultdict(Counter)
        self.binding_since = defaultdict(dict)
        self.wait_pages = {}
        self.shared_wait_ids = {}
        self.waits = {}
        self.published_waits = {}
        self.next_wait_id = 0

    def append(self, raw):
        self.records[self.end % len(self.records)] = raw
        self.end += 1

    def bind_rank(self, process, identity):
        self.process_streams[process][identity] += 1
        self.binding_since[process][identity[:2]] = time.monotonic_ns()

    def unbind_rank(self, process, identity):
        bindings = self.process_streams[process]
        bindings[identity] -= 1
        if bindings[identity] == 0:
            del bindings[identity]
        if not bindings:
            del self.process_streams[process]
        comm, rank, _ = identity
        times = self.binding_since[process]
        if any(key[:2] == (comm, rank) for key in bindings):
            times[comm, rank] = time.monotonic_ns()
        else:
            times.pop((comm, rank), None)
        if not times:
            del self.binding_since[process]
        for wait_id, (owner, records) in list(self.waits.items()):
            if owner == process and any((c[0], c[2]) == (comm, rank) or (p[0], p[2]) == (comm, rank)
                                        for c, p, _, _ in records):
                del self.waits[wait_id]

    def capture_shared_waits(self):
        active = {}
        for process, page in self.wait_pages.items():
            body = page.snapshot()
            if not body:
                continue
            groups = process_groups(self.process_streams.get(process, {}))
            since = self.binding_since.get(process, {})
            for scope, rows in groupby(WAIT_ROW.iter_unpack(body), key=itemgetter(0)):
                records = []
                for _, started, pc, po, pr, cc, co, cr in rows:
                    producer, consumer = (pc, pr), (cc, cr)
                    if (pc == cc or not po or not co or producer not in groups or consumer not in groups
                            or started < max(since[producer], since[consumer])):
                        break
                    records.append(((cc, co, cr), (pc, po, pr), groups[consumer]))
                else:
                    key = (page, scope)
                    wait_id = self.shared_wait_ids.get(key)
                    if wait_id is None:
                        self.next_wait_id += 1
                        wait_id = self.next_wait_id
                    active[key] = wait_id
                    self.waits[wait_id] = (process, [(*fields, wait_id) for fields in records])
        for key in self.shared_wait_ids.keys() - active.keys():
            self.waits.pop(self.shared_wait_ids[key], None)
        self.shared_wait_ids = active

    def capture_waits(self):
        """Export transitions between pull views of the active wait scopes."""
        if self.wait_pages or self.shared_wait_ids:
            self.capture_shared_waits()
        if not self.waits and not self.published_waits:
            return
        for wait_id in sorted(self.published_waits.keys() - self.waits.keys()):
            for fields in self.published_waits[wait_id][1]:
                self.append(evidence_record(*fields, state=2))
        for wait_id in sorted(self.waits.keys() - self.published_waits.keys()):
            for fields in self.waits[wait_id][1]:
                self.append(evidence_record(*fields, state=1))
        self.published_waits = self.waits.copy()

    def batch(self, cursor):
        capacity = len(self.records)
        oldest = max(0, self.end - capacity)
        count = min(MAX_PULL, self.end - cursor) if oldest <= cursor <= self.end else 0
        start = cursor % capacity
        contiguous = self.records[start:start + count]
        records = contiguous + self.records[:count - len(contiguous)]
        raw = b''.join(records)
        compressed = zlib.compress(raw, level=1) if raw else b''
        payload = compressed if len(compressed) < len(raw) else raw
        header = RESPONSE.pack(PROTOCOL, self.epoch, oldest, self.end, count, len(payload),
                               self.failed, self.active)
        return header, payload


async def receive_rank(reader, writer, buffer):
    buffer.active += 1
    process = peer_process(writer)
    identity, ended, bound, remainder = None, False, False, b''
    try:
        while data := await reader.read(64 * RECORD_BYTES):
            data = remainder + data
            boundary = len(data) // RECORD_BYTES * RECORD_BYTES
            for offset in range(0, boundary, RECORD_BYTES):
                raw = data[offset:offset + RECORD_BYTES]
                magic, version, kind, comm, rank, world = RECORD_HEADER.unpack_from(raw)
                rank_identity = (comm, rank, world)
                if (magic != 0x444c4343 or not (1 <= kind <= 8 or kind == TREE_WAIT) or
                        (version != 1 and not (version == FRAGMENT_VERSION and kind in FRAGMENT_KINDS)) or
                        rank >= world or ended or
                        (identity is None and kind != 1) or
                        (identity is not None and (kind == 1 or identity != rank_identity))):
                    raise ValueError('invalid rank telemetry stream')
                if process is not None and kind == 1:
                    buffer.bind_rank(process, rank_identity)
                    bound = True
                identity = rank_identity
                ended = kind == 8
                buffer.append(raw)
                if ended and bound:
                    buffer.unbind_rank(process, identity)
                    bound = False
            remainder = data[boundary:]
        if not ended or remainder:
            buffer.failed += 1
    except (OSError, ValueError):
        buffer.failed += 1
    finally:
        buffer.active -= 1
        if bound:
            buffer.unbind_rank(process, identity)
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()


async def serve_pull(reader, writer, buffer, token):
    try:
        raw = await asyncio.wait_for(reader.readexactly(REQUEST.size), timeout=2)
        protocol, credential, cursor = REQUEST.unpack(raw)
        if protocol == PROTOCOL and hmac.compare_digest(credential, token):
            buffer.capture_waits()
            header, data = buffer.batch(cursor)
            writer.write(header)
            writer.write(data)
            await asyncio.wait_for(writer.drain(), timeout=2)
    except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError):
        pass
    finally:
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()


@dataclass(frozen=True)
class Pull:
    epoch: bytes
    oldest: int
    end: int
    failed: int
    active: int
    records: bytes
    payload_bytes: int = 0

    @property
    def count(self):
        return len(self.records) // RECORD_BYTES


async def pull(host, port, token, cursor, timeout):
    async def exchange():
        reader, writer = await asyncio.open_connection(host, port)
        try:
            writer.write(REQUEST.pack(PROTOCOL, token, cursor))
            await writer.drain()
            raw = await reader.readexactly(RESPONSE.size)
            protocol, epoch, oldest, end, count, encoded_bytes, failed, active = RESPONSE.unpack(raw)
            expected = min(MAX_PULL, end - cursor) if oldest <= cursor <= end else 0
            raw_bytes = count * RECORD_BYTES
            if (protocol != PROTOCOL or oldest > end or count != expected or
                    encoded_bytes > raw_bytes or (count > 0 and encoded_bytes == 0)):
                raise ValueError('invalid node-agent response')
            data = await reader.readexactly(encoded_bytes)
            if encoded_bytes < raw_bytes:
                expansion = zlib.decompressobj()
                try:
                    data = expansion.decompress(data, max_length=raw_bytes + 1)
                except zlib.error as exc:
                    raise ValueError('invalid compressed telemetry batch') from exc
                if len(data) != raw_bytes or not expansion.eof or expansion.unused_data:
                    raise ValueError('compressed telemetry batch size does not match its header')
            return Pull(epoch, oldest, end, failed, active, data, encoded_bytes)
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
    return await asyncio.wait_for(exchange(), timeout=timeout)


async def serve(path, bind, port, capacity, token):
    source_directory = path.parent.stat()
    if source_directory.st_uid != os.getuid() or stat.S_IMODE(source_directory.st_mode) & 0o077:
        raise ValueError('the rank socket must be in a private directory owned by the job user')
    control = Path(wait_socket(path))
    if any(p.exists() or p.is_symlink() for p in (path, control)):
        raise ValueError('node socket already exists')
    node = NodeBuffer(capacity)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    local = await asyncio.start_unix_server(lambda r, w: receive_rank(r, w, node), str(path), backlog=256)
    try:
        waits = await asyncio.start_unix_server(lambda r, w: receive_wait(r, w, node), str(control),
                                                limit=LIMIT, backlog=256)
        try:
            tcp = await asyncio.start_server(lambda r, w: serve_pull(r, w, node, token), bind, port)
            try:
                print(f'CCLD_NODE_READY {bind}:{tcp.sockets[0].getsockname()[1]}', flush=True)
                await stop.wait()
            finally:
                tcp.close()
                await tcp.wait_closed()
        finally:
            waits.close()
            await waits.wait_closed()
            control.unlink(missing_ok=True)
    finally:
        local.close()
        await local.wait_closed()
        path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', type=Path, required=True, help='Unix socket path in a private node-local directory')
    p.add_argument('--bind', required=True, help='hostname or address on the job network')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--records', type=int, default=16384, help='maximum number of retained records on this node')
    args = p.parse_args()
    if not 0 <= args.port < 65536 or args.records < MAX_PULL:
        p.error(f'port out of range or --records below {MAX_PULL}')
    os.umask(0o077)
    try:
        asyncio.run(serve(args.socket, args.bind, args.port, args.records, job_token()))
    except (OSError, ValueError) as exc:
        p.exit(1, f'CCL-D node agent: {exc}\n')


if __name__ == '__main__':
    main()
