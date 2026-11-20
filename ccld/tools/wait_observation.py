"""Report a training framework's actual wait to its node-local CCL-D agent."""
from array import array
import asyncio
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import struct
import threading
import time

from .decode_ccld_binary import Event
from .wait_memory import WaitPage, WaitWriter, new_page_fd

LIMIT = 4096
SOCKET_TIMEOUT = .1
PAGE_REPLY = b'W'
_writers = {}
_retry_after = {}
_writer_lock = threading.Lock()


@dataclass(frozen=True)
class Operation:
    comm_hash: str
    op_count: int
    rank: int

    def fields(self):
        return [self.comm_hash, self.op_count, self.rank]


def wait_socket(rank_socket):
    return str(rank_socket) + '.wait'


def peer_process(writer):
    """Identify the live local process owning a connection on Linux."""
    try:
        peer = writer.get_extra_info('socket')
        pid, uid, _ = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if pid <= 0 or uid != os.getuid():
            return None
        started = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]
        return pid, started
    except (AttributeError, IndexError, OSError, ValueError):
        return None


def parse_operation(raw):
    if (not isinstance(raw, list) or len(raw) != 3 or not isinstance(raw[0], str)
            or not re.fullmatch(r'[0-9a-f]{16}', raw[0])
            or type(raw[1]) is not int or not 0 < raw[1] < 2**64
            or type(raw[2]) is not int or not 0 <= raw[2] < 2**32):
        raise ValueError('invalid wait operation reference')
    return int(raw[0], 16), raw[1], raw[2]


def evidence_record(consumer, producer, world, wait_id, state):
    event = Event()
    event.magic, event.version, event.type, event.reserved = 0x444c4343, 1, 9, state
    event.hash, event.seq, event.rank = consumer
    event.startNs, event.count, event.bytes = producer
    event.nranks, event.windowNs, event.nowNs = world, wait_id, time.monotonic_ns()
    return bytes(event)


def process_groups(bindings):
    groups = {}
    ambiguous = set()
    for (comm, rank, world), count in bindings.items():
        key = (comm, rank)
        if count != 1 or key in groups:
            ambiguous.add(key)
        groups[key] = world
    for key in ambiguous:
        groups.pop(key, None)
    return groups


async def reply(writer, body):
    writer.write(json.dumps(body, separators=(',', ':')).encode() + b'\n')
    await asyncio.wait_for(writer.drain(), timeout=2)


async def receive_wait(reader, writer, buffer):
    wait_id = None
    page = None
    try:
        process = peer_process(writer)
        if process is None:
            raise ValueError('process identity is unavailable')
        message = await asyncio.wait_for(reader.readline(), timeout=2)
        request = json.loads(message)
        if not isinstance(request, dict):
            raise ValueError('invalid wait request')
        groups = process_groups(buffer.process_streams.get(process, {}))
        if request.get('command') == 'groups':
            await reply(writer, {'groups': [(f'{comm:016x}', rank, world)
                                           for (comm, rank), world in sorted(groups.items())]})
            return
        if request.get('command') == 'map':
            if process in buffer.wait_pages:
                raise ValueError('process already has a wait page')
            fd = new_page_fd()
            try:
                page = WaitPage(fd)
                buffer.wait_pages[process] = page
                peer = writer.get_extra_info('socket')
                with peer.dup() as control:
                    control.sendmsg([PAGE_REPLY], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array('i', [fd]))])
            finally:
                os.close(fd)
            await reader.read(1)
            return
        # Keep accepting wait sockets from existing training integrations.
        if request.get('command') != 'wait' or not isinstance(request.get('consumers'), list):
            raise ValueError('invalid wait request')
        producer = parse_operation(request.get('producer'))
        consumers = [parse_operation(row) for row in request['consumers']]
        if not consumers or len(consumers) != len(set(consumers)):
            raise ValueError('expected distinct consumer operations')
        if ((producer[0], producer[2]) not in groups
                or any(c[0] == producer[0] or (c[0], c[2]) not in groups for c in consumers)):
            raise ValueError('wait endpoints must be registered in the same process')
        buffer.next_wait_id += 1
        wait_id = buffer.next_wait_id
        buffer.waits[wait_id] = (process, [(consumer, producer, groups[(consumer[0], consumer[2])], wait_id)
                                         for consumer in consumers])
        buffer.active += 1
        await reply(writer, {'ok': True})
        await reader.read(1)  # The scope ends when the client closes its socket.
    except (asyncio.TimeoutError, OSError, ValueError):
        with suppress(asyncio.TimeoutError, OSError):
            await reply(writer, {'error': 'wait evidence unavailable'})
    finally:
        if page is not None:
            del buffer.wait_pages[process]
            page.close()
        if wait_id is not None:
            buffer.waits.pop(wait_id, None)
            buffer.active -= 1
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()


@contextmanager
def request(body, rank_socket):
    path = rank_socket or os.environ.get('NCCL_CCLD_NODE_SOCKET')
    if not path:
        raise OSError('node telemetry socket is unset')
    data = json.dumps(body, separators=(',', ':')).encode() + b'\n'
    if len(data) > LIMIT:
        raise ValueError('wait request is too large')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(SOCKET_TIMEOUT)
        conn.connect(wait_socket(path))
        conn.sendall(data)
        with conn.makefile('rb') as reply_stream:
            response = json.loads(reply_stream.readline(LIMIT))
        if 'error' in response:
            raise OSError(response['error'])
        yield response


def local_communicators(rank_socket=None):
    """Return registered (comm_hash, local rank, world size) for this process."""
    path = rank_socket or os.environ.get('NCCL_CCLD_NODE_SOCKET')
    with request({'command': 'groups'}, path) as response:
        groups = [tuple(group) for group in response['groups']]
    if groups:
        wait_writer(path)
    return groups


def attach_wait_page(path):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(SOCKET_TIMEOUT)
        conn.connect(wait_socket(path))
        conn.sendall(b'{"command":"map"}\n')
        fds = array('i')
        marker, items, flags, _ = conn.recvmsg(
            1, socket.CMSG_SPACE(fds.itemsize), getattr(socket, 'MSG_CMSG_CLOEXEC', 0))
        for level, kind, value in items:
            if (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                fds.frombytes(value[:len(value) - len(value) % fds.itemsize])
        try:
            if marker != PAGE_REPLY or flags & socket.MSG_CTRUNC or len(fds) != 1:
                raise OSError('shared wait page is unavailable')
            os.set_inheritable(fds[0], False)
            return WaitWriter(WaitPage(fds[0]), conn)
        finally:
            for fd in fds:
                os.close(fd)
    except BaseException:
        conn.close()
        raise


def wait_writer(path):
    writer = _writers.get(path)
    if writer is not None:
        return writer
    with _writer_lock:
        writer = _writers.get(path)
        if writer is not None or _retry_after.get(path, 0) > time.monotonic():
            return writer
        try:
            writer = attach_wait_page(path)
        except (OSError, ValueError):
            _retry_after[path] = time.monotonic() + 1
            return None
        _writers[path] = writer
        _retry_after.pop(path, None)
        return writer


def reset_writers_after_fork():
    try:
        for writer in _writers.values():
            writer.close()
        _writers.clear()
        _retry_after.clear()
    finally:
        _writer_lock.release()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_writer_lock.acquire, after_in_parent=_writer_lock.release,
                        after_in_child=reset_writers_after_fork)


@contextmanager
def observe_wait(producer, consumers, rank_socket=None):
    """Track an observed wait with exact trace IDs. Yield whether it was recorded."""
    path = rank_socket or os.environ.get('NCCL_CCLD_NODE_SOCKET')
    writer, scope = None, None
    if path:
        try:
            rows = tuple(parse_operation(consumer.fields()) for consumer in consumers
                         if consumer.comm_hash != producer.comm_hash)
            if rows and len(rows) == len(set(rows)):
                reference = parse_operation(producer.fields())
                writer = wait_writer(path)
                if writer is not None:
                    scope = writer.begin(reference, rows)
        except (OSError, ValueError):
            pass
    try:
        yield scope is not None
    finally:
        if scope is not None:
            writer.end(scope)
