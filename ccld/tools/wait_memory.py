"""Bounded shared-memory snapshots for a training process's active waits."""
import ctypes
from functools import lru_cache
import mmap
import os
import struct
import threading
import time


PAGE_BYTES = 64 * 1024
BODY_OFFSET = 16
LENGTH = struct.Struct('<Q')
WAIT_ROW = struct.Struct('<QQQQI4xQQI4x')
SEQ_CST = 5


@lru_cache(maxsize=1)
def atomic_word():
    library = ctypes.PyDLL('libatomic.so.1')

    def function(name, result, *arguments):
        fn = getattr(library, name)
        fn.restype = result
        fn.argtypes = arguments
        return fn

    ptr, word = ctypes.c_void_p, ctypes.c_uint64
    is_lock_free = function('__atomic_is_lock_free', ctypes.c_bool, ctypes.c_size_t, ptr)
    if not is_lock_free(8, None):
        raise OSError('shared wait snapshots need lock-free 64-bit atomics')
    load = function('__atomic_load_8', word, ptr, ctypes.c_int)
    store = function('__atomic_store_8', None, ptr, word, ctypes.c_int)
    exchange = function('__atomic_exchange_8', word, ptr, word, ctypes.c_int)
    fence_and_load = function('__atomic_fetch_add_8', word, ptr, word, ctypes.c_int)
    return load, store, exchange, fence_and_load


def new_page_fd():
    import fcntl

    if not hasattr(os, 'memfd_create'):
        raise OSError('shared wait snapshots require memfd on Linux')
    fd = os.memfd_create('ccld-waits', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, PAGE_BYTES)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
        return fd
    except OSError:
        os.close(fd)
        raise


class WaitPage:
    def __init__(self, fd):
        self.load, self.store, self.exchange, self.fence_and_load = atomic_word()
        if os.fstat(fd).st_size != PAGE_BYTES:
            raise ValueError('invalid shared wait page size')
        self.memory = mmap.mmap(fd, PAGE_BYTES, access=mmap.ACCESS_WRITE)
        self.address = ctypes.addressof(ctypes.c_ubyte.from_buffer(self.memory))
        self.version = 0
        self.invalidated = False

    def close(self):
        self.memory.close()

    def clear(self):
        # An odd version hides the previous snapshot and fences any later
        # payload writes. Keep it odd while there are no active waits.
        self.exchange(self.address, self.version + 1, SEQ_CST)
        self.invalidated = True

    def publish(self, body):
        if not self.invalidated:
            self.clear()
        LENGTH.pack_into(self.memory, 8, len(body))
        self.memory[BODY_OFFSET:BODY_OFFSET + len(body)] = body
        self.version += 2
        self.invalidated = False
        try:
            self.store(self.address, self.version, SEQ_CST)
        except BaseException:
            self.clear()
            raise

    def snapshot(self):
        version = self.load(self.address, SEQ_CST)
        if version & 1:
            return b''
        size, = LENGTH.unpack_from(self.memory, 8)
        if not size or size > PAGE_BYTES - BODY_OFFSET or size % WAIT_ROW.size:
            return b''
        body = self.memory[BODY_OFFSET:BODY_OFFSET + size]
        # The read-modify-write fences the payload loads on weakly ordered
        # hosts as well. A concurrent write invalidates the entire copy.
        if self.fence_and_load(self.address, 0, SEQ_CST) != version:
            return b''
        return body


class WaitWriter:
    def __init__(self, page, connection):
        self.page = page
        self.connection = connection
        self.lock = threading.Lock()
        self.scopes = {}
        self.size = 0
        self.next_id = 0

    def close(self):
        self.connection.close()
        page, self.page = self.page, None
        page.close()

    def begin(self, producer, consumers):
        size = len(consumers) * WAIT_ROW.size
        with self.lock:
            if self.size + size > PAGE_BYTES - BODY_OFFSET:
                return None
            self.next_id += 1
            scope = self.next_id
            started = time.monotonic_ns()
            if len(consumers) == 1:
                records = WAIT_ROW.pack(scope, started, *producer, *consumers[0])
            else:
                records = b''.join(WAIT_ROW.pack(scope, started, *producer, *consumer)
                                   for consumer in consumers)
            self.scopes[scope] = records
            self.size += size
            try:
                self.page.publish(records if self.size == size else b''.join(self.scopes.values()))
            except BaseException:
                self.size -= len(self.scopes.pop(scope))
                raise
            return scope

    def end(self, scope):
        if self.page is None:
            return
        with self.lock:
            self.size -= len(self.scopes.pop(scope))
            if self.scopes:
                self.page.publish(b''.join(self.scopes.values()))
            else:
                self.page.clear()
