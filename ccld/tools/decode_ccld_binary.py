#!/usr/bin/env python3
"""Decode fixed-size CCL-D binary records into the existing JSONL contract."""
import argparse
import ctypes
import json
from pathlib import Path
import signal
import time


class Channel(ctypes.LittleEndianStructure):
    _fields_ = [('id', ctypes.c_int32), ('parent', ctypes.c_int32),
                ('send', ctypes.c_uint64), ('recv', ctypes.c_uint64),
                ('sendChanges', ctypes.c_uint64), ('recvChanges', ctypes.c_uint64),
                ('sendDelta', ctypes.c_uint64), ('recvDelta', ctypes.c_uint64)]


class Event(ctypes.LittleEndianStructure):
    _fields_ = [('magic', ctypes.c_uint32), ('version', ctypes.c_uint32),
                ('type', ctypes.c_uint32), ('reserved', ctypes.c_uint32),
                ('hash', ctypes.c_uint64), ('seq', ctypes.c_uint64),
                ('nowNs', ctypes.c_uint64), ('startNs', ctypes.c_uint64),
                ('count', ctypes.c_uint64), ('bytes', ctypes.c_uint64), ('windowNs', ctypes.c_uint64),
                ('rank', ctypes.c_uint32), ('nranks', ctypes.c_uint32),
                ('frameBytes', ctypes.c_uint32), ('frameChannels', ctypes.c_uint32),
                ('frameGroups', ctypes.c_uint32), ('pollUs', ctypes.c_uint32),
                ('mode', ctypes.c_uint32), ('nChannels', ctypes.c_uint32),
                ('host', ctypes.c_char * 64), ('op', ctypes.c_char * 16),
                ('algorithm', ctypes.c_char * 16), ('protocol', ctypes.c_char * 16),
                ('datatype', ctypes.c_char * 16), ('reason', ctypes.c_char * 64),
                ('channels', Channel * 32)]


RECORD_BYTES = 2088
assert ctypes.sizeof(Channel) == 56 and ctypes.sizeof(Event) == RECORD_BYTES
KINDS = {1: 'comm_init', 2: 'entered', 3: 'snapshot', 4: 'complete', 5: 'heartbeat',
         6: 'dropped', 7: 'coverage_gap'}


def text(value):
    return bytes(value).split(b'\0', 1)[0].decode('utf-8')


def decode_record(raw):
    if len(raw) != RECORD_BYTES:
        raise ValueError('incomplete CCL-D binary record')
    e = Event.from_buffer_copy(raw)
    if (e.magic != 0x444c4343 or e.version != 1 or e.type not in KINDS or
            e.nranks < 1 or e.rank >= e.nranks or e.nChannels > 32):
        raise ValueError('invalid CCL-D binary header')
    result = {'event': KINDS[e.type], 'comm_hash': f'{e.hash:016x}', 'rank': e.rank,
              'world_size': e.nranks, 'mono_ns': e.nowNs, 'op_count': e.seq}
    if e.type == 1:
        # Original v1 init records have reserved=0 and always used eight blocks.
        result.update(host=text(e.host), frame_bytes=e.frameBytes, frame_channels=e.frameChannels,
                      frame_groups=e.frameGroups, frame_blocks=e.reserved or 8,
                      queue_slots=e.bytes or 128, queue_bytes=e.windowNs or None,
                      poll_us=e.pollUs, mode=e.mode, completion=text(e.reason))
    elif e.type == 2:
        result.update(op=text(e.op), count=e.count, bytes=e.bytes, datatype=text(e.datatype))
    elif e.type in (3, 4):
        if e.windowNs == 0 or e.nChannels == 0:
            raise ValueError('invalid channel snapshot dimensions')
        duration = max(0, e.nowNs - e.startNs)
        channels = []
        for channel in e.channels[:e.nChannels]:
            channels.append({'id': channel.id, 'tree_parent': channel.parent,
                             'send_count': channel.send, 'recv_count': channel.recv,
                             'send_rate': 1 / channel.sendChanges if channel.sendChanges else None,
                             'recv_rate': 1 / channel.recvChanges if channel.recvChanges else None,
                             'send_delta_per_ms': channel.sendDelta * 1e6 / e.windowNs,
                             'recv_delta_per_ms': channel.recvDelta * 1e6 / e.windowNs})
        result.update(start_mono_ns=e.startNs, duration_ns=duration, op=text(e.op),
                      algorithm=text(e.algorithm), protocol=text(e.protocol), datatype=text(e.datatype),
                      count=e.count, bytes=e.bytes, n_channels=e.nChannels, channels=channels)
        if result['op'] == 'P2PAllPeers':
            result.update(pattern='p2p_all_peers', duration_scope=text(e.reason))
        if e.type == 4:
            factor = 2.0 if text(e.op) == 'AllReduce' else 1.0
            result['busbw_gbps'] = e.bytes * factor * (e.nranks - 1) / e.nranks / duration if duration else 0.0
    elif e.type in (6, 7):
        result['reason'] = text(e.reason)
        if e.type == 6 and result['reason'] == 'binary_queue_overflow':
            result['dropped_records'] = e.count
    return result


def decode_file(source, target):
    data = Path(source).read_bytes()
    if len(data) % RECORD_BYTES:
        raise ValueError(f'{source}: incomplete CCL-D binary record')
    records = [decode_record(data[i:i + RECORD_BYTES]) for i in range(0, len(data), RECORD_BYTES)]
    with Path(target).open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    return len(records)


def follow(source_dir, target_dir, interval):
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    offsets, outputs = {}, {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            for source in sorted(source_dir.glob('ccld-*.ccldbin')):
                if source not in outputs:
                    outputs[source] = (target_dir / (source.stem + '.jsonl')).open('x')
                    offsets[source] = 0
                with source.open('rb') as stream:
                    stream.seek(offsets[source])
                    while True:
                        raw = stream.read(RECORD_BYTES)
                        if len(raw) < RECORD_BYTES:
                            if stopping and raw:
                                raise ValueError(f'{source}: incomplete final binary record')
                            break
                        outputs[source].write(json.dumps(decode_record(raw), ensure_ascii=False, sort_keys=True) + '\n')
                        offsets[source] += RECORD_BYTES
                outputs[source].flush()
            if stopping:
                return
            time.sleep(interval)
    finally:
        for output in outputs.values():
            output.close()
        signal.signal(signal.SIGTERM, previous)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('target', type=Path)
    p.add_argument('--follow', action='store_true')
    p.add_argument('--interval-s', type=float, default=0.05)
    a = p.parse_args()
    if a.interval_s <= 0:
        p.error('interval must be positive')
    if a.follow:
        follow(a.source, a.target, a.interval_s)
    elif a.source.is_dir():
        a.target.mkdir(parents=True, exist_ok=True)
        for source in sorted(a.source.glob('ccld-*.ccldbin')):
            decode_file(source, a.target / (source.stem + '.jsonl'))
    else:
        decode_file(a.source, a.target)


if __name__ == '__main__':
    main()
