#!/usr/bin/env python3
"""Decode experimental paired Ring-wait binary records, including live follow."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

REPRO = Path(os.environ.get('CCLD_REPRO_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPRO / 'tools'))
if __package__:
    from .decode_ccld_binary import (CHANNEL_PARENTS, FRAGMENT_KINDS, FRAGMENT_VERSION, MAX_CHANNELS,
                                     RECORD_BYTES, RECORD_CHANNELS, WAIT_CHANNEL_FIELDS, ChannelMetrics,
                                     Event, channel_payload, decode_event, parse_event, text)
else:
    from decode_ccld_binary import (CHANNEL_PARENTS, FRAGMENT_KINDS, FRAGMENT_VERSION, MAX_CHANNELS,
                                    RECORD_BYTES, RECORD_CHANNELS, WAIT_CHANNEL_FIELDS, ChannelMetrics,
                                    Event, channel_payload, decode_event, parse_event, text)


def fragment_info(e):
    kind = e.reserved & 0xff
    index = (e.reserved >> 8) & 0xff
    total = (e.reserved >> 16) & 0xff
    if (e.type not in FRAGMENT_KINDS or e.reserved >> 24 or kind not in (3, 4) or
            (e.type != 7 and kind != e.type) or not 2 <= total <= MAX_CHANNELS // RECORD_CHANNELS or
            index >= total or not 1 <= e.nChannels <= RECORD_CHANNELS or
            (index < total - 1 and e.nChannels != RECORD_CHANNELS)):
        raise ValueError('invalid CCL-D fragment header')
    return kind, index, total


def progress_key(e, kind):
    return e.hash, e.rank, e.seq, e.nowNs, kind


class WaitDecoder:
    """Join wait records and publish only complete progress observations."""

    def __init__(self, defer_channels=False):
        self.pending = {}
        self.fragments = {}
        self.defer_channels = defer_channels

    def consume(self, raw):
        e = parse_event(raw)
        part = fragment_info(e) if e.version == FRAGMENT_VERSION else None
        if e.type == 1 and e.startNs:
            raise ValueError('new device-frame format requires decode_frame_wait.py')
        if e.type == 7:
            if (e.magic != 0x444c4343 or e.nranks < 1 or e.rank >= e.nranks or
                    not 1 <= e.nChannels <= RECORD_CHANNELS or
                    (part is None and (e.version != 1 or e.reserved not in (3, 4)))):
                raise ValueError('invalid Ring-wait record')
            key = progress_key(e, e.reserved) if part is None else (*progress_key(e, part[0]), *part[1:])
            if key in self.pending or len(self.pending) >= 4096:
                raise ValueError('duplicate or excessive unmatched Ring-wait records')
            payload = channel_payload(raw, e.nChannels)
            ids = set()
            for cid, prev, _, _, nxt in WAIT_CHANNEL_FIELDS.iter_unpack(payload):
                if cid in ids or not 0 <= cid < MAX_CHANNELS or not 0 <= prev < e.nranks or nxt >= e.nranks:
                    raise ValueError('invalid Ring-wait channel or neighbor')
                ids.add(cid)
            self.pending[key] = (ids, payload)
            return None
        record = decode_event(e, raw, allow_fragment=part is not None)
        if e.type == 1:
            record['wait_frame_bytes'] = e.count
            record['wait_requested'] = bool(e.count)
        elif e.type in (3, 4):
            channels = record['channels']
            record['wait_probe'] = text(e.reason) == 'ring_wait_v1'
            if record['wait_probe']:
                key = progress_key(e, e.type)
                if part is not None:
                    key = (*key, *part[1:])
                wait = self.pending.pop(key, None)
                if wait is None or wait[0] != channels.parents().keys():
                    raise ValueError('missing or inconsistent Ring-wait pair')
                channels.wait_payload = wait[1]
            if part is not None:
                record = self.join_fragments(progress_key(e, e.type), part[1], part[2], record)
                if record is None:
                    return None
            if not self.defer_channels:
                record['channels'] = list(record['channels'])
        return record

    def join_fragments(self, key, index, total, record):
        header = {k: v for k, v in record.items() if k not in ('channels', 'n_channels')}
        if index == 0:
            if key in self.fragments or len(self.fragments) >= 4096:
                raise ValueError('duplicate or excessive progress fragments')
            self.fragments[key] = (total, header, [])
        current = self.fragments.get(key)
        if current is None:
            raise ValueError('out of order progress fragment')
        expected, first, channels = current
        if expected != total or len(channels) != index or first != header:
            raise ValueError('inconsistent progress fragments')
        channels.append(record['channels'])
        if len(channels) < total:
            return None
        del self.fragments[key]
        payload = b''.join(chunk.payload for chunk in channels)
        ids = [cid for cid, _ in CHANNEL_PARENTS.iter_unpack(payload)]
        if ids[0] < 0 or ids[-1] >= MAX_CHANNELS or ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError('invalid channels in progress fragments')
        wait = b''.join(chunk.wait_payload for chunk in channels) if first['wait_probe'] else None
        return dict(first, n_channels=len(ids), channels=ChannelMetrics(payload, channels[0].window_ns, wait))

    def finish(self):
        if self.pending:
            raise ValueError('unmatched final Ring-wait records')
        if self.fragments:
            raise ValueError('incomplete final progress fragments')


def decode_file(source, target, decoder_factory=WaitDecoder):
    decoder = decoder_factory()
    with Path(source).open('rb') as src, Path(target).open('x') as out:
        while raw := src.read(RECORD_BYTES):
            record = decoder.consume(raw)
            if record is not None:
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        decoder.finish()


def follow(source_dir, target_dir, interval, decoder_factory=WaitDecoder):
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    offsets, outputs, decoders = {}, {}, {}
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            final_pass = stopping
            for source in sorted(source_dir.glob('ccld-*.ccldbin')):
                if source not in outputs:
                    outputs[source] = (target_dir / (source.stem + '.jsonl')).open('x')
                    offsets[source], decoders[source] = 0, decoder_factory()
                with source.open('rb') as stream:
                    stream.seek(offsets[source])
                    while True:
                        raw = stream.read(RECORD_BYTES)
                        if len(raw) < RECORD_BYTES:
                            if final_pass and raw:
                                raise ValueError(f'{source}: incomplete final binary record')
                            break
                        record = decoders[source].consume(raw)
                        if record is not None:
                            outputs[source].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                        offsets[source] += RECORD_BYTES
                outputs[source].flush()
            if final_pass:
                for decoder in decoders.values():
                    decoder.finish()
                return
            if not stopping:
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
