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
    from .decode_ccld_binary import (Event, RANK_STREAM_LOST, RECORD_BYTES, WAIT_CHANNEL_FIELDS, channel_payload,
                                     decode_event, parse_event, text)
else:
    from decode_ccld_binary import (Event, RANK_STREAM_LOST, RECORD_BYTES, WAIT_CHANNEL_FIELDS, channel_payload,
                                    decode_event, parse_event, text)


class WaitDecoder:
    """Join a wait record with its matching snapshot/completion before output."""

    def __init__(self, defer_channels=False):
        self.pending = {}
        self.defer_channels = defer_channels

    def consume(self, raw):
        e = parse_event(raw)
        if e.type == 1 and e.startNs:
            raise ValueError('new device-frame format requires decode_frame_wait.py')
        if e.type == 7:
            if (e.magic != 0x444c4343 or e.version != 1 or e.nranks < 1 or e.rank >= e.nranks or
                    e.reserved not in (3, 4) or not 1 <= e.nChannels <= 32):
                raise ValueError('invalid Ring-wait record')
            key = (e.hash, e.rank, e.seq, e.nowNs, e.reserved)
            if key in self.pending or len(self.pending) >= 4096:
                raise ValueError('duplicate or excessive unmatched Ring-wait records')
            payload = channel_payload(raw, e.nChannels)
            ids = set()
            for cid, prev, _, _, nxt in WAIT_CHANNEL_FIELDS.iter_unpack(payload):
                if cid in ids or not 0 <= cid < 32 or not 0 <= prev < e.nranks or nxt >= e.nranks:
                    raise ValueError('invalid Ring-wait channel or neighbor')
                ids.add(cid)
            self.pending[key] = (ids, payload)
            return None
        record = decode_event(e, raw)
        if e.type == RANK_STREAM_LOST:
            stream = e.hash, e.rank
            for key in [key for key in self.pending if key[:2] == stream]:
                del self.pending[key]
        elif e.type == 1:
            record['wait_frame_bytes'] = e.count
            record['wait_requested'] = bool(e.count)
        elif e.type in (3, 4):
            channels = record['channels']
            record['wait_probe'] = text(e.reason) == 'ring_wait_v1'
            if record['wait_probe']:
                key = (e.hash, e.rank, e.seq, e.nowNs, e.type)
                wait = self.pending.pop(key, None)
                if wait is None or wait[0] != channels.parents().keys():
                    raise ValueError('missing or inconsistent Ring-wait pair')
                channels.wait_payload = wait[1]
            if not self.defer_channels:
                record['channels'] = list(channels)
        return record

    def finish(self):
        if self.pending:
            raise ValueError('unmatched final Ring-wait records')


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
