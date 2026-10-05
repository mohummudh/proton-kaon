#!/usr/bin/env python3
"""Extract a few pinned PILArNet-M test events using bounded HTTP range reads."""

import argparse
import io
import json
from pathlib import Path
import time
import urllib.request

import h5py
import numpy as np

REVISION = 'f32f36bd1c17d707d0a24f0c63ec16419475c20f'
URL = f'https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M/resolve/{REVISION}/test/generic_v2_50000_v2.h5'
SIZE = 7023049419


class RangeFile(io.RawIOBase):
    """Read-only HDF5 backend; refuse a server that tries sending the whole file."""

    def __init__(self, transfer_limit_bytes):
        self.position = 0
        self.blocks = {}
        self.transferred = 0
        self.transfer_limit_bytes = transfer_limit_bytes
        self.block_bytes = 262144

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: SIZE}
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError('Invalid seek')
        self.position = bases[whence] + offset
        return self.position

    def read(self, size=-1):
        end = SIZE if size < 0 else min(SIZE, self.position + size)
        output = bytearray()
        while self.position < end:
            key = self.position // self.block_bytes
            if key not in self.blocks:
                start = key * self.block_bytes
                stop = min(SIZE - 1, start + self.block_bytes - 1)
                length = stop - start + 1
                if self.transferred + length > self.transfer_limit_bytes:
                    raise RuntimeError('Transfer limit reached; increase --max-mib if needed')
                request = urllib.request.Request(URL + f'?range_block={key}',
                                                 headers={'Range': f'bytes={start}-{stop}'})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=45) as response:
                            content_range = response.headers.get('Content-Range', '')
                            if response.status != 206 or content_range != f'bytes {start}-{stop}/{SIZE}':
                                raise RuntimeError('Server did not honor the bounded range request')
                            data = response.read(length + 1)
                            if len(data) != length:
                                raise RuntimeError('Range length mismatch')
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        time.sleep(1)
                self.blocks[key] = data
                self.transferred += length
            offset = self.position - key * self.block_bytes
            take = min(end - self.position, len(self.blocks[key]) - offset)
            output += self.blocks[key][offset:offset + take]
            self.position += take
        return bytes(output)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int, default=6)
    parser.add_argument('--max-mib', type=float, default=100)
    args = parser.parse_args()
    if not 1 <= args.events <= 50000 or args.max_mib <= 0:
        parser.error('Require 1–50000 events and a positive transfer limit')
    args.output.mkdir(parents=True, exist_ok=True)
    stream = RangeFile(int(args.max_mib * 1024**2))
    with h5py.File(stream, 'r') as dataset:
        for index in range(args.events):
            arrays = {key: np.asarray(dataset[key][index]).reshape(-1, width)
                      for key, width in (('point', 8), ('cluster', 6), ('cluster_extra', 5))}
            np.savez_compressed(args.output / f'event_{index:05d}.npz', **arrays)
            print(f'Saved event {index}: {len(arrays["point"])} voxels', flush=True)
    provenance = {'url': URL, 'repository_sha': REVISION, 'events': list(range(args.events)),
                  'bytes_downloaded': stream.transferred,
                  'source': 'PILArNet-M official test split; no training performed'}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Transferred {stream.transferred / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
