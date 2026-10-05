#!/usr/bin/env python3
"""Download all pinned PILArNet-M files, resume partial files, verify LFS SHA256."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request

REVISION = 'f32f36bd1c17d707d0a24f0c63ec16419475c20f'
REPOSITORY = 'DeepLearnPhysics/PILArNet-M'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use 1–8 workers')
    args.output.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output / 'remote_inventory.json'
    if inventory_path.exists():
        files = json.loads(inventory_path.read_text())
    else:
        url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true&expand=true'
        with urllib.request.urlopen(url, timeout=45) as response:
            files = [record for record in json.load(response) if record['type'] == 'file']
        inventory_path.write_text(json.dumps(files, indent=2))
    if any(Path(f['path']).is_absolute() or '..' in Path(f['path']).parts for f in files):
        raise ValueError('Unsafe repository path')
    total = sum(f['size'] for f in files)
    present = sum(min(f['size'], max((args.output/f['path']).stat().st_size if (args.output/f['path']).exists() else 0,
                                  (args.output/(f['path']+'.partial')).stat().st_size if (args.output/(f['path']+'.partial')).exists() else 0))
                  for f in files)
    if shutil.disk_usage(args.output).free < total - present + 10 * 1024**3:
        raise RuntimeError('Not enough free space, including a 10-GiB reserve')
    status = {'repository': REPOSITORY, 'revision': REVISION, 'total_bytes': total,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'state': 'running', 'files': {f['path']: {'expected_bytes': f['size'], 'state': 'queued'} for f in files}}
    lock = threading.Lock()

    def download(record):
        path = args.output / record['path']
        partial = Path(str(path) + '.partial')
        path.parent.mkdir(parents=True, exist_ok=True)
        item = status['files'][record['path']]
        try:
            with lock:
                item['state'] = 'downloading'
            if not path.exists():
                for attempt in range(5):
                    url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{record["path"]}?download=true&attempt={time.time_ns()}'
                    command = ['curl', '--location', '--fail', '--retry', '3', '--retry-delay', '5',
                               '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                               '--continue-at', '-', '--output', str(partial), '--silent', '--show-error', url]
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode == 0:
                        break
                    if attempt == 4:
                        raise RuntimeError(f'curl exit {result.returncode}: {result.stderr[-300:]}')
                    time.sleep(5)
                if partial.stat().st_size != record['size']:
                    raise RuntimeError('Downloaded size does not match the pinned inventory')
                candidate = partial
            else:
                candidate = path
            if candidate.stat().st_size != record['size']:
                raise RuntimeError('Existing file has an unexpected size')
            with lock:
                item['state'] = 'verifying'
            digest = sha256(candidate)
            expected = record.get('lfs', {}).get('oid')
            if expected and digest != expected:
                raise RuntimeError('SHA256 mismatch; partial file retained for inspection')
            if candidate == partial:
                os.replace(partial, path)
            with lock:
                item.update(state='verified', sha256=digest)
            print(f'Verified {record["path"]}', flush=True)
        except Exception as error:
            with lock:
                item.update(state='failed', error=str(error))
            print(f'Failed {record["path"]}: {error}', flush=True)

    # Small metadata first; acquire a training shard while the other splits download.
    def priority(record):
        if not record['path'].endswith('.h5'):
            return (0, record['size'])
        preferred = {'train/generic_v2_51800_v2.h5': 1, 'test/generic_v2_50000_v2.h5': 2,
                     'val/generic_v2_66800_v2.h5': 3}
        return (preferred.get(record['path'], 4), record['size'])

    def write_status():
        with lock:
            downloaded = 0
            for record in files:
                path = args.output / record['path']
                partial = Path(str(path) + '.partial')
                size = path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
                size = min(size, record['size'])
                status['files'][record['path']]['downloaded_bytes'] = size
                downloaded += size
            status.update(downloaded_bytes=downloaded,
                          updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            temporary = args.output / 'download_status.json.tmp'
            temporary.write_text(json.dumps(status, indent=2))
            os.replace(temporary, args.output / 'download_status.json')
        return downloaded

    print(f'Downloading {len(files)} files, {total/1e9:.2f} GB, to {args.output}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, f) for f in sorted(files, key=priority)]
        while not all(f.done() for f in futures):
            done = write_status()
            print(f'Progress: {done/1e9:.2f} / {total/1e9:.2f} GB', flush=True)
            time.sleep(15)
    status['state'] = 'complete' if all(f['state'] == 'verified' for f in status['files'].values()) else 'failed'
    write_status()
    if status['state'] != 'complete':
        raise SystemExit('Some files failed; rerun this command to resume. See download_status.json.')
    print('All files downloaded and verified.', flush=True)


if __name__ == '__main__':
    main()
