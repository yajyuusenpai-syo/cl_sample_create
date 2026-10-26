"""Restore the pinned Linux/Python3.10/CUDA13 wheelhouse on an ONLINE machine.

Standard library only. Once downloaded, copy this project to the offline machine
and run offline_teacher/install_offline.py there. The binary wheels are omitted
from Git; all SHA256 hashes and original source URLs are committed in pkgs/.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def download(item, root, verify_only=False):
    path=root/item['filename']
    if path.is_file() and sha256(path)==item['sha256']:
        print(f'Verified {path.name}',flush=True);return
    if verify_only:raise RuntimeError(f'Missing/corrupt wheel: {path}')
    partial=path.with_suffix('.whl.partial')
    size=item.get('size')
    failures=0
    while True:
        offset=partial.stat().st_size if partial.exists() else 0
        if size is not None and offset==size:break
        if size is not None and offset>size:raise RuntimeError(f'Oversized partial file: {partial}')
        end=offset+4*1024*1024-1
        if size is not None:end=min(end,size-1)
        request=Request(item['url'],headers={'Range':f'bytes={offset}-{end}','User-Agent':'cl-sample-create-wheelhouse/1'})
        try:
            with urlopen(request,timeout=120) as response:
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if response.status!=206 or match is None or int(match[1])!=offset:
                    raise RuntimeError('Server did not honor the requested byte range')
                size=int(match[3]);remaining=int(match[2])-offset+1
                with partial.open('ab') as dest:
                    while remaining:
                        block=response.read(min(1024*1024,remaining))
                        if not block:raise RuntimeError('Truncated byte range')
                        dest.write(block);remaining-=len(block)
            failures=0
        except Exception:
            failures+=1
            if failures>=6:raise
            time.sleep(min(2**failures,30))
    if sha256(partial)!=item['sha256']:
        raise RuntimeError(f'SHA256 mismatch: {partial}; file retained for inspection')
    partial.replace(path)
    print(f'Downloaded and verified {path.name}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkgs-dir',type=Path,default=Path(__file__).resolve().parent/'pkgs')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    root=args.pkgs_dir.expanduser().resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    print(f'{len(manifest["packages"])} wheels; {manifest["total_bytes"]/2**30:.3f} GiB',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda item:download(item,root,args.verify_only),manifest['packages']))
    print('Wheelhouse ready. Transfer the project to Linux and run offline_teacher/install_offline.py.',flush=True)

if __name__=='__main__':main()
