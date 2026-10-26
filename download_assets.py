"""Download official inference assets into the deployed project, resumable by file."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile
import requests


def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        print(f"Already present: {path.name} ({path.stat().st_size:,} bytes)", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    start = time.monotonic()
    last = start
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0))
        count = 0
        with part.open('wb') as out:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                count += len(chunk)
                if time.monotonic() - last > 5:
                    print(f"{path.name}: {count / 2**20:.1f} MiB / {total / 2**20:.1f} MiB", flush=True)
                    last = time.monotonic()
    if total and count != total:
        raise RuntimeError(f"Incomplete download: {count} != {total}")
    part.replace(path)
    print(f"Downloaded {path.name}: {count:,} bytes in {time.monotonic()-start:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--tools-only', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    assets = [
        ('https://huggingface.co/sarulab-speech/SpatialCLAP/resolve/main/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt',
         root / 'weights' / 'spatialclap.pt'),
        ('https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
         root / 'tools' / 'ffmpeg-essentials.zip'),
    ]
    if args.tools_only:
        assets = assets[1:]
    for url, path in assets:
        download(url, path)
    tools = root / 'tools'
    if not (tools / 'ffprobe.exe').exists() or not (tools / 'ffmpeg.exe').exists():
        with zipfile.ZipFile(tools / 'ffmpeg-essentials.zip') as z:
            for name in ('ffmpeg.exe', 'ffprobe.exe'):
                match = next(n for n in z.namelist() if n.endswith('/bin/' + name))
                with z.open(match) as src, (tools / name).open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
            licenses = [n for n in z.namelist() if n.endswith('/LICENSE') or n.endswith('/LICENSE.txt')]
            for name in licenses:
                (tools / 'FFMPEG-LICENSE.txt').write_bytes(z.read(name))
    manifest = []
    for url, path in assets:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        manifest.append(dict(url=url, path=str(path.relative_to(root)), bytes=path.stat().st_size,
                             sha256=digest.hexdigest()))
    (root / 'assets_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Asset setup finished.', flush=True)


if __name__ == '__main__':
    main()
