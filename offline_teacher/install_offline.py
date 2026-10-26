"""Run with Linux python3.10; build .venv here using verified local wheels only."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    pkgs=ROOT.parent/'pkgs';manifest=json.loads((pkgs/'manifest.json').read_text(encoding='utf-8'))
    print('Verifying all wheel SHA256 checksums...',flush=True)
    for item in manifest['packages']:
        path=pkgs/item['filename']
        if not path.is_file() or sha256(path)!=item['sha256']:raise RuntimeError(f'Missing/corrupt wheel: {path}')
    own=json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,digest in own['files'].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError(f'Missing/corrupt bundle file: {name}')
    print(f'Verified {len(manifest["packages"])} wheels and teacher bundle.',flush=True)
    if args.verify_only:return
    if sys.platform!='linux' or platform.machine().lower() not in ['x86_64','amd64'] or sys.version_info[:2]!=(3,10):
        raise RuntimeError('Run this installer using Linux x86_64 CPython 3.10, not Windows or another Python version.')
    libc=ctypes.CDLL(None);libc.gnu_get_libc_version.restype=ctypes.c_char_p
    glibc=libc.gnu_get_libc_version().decode()
    if tuple(map(int,glibc.split('.')[:2]))<(2,28):
        raise RuntimeError(f'glibc {glibc} < 2.28; use a compatible offline container/OS for these wheels.')
    venv=ROOT/'.venv';python=venv/'bin/python';marker=venv/'.spatial_teacher_offline_env'
    if venv.exists() and not marker.is_file():
        raise RuntimeError(f'Refusing to overwrite an unrelated environment at {venv}; choose a fresh project location.')
    clean={k:v for k,v in os.environ.items() if k not in ['PYTHONPATH','PYTHONHOME','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_FIND_LINKS','PIP_TARGET','PIP_PREFIX','PIP_USER','VIRTUAL_ENV']}
    clean.update(PYTHONNOUSERSITE='1',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
                 PIP_CONFIG_FILE=os.devnull,PIP_NO_CACHE_DIR='1')
    if not venv.exists():
        subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],env=clean,check=True)
        marker.write_text('spatial-teacher-linux-py310-cu130\n',encoding='utf-8')
    actual=subprocess.check_output([str(python),'-c','import sys;print("%d.%d"%sys.version_info[:2])'],env=clean,text=True).strip()
    if actual!='3.10':raise RuntimeError('Existing bundle venv is not Python3.10.')
    pipwheel=next(pkgs.glob('pip-25.1.1-*.whl'))
    bootstrap={**clean,'PYTHONPATH':str(pipwheel)}
    subprocess.run([str(python),'-m','pip','install','--no-index','--no-cache-dir','--only-binary=:all:',
                    '--find-links',str(pkgs),'--require-hashes','-r',str(pkgs/'requirements-linux-py310-cu130.lock')],
                   env=bootstrap,check=True)
    subprocess.run([str(python),'-m','pip','check'],env=clean,check=True)
    subprocess.run([str(python),str(ROOT/'check_teacher.py'),'--device',args.device,
                    '--output',str(ROOT/'installation_check.json')],env=clean,check=True)
    print(f'Offline environment ready: {venv}\nActivate: source "{venv}/bin/activate"',flush=True)

if __name__=='__main__':main()
