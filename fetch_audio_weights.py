"""Read only inference audio tensors from the official remote PyTorch ZIP.

HTTP byte ranges avoid downloading optimizer state and the unused text encoder.
The original metadata is parsed without constructing or executing tensor code.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import json
from pathlib import Path
import pickle
import struct
import time
import zipfile
import requests
import torch

ROOT = Path(__file__).resolve().parent
URL = 'https://huggingface.co/sarulab-speech/SpatialCLAP/resolve/main/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt'


class Remote:
    def __init__(self):
        for attempt in range(5):
            try:
                r = requests.get(URL, headers={'Range':'bytes=0-0'}, timeout=(30,90))
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(attempt+1)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError('Official server does not support byte ranges.')
        self.url = r.url
        self.length = int(r.headers['Content-Range'].split('/')[-1])
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.length)+offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def fetch(self, offset, count):
        if count <= 0:
            return b''
        last = min(self.length-1, offset+count-1)
        for attempt in range(5):
            try:
                # Unique range query avoids proxies serving a cached different range.
                r = requests.get(self.url, headers={'Range':f'bytes={offset}-{last}'}, timeout=(30,120))
                r.raise_for_status()
                expected = f'bytes {offset}-{last}/{self.length}'
                if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                    raise RuntimeError(f'Invalid range response: {r.status_code}, {r.headers.get("Content-Range")}, expected {expected}')
                if len(r.content) != last-offset+1:
                    raise RuntimeError('Truncated byte range.')
                return r.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt+1)

    def read(self, count=-1):
        if count < 0:
            count = self.length-self.position
        data = self.fetch(self.position, min(count,self.length-self.position))
        self.position += len(data)
        return data


@dataclass
class Storage:
    dtype: object
    key: str
    elements: int


@dataclass
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


def rebuild(storage, offset, shape, stride, *args, **kwargs):
    return Tensor(storage, offset, shape, stride)


class MetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, value):
        kind, storage_type, key, location, elements = value
        if kind != 'storage':
            raise ValueError(f'Unexpected persistent object: {kind}')
        return Storage(storage_type.dtype, str(key), elements)

    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):
            return rebuild
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return lambda tensor, *args: tensor
        # Official checkpoint metadata; never instantiate a model or import remote code.
        if module in ('collections','torch','numpy','numpy.core.multiarray','numpy._core.multiarray','builtins'):
            return super().find_class(module,name)
        raise pickle.UnpicklingError(f'Unexpected metadata class: {module}.{name}')


def main():
    root = ROOT/'weights'
    root.mkdir(exist_ok=True)
    dest = root/'audio_weights.pt'
    if dest.exists():
        print(f'Already present: {dest}', flush=True)
        return
    remote = Remote()
    print(f'Official training checkpoint: {remote.length:,} bytes', flush=True)
    archive = zipfile.ZipFile(remote)
    data_name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
    prefix = data_name[:-len('data.pkl')]
    metadata = MetadataUnpickler(io.BytesIO(archive.read(data_name))).load()
    state = metadata['model_state_dict']
    selected = {k.removeprefix('module.'): v for k,v in state.items()
                if k.removeprefix('module.').startswith(('audio_encoder.','audio_projection.'))}
    if not selected or not all(isinstance(v,Tensor) for v in selected.values()):
        raise RuntimeError('Unexpected audio state metadata.')
    specs = {v.storage.key:v.storage for v in selected.values()}
    jobs = []
    offsets = {}
    byte_total = 0
    def storage_offset(item):
        key, storage = item
        info = archive.getinfo(prefix+'data/'+key)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError('Compressed tensor storage is unsupported.')
        header = remote.fetch(info.header_offset,30)
        fields = struct.unpack('<4s5H3I2H',header)
        offset = info.header_offset+30+fields[-2]+fields[-1]
        expected = storage.elements*torch.empty((),dtype=storage.dtype).element_size()
        if info.file_size != expected:
            raise RuntimeError(f'Storage {key} has unexpected size.')
        return key,(offset,expected,info.CRC)
    with ThreadPoolExecutor(max_workers=20) as pool:
        for key, details in pool.map(storage_offset,specs.items()):
            offsets[key] = details
            byte_total += details[1]
    print(f'Audio state: {len(selected)} keys, {len(specs)} storages, {byte_total/2**20:.1f} MiB',flush=True)
    for key,(offset,size,crc) in offsets.items():
        for start in range(0,size,2*1024*1024):
            jobs.append((key,start,offset+start,min(2*1024*1024,size-start)))
    buffers = {key:bytearray(size) for key,(_,size,_) in offsets.items()}
    count = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=20) as pool:
        pending = {pool.submit(remote.fetch,offset,size):(key,start,size) for key,start,offset,size in jobs}
        last = started
        for future in as_completed(pending):
            key,start,size = pending[future]
            buffers[key][start:start+size] = future.result()
            count += size
            if time.monotonic()-last > 5:
                print(f'Audio tensors: {count/2**20:.1f}/{byte_total/2**20:.1f} MiB',flush=True)
                last = time.monotonic()
    import zlib
    storages = {}
    for key, buffer in buffers.items():
        if zlib.crc32(buffer) & 0xffffffff != offsets[key][2]:
            raise RuntimeError(f'CRC validation failed for official storage {key}')
        storages[key] = torch.frombuffer(buffer,dtype=specs[key].dtype)
    result = {k:torch.as_strided(storages[v.storage.key],v.shape,v.stride,v.offset).clone()
              for k,v in selected.items()}
    part = dest.with_suffix('.pt.part')
    torch.save({'model_state_dict':result},part)
    part.replace(dest)
    manifest = dict(source=URL,source_checkpoint_bytes=remote.length,
                    audio_tensor_bytes=byte_total,audio_state_keys=len(result),
                    all_downloaded_storages_crc_verified=True,
                    excluded='optimizer state, text encoder and training-only tensors',
                    elapsed_seconds=time.monotonic()-started)
    (root/'audio_weights_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
