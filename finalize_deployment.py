"""Check final artifacts and record reproducibility and project disk usage."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import numpy as np

root = Path(__file__).resolve().parent
out = root/'outputs'/'full'
report_path = out/'summary.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
with np.load(out/'teacher_features.npz') as data:
    n = report['signal_accepted_windows']
    assert data['S'].shape == (n,512)
    for key in ('S','S_exc','S_gain','S_shuf'):
        assert np.isfinite(data[key]).all(), key
        assert np.allclose(np.linalg.norm(data[key],axis=1),1,atol=1e-5), key
    valid = data['shuf_valid']
    indices = data['shuf_index'][valid]
    assert np.all((indices >= 0) & (indices < n))
    assert np.all(indices != np.arange(n)[valid])
    assert np.allclose(data['S_shuf'][valid],data['S'][indices])
    assert len(np.unique(data['window_index'])) == n
    assert report['teacher']['strict_audio_load']

head = (root/'upstream'/'.git'/'HEAD').read_text().strip()
if head.startswith('ref: '):
    head = (root/'upstream'/'.git'/head[5:]).read_text().strip()
report['upstream_commit'] = head
names = ('torch','torchaudio','numpy','scipy','torchlibrosa','librosa','soundfile',
         'numba','llvmlite','requests','scikit-learn')
report['package_versions'] = {name:importlib.metadata.version(name) for name in names}
for name in ('ffmpeg','ffprobe'):
    result = subprocess.run([str(root/'tools'/f'{name}.exe'),'-version'],capture_output=True,
                            encoding='utf-8',errors='replace',check=True)
    report[name+'_version'] = result.stdout.splitlines()[0]
weight = root/'weights'/'audio_weights.pt'
digest = hashlib.sha256()
with weight.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):
        digest.update(chunk)
report['teacher']['audio_checkpoint_sha256'] = digest.hexdigest()
report['artifact_checks_passed'] = True
report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
sizes = {}
for directory in root.iterdir():
    if directory.is_dir():
        sizes[directory.name] = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
sizes['root_files'] = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
result = dict(project=str(root),bytes_by_directory=sizes,total_bytes=sum(sizes.values()),
              total_gib=sum(sizes.values())/2**30,upstream_commit=head,
              package_versions=report['package_versions'],artifact_checks_passed=True)
(root/'deployment_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
(root/'requirements-extra.lock.txt').write_text('\n'.join(
    f'{name}=={importlib.metadata.version(name)}' for name in
    ('torchlibrosa','librosa','soundfile','numba','llvmlite','audioread','soxr','msgpack','pooch','lazy_loader'))+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
