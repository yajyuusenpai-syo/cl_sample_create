"""Offline checkpoint/CPU or GPU smoke test; exits nonzero on GPU failure."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from spatial_teacher import SpatialTeacher

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check NVIDIA driver/container GPU visibility; CPU fallback is not accepted for this check.')
    root=Path(__file__).resolve().parent
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    fixture=np.load(root/'smoke_fixture.npz')
    teacher=SpatialTeacher().to(args.device)
    x=torch.from_numpy(fixture['audio'])
    first=teacher(x);repeat=teacher(x);swap=teacher(x.flip(1))
    assert first.shape==(len(x),256) and torch.isfinite(first).all()
    assert torch.allclose(first.norm(dim=-1),torch.ones(len(x),device=args.device),atol=1e-5)
    repeat_max=float((first-repeat).norm(dim=-1).max())
    reference_max=float(np.linalg.norm(first.cpu().numpy()-fixture['reference'],axis=1).max())
    assert repeat_max<1e-4,f'Repeat mismatch: {repeat_max}'
    # Allow backend/version float32 rounding without allowing random/wrong weights.
    assert reference_max<5e-3,f'Official model feature mismatch: {reference_max}'
    result=dict(status='passed',python=platform.python_version(),torch=torch.__version__,cuda_runtime=torch.version.cuda,
                device=args.device,parameters=sum(p.numel() for p in teacher.parameters()),
                feature_shape=list(first.shape),repeat_max_l2=repeat_max,official_reference_max_l2=reference_max,
                swap_cosines=(first*swap).sum(dim=-1).cpu().tolist())
    if args.device=='cuda':
        torch.cuda.synchronize()
        result.update(gpu=torch.cuda.get_device_name(),compute_capability=list(torch.cuda.get_device_capability()),
                      wheel_cuda_archs=torch.cuda.get_arch_list())
    text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8')

if __name__=='__main__':main()
