"""Extract non-overlapping 256 ms spatial features from stereo WAV/FLAC/NPY."""
import argparse
import io
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from spatial_teacher import SpatialTeacher

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--sample-rate',type=int,help='Required for NPY; array must be [samples,2].')
    args=p.parse_args()
    if args.batch_size<1:p.error('batch-size must be positive')
    if args.input.suffix.lower()=='.npy':
        if args.sample_rate is None:p.error('NPY requires --sample-rate')
        wave=np.load(args.input,allow_pickle=False);sr=args.sample_rate
    elif args.input.suffix.lower() in ['.mp4','.mkv','.mov','.webm','.avi']:
        import imageio_ffmpeg
        # Keep source channel count. In particular, never use '-ac 2' to upmix.
        decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(args.input),
                               '-map','0:a:0','-vn','-acodec','pcm_f32le','-f','wav','pipe:1'],
                              capture_output=True,check=True)
        wave,sr=sf.read(io.BytesIO(decoded.stdout),dtype='float32',always_2d=True)
    else:wave,sr=sf.read(args.input,dtype='float32',always_2d=True)
    if wave.ndim!=2 or wave.shape[1]!=2:raise ValueError('Native stereo [samples,2] required; no mono upmix.')
    if not np.isfinite(wave).all():raise ValueError('Nonfinite audio')
    if sr!=16000:
        factor=gcd(sr,16000);wave=resample_poly(wave,16000//factor,sr//factor,axis=0)
    wave=wave.astype(np.float32);count=len(wave)//4096
    if count==0:raise ValueError('Need at least 256 ms of stereo audio.')
    model=SpatialTeacher().to(args.device);bank=[]
    windows=wave[:count*4096].reshape(count,4096,2).transpose(0,2,1)
    for start in range(0,count,args.batch_size):
        x=torch.from_numpy(np.ascontiguousarray(windows[start:start+args.batch_size]))
        bank.append(model(x).cpu().numpy())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,features=np.concatenate(bank),start_seconds=np.arange(count)*.256,
                        end_seconds=(np.arange(count)+1)*.256,sample_rate=16000)
    print(f'Saved {count} x 256 spatial features to {args.output}; trailing incomplete window omitted.')

if __name__=='__main__':main()
