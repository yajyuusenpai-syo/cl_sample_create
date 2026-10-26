"""Evaluate within-window synchronized L/R 100 ms block permutation negatives.

This S_shuf experiment differs from cross-window negative pairing in the original
validation. It measures temporal order / edit sensitivity, not spatial accuracy.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np
import soundfile as sf
import torch

from validate_spatialclap import ROOT, load_teacher, summarize, write_csv


def shuffle_chunks(wave, chunk_samples, rng):
    count = len(wave)//chunk_samples
    if count < 2:
        raise ValueError('At least two chunks are required.')
    blocks = wave[:count*chunk_samples].reshape(count,chunk_samples,2)
    order = rng.permutation(count)
    while np.array_equal(order,np.arange(count)):
        order = rng.permutation(count)
    shuffled = blocks[order].reshape(-1,2)
    if count*chunk_samples < len(wave):
        shuffled = np.concatenate((shuffled,wave[count*chunk_samples:]),axis=0)
    # No resampling, no independent channel permutations, no fading or gain.
    assert shuffled.shape == wave.shape
    restored = shuffled[:count*chunk_samples].reshape(count,chunk_samples,2)[np.argsort(order)]
    assert np.array_equal(restored,blocks)
    energy_error = float(np.max(np.abs(np.mean(wave.astype(np.float64)**2,axis=0)-
                                           np.mean(shuffled.astype(np.float64)**2,axis=0))))
    boundaries = np.arange(1,count)*chunk_samples
    before = float(np.mean(np.abs(wave[boundaries]-wave[boundaries-1])))
    after = float(np.mean(np.abs(shuffled[boundaries]-shuffled[boundaries-1])))
    return shuffled,order,dict(fixed_chunks=int(np.sum(order==np.arange(count))),
                               original_adjacencies_retained=int(np.sum(np.diff(order)==1)),
                               rms_energy_error=energy_error,original_join_jump=before,
                               shuffled_join_jump=after)


def gap_analysis(positive,negative):
    gap=np.asarray(positive)-np.asarray(negative)
    valid=gap>1e-4
    selected=gap[valid]
    return dict(pass_count=int(valid.sum()),pass_fraction=float(valid.mean()),
                all_gap=summarize(gap.tolist()),correct_gap=summarize(selected.tolist()),
                correct_gap_median=float(np.median(selected)) if len(selected) else None,
                correct_gap_le_001_count=int(np.sum(selected<=.01)),
                correct_gap_le_001_fraction=float(np.mean(selected<=.01)) if len(selected) else None,
                correct_gap_gt_005_count=int(np.sum(selected>.05)),
                correct_gap_gt_005_fraction=float(np.mean(selected>.05)) if len(selected) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'outputs'/'full')
    p.add_argument('--gain-baseline',type=Path,default=ROOT/'outputs'/'gain_plus3db')
    p.add_argument('--out',type=Path,default=ROOT/'outputs'/'shuffle_100ms')
    p.add_argument('--chunk-ms',type=float,default=100)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    out=args.out.resolve()
    if out in (args.baseline.resolve(),args.gain_baseline.resolve()):
        raise ValueError('A separate output directory is required.')
    out.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((args.baseline/'summary.json').read_text(encoding='utf-8'))
    with np.load(args.baseline/'teacher_features.npz') as b:
        original=b['S'].copy()
        exchanged=b['S_exc'].copy()
        minus=b['S_gain'].copy()
        content=b['content'].copy()
        spatial=b['spatial'].copy()
        indices=b['window_index'].copy()
    with np.load(args.gain_baseline/'gain_features.npz') as b:
        plus=b['S_plus3db'].copy()
        assert np.array_equal(indices,b['window_index'])
        assert np.array_equal(original,b['S'])
    with (args.baseline/'windows.csv').open(encoding='utf-8-sig',newline='') as f:
        windows={int(r['window_index']):r for r in csv.DictReader(f)}
    checkpoint=Path(baseline['config']['checkpoint'])
    sha=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            sha.update(chunk)
    if sha.hexdigest()!=baseline['teacher']['audio_checkpoint_sha256']:
        raise RuntimeError('Teacher differs from baseline.')
    torch.manual_seed(baseline['config']['seed'])
    torch.set_num_threads(6)
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model,info=load_teacher(checkpoint,device)
    rng=np.random.default_rng(args.seed)
    chunk_samples=round(args.chunk_ms*16000/1000)
    if chunk_samples<1:
        raise ValueError('Chunk size must be positive.')
    features,rows,orders,times,repeat_errors=[],[],[],[],[]

    @torch.inference_mode()
    def encode(wave):
        x=torch.from_numpy(np.ascontiguousarray(wave.T)).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        start=time.perf_counter()
        result=model.features(x)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        result=[r.float().cpu().numpy()[0] for r in result]
        if not all(np.isfinite(r).all() and np.isclose(np.linalg.norm(r),1,atol=1e-5) for r in result):
            raise RuntimeError('Invalid teacher features.')
        return result,elapsed

    for i,key in enumerate(indices):
        meta=windows[int(key)]
        start=round(float(meta['start_seconds'])*16000)
        length=round((float(meta['end_seconds'])-float(meta['start_seconds']))*16000)
        wave,sr=sf.read(meta['wav'],start=start,frames=length,dtype='float32',always_2d=True)
        assert sr==16000 and wave.shape==(length,2)
        if i<3:
            repeat,_=encode(wave)
            error=float(np.linalg.norm(repeat[0]-original[i]))
            repeat_errors.append(error)
            if error>1e-5:
                raise RuntimeError(f'Baseline mismatch: {error}')
        shuffled,order,checks=shuffle_chunks(wave,chunk_samples,rng)
        (s,c,z),elapsed=encode(shuffled)
        features.append(s)
        orders.append(order)
        times.append(elapsed)
        cosine=float(np.clip(np.dot(original[i],s),-1,1))
        plus_cos=float(np.clip(np.dot(original[i],plus[i]),-1,1))
        minus_cos=float(np.clip(np.dot(original[i],minus[i]),-1,1))
        swap_cos=float(np.clip(np.dot(original[i],exchanged[i]),-1,1))
        row=dict(embedding_index=i,window_index=int(key),video=meta['video'],
                 start_seconds=meta['start_seconds'],end_seconds=meta['end_seconds'],
                 shuffled_cosine=cosine,swap_cosine=swap_cos,minus3db_cosine=minus_cos,
                 plus3db_cosine=plus_cos,minus3db_minus_shuf=minus_cos-cosine,
                 plus3db_minus_shuf=plus_cos-cosine,
                 conservative_gap=min(plus_cos,minus_cos)-cosine,
                 minus3db_beats_shuf=bool(minus_cos>cosine+1e-4),
                 plus3db_beats_shuf=bool(plus_cos>cosine+1e-4),
                 both_gains_beat_shuf=bool(min(plus_cos,minus_cos)>cosine+1e-4),
                 content_cosine=float(np.clip(np.dot(content[i],c),-1,1)),
                 spatial_cosine=float(np.clip(np.dot(spatial[i],z),-1,1)),
                 chunk_order=' '.join(map(str,order.tolist())),**checks)
        rows.append(row)
        if i<3:
            demo=out/'examples'
            demo.mkdir(exist_ok=True)
            sf.write(demo/f'{i:03d}_original.wav',wave,16000,subtype='PCM_16')
            sf.write(demo/f'{i:03d}_shuffled100ms.wav',shuffled,16000,subtype='PCM_16')
        if i==0 or (i+1)%25==0 or i+1==len(indices):
            print(f'{i+1}/{len(indices)}: shuf={cosine:.6f}, swap={swap_cos:.6f}, -3dB={minus_cos:.6f}, +3dB={plus_cos:.6f}',flush=True)
    negative=[r['shuffled_cosine'] for r in rows]
    pminus=[r['minus3db_cosine'] for r in rows]
    pplus=[r['plus3db_cosine'] for r in rows]
    conservative=np.minimum(pminus,pplus)
    report=dict(windows=len(rows),same_signal_filtered_cohort=True,teacher_score_filtering=False,
                chunk_ms=args.chunk_ms,chunk_samples=chunk_samples,chunks_per_window=len(orders[0]),
                seed=args.seed,left_right_share_permutation=True,permutation='uniform random nonidentity; fixed chunks allowed',
                no_crossfade=True,no_gain_or_sample_modification=True,checkpoint_sha256=sha.hexdigest(),
                teacher=info,baseline_repeat_l2=repeat_errors,
                shuffled_cosine=summarize(negative),swap_cosine=summarize([r['swap_cosine'] for r in rows]),
                minus3db_vs_shuf=gap_analysis(pminus,negative),plus3db_vs_shuf=gap_analysis(pplus,negative),
                both_gains_vs_shuf=gap_analysis(conservative,negative),
                content_cosine=summarize([r['content_cosine'] for r in rows]),
                spatial_cosine=summarize([r['spatial_cosine'] for r in rows]),
                max_channel_energy_error=max(r['rms_energy_error'] for r in rows),
                join_jump_original_mean=float(np.mean([r['original_join_jump'] for r in rows])),
                join_jump_shuffled_mean=float(np.mean([r['shuffled_join_jump'] for r in rows])),
                forward_seconds=summarize(times),peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                limitations=['Temporal chunk shuffling is not a pure spatial negative.',
                             'Hard joins introduce discontinuities; sensitivity can reflect edit artifacts.',
                             'Each window uses one fixed-seed shuffle; this does not estimate variability over permutations.',
                             'Correct ranking does not prove localization accuracy or visual/camera alignment.',
                             'This S_shuf differs from the original cross-window pairing S_shuf.'])
    write_csv(out/'shuffle_comparison.csv',rows)
    correct=sorted([r for r in rows if r['both_gains_beat_shuf']],key=lambda r:r['conservative_gap'])
    write_csv(out/'both_correct_ranked.csv',correct)
    np.savez_compressed(out/'shuffle_features.npz',S=original,S_exc=exchanged,S_minus3db=minus,
                        S_plus3db=plus,S_shuf=np.stack(features),window_index=indices,
                        chunk_permutation=np.stack(orders),chunk_samples=chunk_samples,seed=args.seed)
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 100 ms 段内时序打乱负样本实验','',
           f'相同 {len(rows)} 个两秒有效窗口；每窗 {len(orders[0])} 段，L/R 共用同一随机排列，seed={args.seed}。',
           '保留段内双声道关系、全部采样点及各声道能量，不做淡化、增益或重新筛选。','',
           f"原始/时序打乱余弦：均值 {report['shuffled_cosine']['mean']:.6f}，中位数 {report['shuffled_cosine']['median']:.6f}。",'',
           f"原始/交换声道余弦：均值 {report['swap_cosine']['mean']:.6f}，中位数 {report['swap_cosine']['median']:.6f}。",'',
           '| 与打乱负样本比较 | 排序通过 | 正确组 Δ 均值 | 正确组 Δ 中位数 |',
           '|---|---:|---:|---:|']
    for key,label in [('minus3db_vs_shuf','−3 dB'),('plus3db_vs_shuf','+3 dB'),('both_gains_vs_shuf','两者均通过；取较小间隔')]:
        r=report[key]
        if r['correct_gap']:
            lines.append(f"| {label} | {r['pass_count']}/{len(rows)}（{r['pass_fraction']:.2%}） | {r['correct_gap']['mean']:.6f} | {r['correct_gap_median']:.6f} |")
        else:
            lines.append(f"| {label} | 0/{len(rows)} | — | — |")
    lines+=['','Δ = cos(原始, 增益) − cos(原始, 时序打乱)，通过阈值为 1e-4。','',
            '此实验衡量时间顺序/剪辑敏感性，不是左右定位能力。100 ms 硬切拼接带来接缝，不能把全部特征变化归因于时序理解。','',
            'shuffle_comparison.csv 保存每个窗口的排列、相似度和能量检查；examples 保存前三组原始/打乱听检样本。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
