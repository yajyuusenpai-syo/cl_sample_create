"""Compare +3 dB common gain with L/R exchange on the unchanged baseline cohort."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=ROOT/'outputs'/'full')
    p.add_argument('--out', type=Path, default=ROOT/'outputs'/'gain_plus3db')
    args = p.parse_args()
    baseline = json.loads((args.baseline/'summary.json').read_text(encoding='utf-8'))
    out = args.out.resolve()
    if out == args.baseline.resolve():
        raise ValueError('Use a separate output directory to preserve the baseline.')
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.baseline/'teacher_features.npz') as bank:
        anchors = bank['S'].copy()
        exchanged = bank['S_exc'].copy()
        minus = bank['S_gain'].copy()
        window_indices = bank['window_index'].copy()
    with (args.baseline/'windows.csv').open(encoding='utf-8-sig',newline='') as f:
        windows = {int(row['window_index']):row for row in csv.DictReader(f)}
    assert len(anchors) == baseline['signal_accepted_windows']
    assert baseline['config']['device'] == 'cuda'
    torch.manual_seed(baseline['config']['seed'])
    torch.set_num_threads(6)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    checkpoint = Path(baseline['config']['checkpoint'])
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            digest.update(block)
    expected = baseline['teacher']['audio_checkpoint_sha256']
    if digest.hexdigest() != expected:
        raise ValueError('Teacher checkpoint differs from the baseline.')
    model, info = load_teacher(checkpoint,device)
    gain = 10**(3/20)
    margin = 1e-4
    features, rows, repeat_errors, times = [], [], [], []

    @torch.inference_mode()
    def encode(wave):
        x = torch.from_numpy(np.ascontiguousarray(wave.T)).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        s, _, _ = model.features(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        s = s.float().cpu().numpy()[0]
        if not np.isfinite(s).all() or not np.isclose(np.linalg.norm(s),1,atol=1e-5):
            raise RuntimeError('Invalid normalized teacher embedding.')
        return s,elapsed

    for i, key in enumerate(window_indices):
        meta = windows[int(key)]
        start = round(float(meta['start_seconds'])*16000)
        size = round((float(meta['end_seconds'])-float(meta['start_seconds']))*16000)
        wave,sr = sf.read(meta['wav'],start=start,frames=size,dtype='float32',always_2d=True)
        if sr != 16000 or wave.shape != (size,2):
            raise RuntimeError('Cached waveform format differs from the baseline.')
        if i < 3:
            repeated,_ = encode(wave)
            error = float(np.linalg.norm(repeated-anchors[i]))
            repeat_errors.append(error)
            if error > 1e-5:
                raise RuntimeError(f'Cached baseline mismatch at {i}: {error}')
        boosted = wave * np.float32(gain)
        s,elapsed = encode(boosted)
        features.append(s)
        times.append(elapsed)
        swap_cos = float(np.clip(np.dot(anchors[i],exchanged[i]),-1,1))
        minus_cos = float(np.clip(np.dot(anchors[i],minus[i]),-1,1))
        plus_cos = float(np.clip(np.dot(anchors[i],s),-1,1))
        plus_ok = plus_cos > swap_cos+margin
        minus_ok = minus_cos > swap_cos+margin
        rows.append(dict(embedding_index=i,window_index=int(key),video=meta['video'],
                         start_seconds=meta['start_seconds'],end_seconds=meta['end_seconds'],
                         swap_cosine=swap_cos,minus3db_cosine=minus_cos,plus3db_cosine=plus_cos,
                         plus_minus_cosine_difference=plus_cos-minus_cos,
                         plus3db_minus_swap=plus_cos-swap_cos,minus3db_minus_swap=minus_cos-swap_cos,
                         plus3db_beats_swap=plus_ok,minus3db_beats_swap=minus_ok,
                         original_peak=float(np.max(np.abs(wave))),
                         boosted_peak=float(np.max(np.abs(boosted))),
                         boosted_over_full_scale_fraction=float(np.mean(np.abs(boosted)>1)),
                         forward_seconds=elapsed))
        if (i+1)%25 == 0 or i == 0 or i+1 == len(window_indices):
            print(f'{i+1}/{len(window_indices)}: +3dB cos={plus_cos:.6f}, -3dB cos={minus_cos:.6f}, swap={swap_cos:.6f}',flush=True)
    plus_ok = np.asarray([r['plus3db_beats_swap'] for r in rows])
    minus_ok = np.asarray([r['minus3db_beats_swap'] for r in rows])
    groups = {}
    for label, group in (('no_over_full_scale',[r for r in rows if r['boosted_over_full_scale_fraction']==0]),
                         ('over_full_scale_float',[r for r in rows if r['boosted_over_full_scale_fraction']>0])):
        groups[label] = dict(windows=len(group),
                            plus3db_beats_swap_fraction=float(np.mean([r['plus3db_beats_swap'] for r in group])) if group else None,
                            minus3db_beats_swap_fraction=float(np.mean([r['minus3db_beats_swap'] for r in group])) if group else None)
    report = dict(windows=len(rows),same_signal_filtered_cohort=True,teacher_score_filtering=False,
                  baseline=str(args.baseline.resolve()),gain_db=3,gain_linear=gain,margin=margin,
                  waveform_processing='float32 multiply, no clipping, no limiting, no per-channel normalization',
                  teacher=info,checkpoint_sha256=expected,baseline_repeat_l2=repeat_errors,
                  plus3db_cosine=summarize([r['plus3db_cosine'] for r in rows]),
                  minus3db_cosine=summarize([r['minus3db_cosine'] for r in rows]),
                  swap_cosine=summarize([r['swap_cosine'] for r in rows]),
                  plus3db_beats_swap_count=int(plus_ok.sum()),plus3db_beats_swap_fraction=float(plus_ok.mean()),
                  minus3db_beats_swap_count=int(minus_ok.sum()),minus3db_beats_swap_fraction=float(minus_ok.mean()),
                  both_gains_beat_swap=int((plus_ok&minus_ok).sum()),
                  plus_only_beats_swap=int((plus_ok&~minus_ok).sum()),
                  minus_only_beats_swap=int((~plus_ok&minus_ok).sum()),
                  neither_beats_swap=int((~plus_ok&~minus_ok).sum()),
                  full_scale_subgroups=groups,
                  forward_seconds=summarize(times),
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  interpretation='Feature ranking under gain versus channel exchange; not localization accuracy or audio quality.')
    write_csv(out/'gain_comparison.csv',rows)
    np.savez_compressed(out/'gain_features.npz',S=anchors,S_exc=exchanged,S_minus3db=minus,
                        S_plus3db=np.stack(features),window_index=window_indices)
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines = ['# SpatialCLAP ±3 dB 与声道交换对照','',
             f'使用原有 {len(rows)} 个有效窗口，筛选标准不变，不按教师结果排除样本。','',
             '| 比较对象 | 与原始特征的平均余弦 | 比交换声道更相似的窗口 |',
             '|---|---:|---:|',
             f"| 同时 -3 dB | {report['minus3db_cosine']['mean']:.6f} | {minus_ok.sum()}/{len(rows)}（{minus_ok.mean():.2%}） |",
             f"| 同时 +3 dB | {report['plus3db_cosine']['mean']:.6f} | {plus_ok.sum()}/{len(rows)}（{plus_ok.mean():.2%}） |",
             f"| 交换 L/R | {report['swap_cosine']['mean']:.6f} | 比较基准 |",'',
             f"两种增益都通过：{report['both_gains_beat_swap']}；仅 +3 dB 通过：{report['plus_only_beats_swap']}；仅 -3 dB 通过：{report['minus_only_beats_swap']}；都未通过：{report['neither_beats_swap']}。",'',
             f"放大后有采样点超出 ±1 的窗口：{groups['over_full_scale_float']['windows']}。使用浮点输入，不截幅、不写成 PCM16，以免混入削波失真。",'',
             f'配对排序 margin={margin}。这是表征敏感性测试，不是定位准确率，也不是音频优劣判断。','',
             '原始缓存核对通过。详细逐窗口结果见 gain_comparison.csv。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
