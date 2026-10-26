"""Offline, signal-gated SpatialCLAP stereo sensitivity validation.

No VAE is needed. S_shuf denotes a shuffled sample/window pairing, not scrambled
waveform samples or latent channels. No teacher scores gate the primary cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / 'new_100_samp_720p'


def command_json(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, check=True,
                            encoding='utf-8', errors='replace')
    return json.loads(result.stdout)


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def cue_metrics(wave, sr, args):
    import numpy as np
    from scipy import signal
    x = wave.astype(np.float64, copy=False)
    eps = 1e-12
    rms = np.sqrt(np.mean(x * x, axis=0) + eps)
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(x*x))) + eps)
    channel_db = 20 * np.log10(rms + eps)
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2
    side_db = 10 * np.log10((np.mean(side*side)+eps)/(np.mean(x*x)+eps))
    corr = float(np.corrcoef(x.T)[0, 1]) if min(np.std(x, axis=0)) > 1e-7 else 0.0
    clip = float(np.mean(np.abs(x) >= .999))
    frame = sr // 20
    blocks = x[:len(x)//frame*frame].reshape(-1, frame, 2)
    energy = np.mean(blocks*blocks, axis=(1, 2))
    activity = float(np.mean(10*np.log10(energy+eps) > args.silence_db))
    _, _, stft = signal.stft(x, fs=sr, nperseg=1024, noverlap=768, axis=0,
                            boundary=None, padded=False)
    # scipy output is frequency x channel x time for a samples x channels input.
    left, right = stft[:, 0, :], stft[:, 1, :]
    power = (np.abs(left)**2 + np.abs(right)**2)/2
    ild = 20*np.log10((np.abs(left)+1e-8)/(np.abs(right)+1e-8))
    freq = np.fft.rfftfreq(1024, 1/sr)
    band_edges = np.geomspace(100, 7500, 9)
    bands = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        select = (freq >= lo) & (freq < hi)
        w = power[select]
        mask = w > max(float(power.max())*1e-4, 1e-12)
        w = w*mask
        bands.append(float((ild[select]*w).sum()/(w.sum()+eps)))
    bands = np.asarray(bands)
    ild_signed = float((ild*power).sum()/(power.sum()+eps))
    ild_abs = float(np.mean(np.abs(bands)))

    # GCC-PHAT over 100 ms blocks: reject unstable/nonphysical delay as a sole cue.
    lags, qualities = [], []
    n = round(sr*getattr(args, 'gcc_frame_ms', 100)/1000)
    maxlag = max(1, int(sr*args.max_itd_ms/1000))
    for offset in range(0, len(x)-n+1, n):
        b = x[offset:offset+n]
        if 10*np.log10(np.mean(b*b)+eps) < args.silence_db:
            continue
        nfft = 1 << (2*n-1).bit_length()
        a = np.fft.rfft(b[:, 0]*np.hanning(n), n=nfft)
        b_fft = np.fft.rfft(b[:, 1]*np.hanning(n), n=nfft)
        cross = a*np.conj(b_fft)
        cc = np.fft.irfft(cross/(np.abs(cross)+1e-10), n=nfft)
        cc = np.concatenate((cc[-maxlag:], cc[:maxlag+1]))
        index = int(np.argmax(np.abs(cc)))
        lags.append(index-maxlag)
        qualities.append(float(np.abs(cc[index])/(np.mean(np.abs(cc))+eps)))
    lag = float(np.median(lags)) if lags else 0.0
    stability = float(np.mean(np.abs(np.asarray(lags)-lag) <= 1)) if lags else 0.0
    quality = float(np.median(qualities)) if qualities else 0.0
    valid_delay = abs(lag) >= 1 and stability >= .6 and quality >= 3
    metrics = dict(rms_db=round(rms_db, 4), left_db=round(float(channel_db[0]), 4),
                   right_db=round(float(channel_db[1]), 4), activity=round(activity, 4),
                   side_ratio_db=round(float(side_db), 4), lr_correlation=round(corr, 6),
                   clipped_fraction=clip, ild_signed_db=round(ild_signed, 4),
                   ild_abs_db=round(ild_abs, 4), itd_us=round(lag/sr*1e6, 3),
                   itd_stability=round(stability, 4), gcc_peak_ratio=round(quality, 4))
    reason = ''
    if rms_db < args.silence_db or activity < .25:
        reason = 'silent_or_low_activity'
    elif clip > .01:
        reason = 'excessive_clipping'
    elif ild_abs < args.min_ild_db and not valid_delay:
        reason = 'weak_or_unreliable_direction_cues'
    return metrics, bands.astype(np.float32), reason


def load_teacher(checkpoint, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT/'upstream'))
    import torchaudio
    from htsat import create_htsat_model
    from seldnet import SELDModel
    # The release is a training checkpoint, potentially including optimizer state.
    # mmap avoids eagerly reading unrelated training tensors into physical RAM.
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = payload['model_state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}

    class AudioTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            # Same audio modules and state names as upstream AudioEncoder.
            # Importing model.py would unnecessarily import the text stack.
            self.audio_encoder = nn.Module()
            self.audio_encoder.mel_encoder = create_htsat_model()
            self.audio_encoder.spatial_encoder = SELDModel()
            self.audio_encoder.resampler = torchaudio.transforms.Resample(16000,48000)
            shape = state['audio_projection.0.weight'].shape
            last = state['audio_projection.2.weight'].shape
            self.audio_projection = nn.Sequential(nn.Linear(shape[1], shape[0]), nn.ReLU(),
                                                  nn.Linear(last[1], last[0]))

        def features(self, x):
            encoder = self.audio_encoder
            mono = (x[:, 0]+x[:, 1])/2
            content = encoder.mel_encoder({'waveform': encoder.resampler(mono)})['embedding']
            spatial = encoder.spatial_encoder(x)
            merged = torch.cat([content, spatial], dim=-1)
            return (F.normalize(self.audio_projection(merged), dim=-1),
                    F.normalize(content, dim=-1), F.normalize(spatial, dim=-1))

    model = AudioTeacher()
    filtered = {k: v for k, v in state.items()
                if k.startswith(('audio_encoder.', 'audio_projection.'))}
    model.load_state_dict(filtered, strict=True)
    model.requires_grad_(False).eval().to(device)
    info = dict(audio_parameters=sum(p.numel() for p in model.parameters()),
                audio_state_keys=len(filtered), strict_audio_load=True,
                checkpoint_bytes=checkpoint.stat().st_size,
                checkpoint_top_level_keys=list(payload.keys()), text_encoder_loaded=False)
    del state, filtered, payload
    return model, info


def summarize(values):
    import numpy as np
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                min=float(a.min()), max=float(a.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--out', type=Path, default=ROOT/'outputs'/'full')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'weights'/'audio_weights.pt')
    p.add_argument('--window-seconds', type=float, default=2)
    p.add_argument('--hop-seconds', type=float, default=2)
    p.add_argument('--limit-videos', type=int, default=0)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--silence-db', type=float, default=-45)
    p.add_argument('--min-side-db', type=float, default=-30)
    p.add_argument('--min-ild-db', type=float, default=1)
    p.add_argument('--max-itd-ms', type=float, default=1)
    p.add_argument('--min-semantic-cos', type=float, default=.5)
    p.add_argument('--min-shuf-ild-distance', type=float, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not (0 < args.window_seconds <= 10 and args.hop_seconds > 0):
        p.error('Window must be in (0, 10] seconds; hop must be positive.')
    import numpy as np
    import soundfile as sf
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = ROOT/'audio_cache'
    audio_dir.mkdir(exist_ok=True)
    ffprobe, ffmpeg = ROOT/'tools'/'ffprobe.exe', ROOT/'tools'/'ffmpeg.exe'
    if not ffprobe.exists() or not ffmpeg.exists():
        raise FileNotFoundError('Run download_assets.py first: FFmpeg/ffprobe are missing.')
    videos = sorted(f for f in args.input.rglob('*') if f.suffix.lower() in {'.mp4','.mkv','.mov','.webm','.avi'})
    if args.limit_videos:
        videos = videos[:args.limit_videos]
    probes, windows, bands_all, waves = [], [], [], []
    sr = 16000
    win, hop = round(args.window_seconds*sr), round(args.hop_seconds*sr)
    run_start = time.perf_counter()
    for n, video in enumerate(videos):
        relative = str(video.relative_to(args.input))
        record = dict(video=relative)
        try:
            probe = command_json([ffprobe, '-v','error','-show_streams','-show_format','-of','json',video])
            streams = [s for s in probe['streams'] if s.get('codec_type') == 'audio']
            record.update(duration=probe.get('format',{}).get('duration'), audio_streams=len(streams))
            stereo = [s for s in streams if s.get('channels') == 2]
            if not stereo:
                record.update(status='no_native_stereo_stream', channel_counts=json.dumps([s.get('channels') for s in streams]))
                probes.append(record)
                print(f'[{n+1}/{len(videos)}] SKIP no native stereo: {video.name}', flush=True)
                continue
            stream = stereo[0]
            record.update(stream_index=stream['index'], channels=stream['channels'],
                          sample_rate=stream.get('sample_rate'), channel_layout=stream.get('channel_layout','unknown'))
            key = hashlib.sha256(relative.encode()).hexdigest()[:12]
            wav = audio_dir/f'{video.stem}_{key}.wav'
            if not wav.exists():
                part = wav.with_suffix('.part.wav')
                cmd = [ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(video),
                       '-map',f"0:{stream['index']}",'-vn','-ar',str(sr),'-c:a','pcm_s16le',str(part)]
                subprocess.run([str(x) for x in cmd], check=True, capture_output=True)
                part.replace(wav)
            wave, actual_sr = sf.read(wav, dtype='float32', always_2d=True)
            if wave.shape[1] != 2 or actual_sr != sr:
                raise RuntimeError(f'Unexpected decoded audio: {wave.shape}, {actual_sr}')
            record.update(status='decoded', decoded_seconds=len(wave)/sr, wav=str(wav))
            accepted = 0
            for start in range(0, len(wave)-win+1, hop):
                segment = wave[start:start+win]
                metrics, bands, reason = cue_metrics(segment, sr, args)
                row = dict(window_index=len(windows), video=relative, wav=str(wav), start_seconds=start/sr,
                           end_seconds=(start+win)/sr, signal_accepted=not reason, rejection_reason=reason, **metrics)
                if not reason:
                    row['embedding_index'] = len(waves)
                    waves.append((wav, start))
                    bands_all.append(bands)
                    accepted += 1
                windows.append(row)
            record['accepted_windows'] = accepted
            print(f'[{n+1}/{len(videos)}] {video.name}: {accepted} accepted', flush=True)
        except Exception as e:
            record.update(status='error', error=f'{type(e).__name__}: {e}')
            print(f'[{n+1}/{len(videos)}] ERROR {video.name}: {e}', flush=True)
        probes.append(record)
    write_csv(out/'video_probe.csv', probes)
    write_csv(out/'windows.csv', windows)
    (out/'probe_manifest.json').write_text(json.dumps(probes, indent=2, ensure_ascii=False), encoding='utf-8')
    report = dict(input=str(args.input), config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  videos=len(videos), video_statuses=dict(Counter(r['status'] for r in probes)),
                  total_complete_windows=len(windows), signal_accepted_windows=len(waves),
                  signal_rejections=dict(Counter(r['rejection_reason'] for r in windows if not r['signal_accepted'])),
                  teacher_gated_primary_cohort=False,
                  scope='Audio-only teacher validation; no VAE, no actual latent tensors, no camera supervision.')
    if not waves:
        report['conclusion'] = 'No signal-valid windows; teacher sensitivity was not evaluated.'
        (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(report['conclusion'], flush=True)
        return

    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly for CPU validation.')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        report['gpu'] = torch.cuda.get_device_name()
    model, info = load_teacher(args.checkpoint, device)
    report['teacher'] = info
    report['torch'] = torch.__version__
    head = (ROOT/'upstream'/'.git'/'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = ROOT/'upstream'/'.git'/head[5:]
        report['upstream_commit'] = ref.read_text().strip() if ref.exists() else head
    else:
        report['upstream_commit'] = head
    infer_seconds, repeat_diffs, rows, bank = [], [], [], []

    @torch.inference_mode()
    def encode(x):
        tensor = torch.from_numpy(np.ascontiguousarray(x.T)).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        # FP32 baseline avoids conflating mixed precision drift with swap sensitivity.
        features = model.features(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        arrays = tuple(a.float().cpu().numpy()[0] for a in features)
        if not all(np.isfinite(a).all() for a in arrays):
            raise RuntimeError('Nonfinite teacher features.')
        return arrays, elapsed

    first_wav, first_start = waves[0]
    first, _ = sf.read(first_wav, start=first_start, frames=win, dtype='float32', always_2d=True)
    encode(first)  # Warm up kernels outside timed statistics.
    for i, (wav, start) in enumerate(waves):
        wave, _ = sf.read(wav, start=start, frames=win, dtype='float32', always_2d=True)
        (s, content, spatial), elapsed = encode(wave)
        (exc, _, _), t_exc = encode(wave[:, ::-1])
        (gain, _, _), t_gain = encode(wave*.7079458)  # -3 dB, common gain on L/R.
        infer_seconds.extend([elapsed, t_exc, t_gain])
        repeat_cos = None
        if i < 20:
            (repeat, _, _), _ = encode(wave)
            repeat_cos = float(np.clip(np.dot(s, repeat), -1, 1))
            repeat_diffs.append(float(np.linalg.norm(s-repeat)))
        swap_cos = float(np.clip(np.dot(s, exc), -1, 1))
        gain_cos = float(np.clip(np.dot(s, gain), -1, 1))
        rows.append(dict(embedding_index=i, swap_cosine=swap_cos, gain_positive_cosine=gain_cos,
                         positive_minus_swap=gain_cos-swap_cos,
                         gain_beats_swap=bool(gain_cos > swap_cos+1e-4), repeat_cosine=repeat_cos))
        bank.append((s, exc, gain, content, spatial))
        if (i+1) % 10 == 0 or i == 0 or i+1 == len(waves):
            print(f'Teacher {i+1}/{len(waves)}: swap_cos={swap_cos:.6f}, gain_cos={gain_cos:.6f}',flush=True)
        if (i+1) % 25 == 0:
            write_csv(out/'teacher_scores.partial.csv', rows)

    arrays = [np.stack([row[k] for row in bank]) for k in range(5)]
    s, exc, gain, content, spatial = arrays
    cues = np.stack(bands_all)
    accepted = [r for r in windows if r['signal_accepted']]
    pairs, negatives = [], np.full(len(waves), -1, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    # Random candidate permutation; rank eligible candidates by content similarity.
    # Geometry eligibility comes from signal cues, never final teacher distance.
    for i, anchor in enumerate(accepted):
        eligible = []
        for j in rng.permutation(len(waves)):
            if i == j or anchor['video'] == accepted[j]['video']:
                continue
            sem_cos = float(np.dot(content[i], content[j]))
            if sem_cos < args.min_semantic_cos:
                continue
            signed_diff = abs(anchor['ild_signed_db']-accepted[j]['ild_signed_db'])
            direction_flip = anchor['ild_signed_db']*accepted[j]['ild_signed_db'] < 0
            delay_flip = (anchor['itd_us']*accepted[j]['itd_us'] < 0
                          and min(anchor['itd_stability'],accepted[j]['itd_stability']) >= .6
                          and abs(anchor['itd_us']-accepted[j]['itd_us']) >= 125)
            if not ((direction_flip and signed_diff >= args.min_shuf_ild_distance) or delay_flip):
                continue
            if float(np.dot(s[i], s[j])) > .9999:
                # Post-hoc duplicate check; report this separate from signal gating.
                continue
            eligible.append((sem_cos,int(j)))
        if eligible:
            sem_cos, j = max(eligible)
            negatives[i] = j
            shuf_cos = float(np.clip(np.dot(s[i],s[j]),-1,1))
            gain_cos = rows[i]['gain_positive_cosine']
            pairs.append(dict(anchor=i, negative=j, anchor_video=anchor['video'],
                              negative_video=accepted[j]['video'], semantic_cosine=sem_cos,
                              shuffled_cosine=shuf_cos, positive_minus_shuf=gain_cos-shuf_cos,
                              gain_beats_shuf=bool(gain_cos > shuf_cos+1e-4)))
    np.savez_compressed(out/'teacher_features.npz', S=s, S_exc=exc, S_gain=gain,
                        content=content, spatial=spatial, ild_bands=cues,
                        shuf_index=negatives, S_shuf=s[np.maximum(negatives,0)],
                        shuf_valid=negatives>=0,
                        window_index=np.asarray([r['window_index'] for r in accepted]))
    write_csv(out/'teacher_scores.csv', rows)
    write_csv(out/'shuffled_pairs.csv', pairs)
    report.update(swap_cosine=summarize([r['swap_cosine'] for r in rows]),
                  gain_positive_cosine=summarize([r['gain_positive_cosine'] for r in rows]),
                  positive_minus_swap=summarize([r['positive_minus_swap'] for r in rows]),
                  gain_beats_swap_fraction=float(np.mean([r['gain_beats_swap'] for r in rows])),
                  repeat_l2=summarize(repeat_diffs), shuffled_pairs=len(pairs),
                  shuffled_cosine=summarize([r['shuffled_cosine'] for r in pairs]),
                  gain_beats_shuf_fraction=float(np.mean([r['gain_beats_shuf'] for r in pairs])) if pairs else None,
                  forward_seconds=summarize(infer_seconds),
                  measured_forward_windows_per_second=1/float(np.mean(infer_seconds)),
                  elapsed_seconds=time.perf_counter()-run_start)
    if device.type == 'cuda':
        report['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        report['peak_gpu_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
    report['limitations'] = [
        'A swap embedding difference does not prove correct localization or camera control.',
        'Gain augmentation is a robustness control, not a guaranteed invariant target.',
        'No ground-truth direction labels: accuracy here means pair ranking only.',
        'Shuffled targets use window pairing indices; actual VAE latents are not present.',
        'S_shuf placeholders with shuf_valid=False MUST be masked in downstream losses.',
        'Content similarity and spatial cue thresholds are heuristics; inspect retained pairs.',
        'CUDA memory reports cover PyTorch allocated/reserved memory, not all process overhead.',
        'All original signal-valid windows remain in the primary swap evaluation.',
    ]
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines = ['# SpatialCLAP 双声道验证报告', '',
             f"视频数：{len(videos)}；完整窗口：{len(windows)}；独立信号筛选后：{len(waves)}。", '',
             f"原始/交换余弦均值：{report['swap_cosine']['mean']:.6f}。", '',
             f"共同增益正例优于交换负例比例（margin=1e-4）：{report['gain_beats_swap_fraction']:.2%}。", '',
             f"有效打乱配对：{len(pairs)}；其余窗口不强行生成负例。", '',
             f"预热后单次前向平均：{report['forward_seconds']['mean']*1000:.1f} ms。", '',
             '这是特征敏感性和配对排序验证，不是有标签的定位准确率，也不证明机位控制有效。', '',
             '主评估只采用独立信号筛选，不按 SpatialCLAP 的交换分数剔除失败窗口。', '',
             '详见 summary.json、windows.csv、teacher_scores.csv、shuffled_pairs.csv。', '',
             'teacher_features.npz 中 S_shuf 必须结合 shuf_valid 使用；无有效负例时跳过该损失。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
