"""CPU-only left/right energy and GCC-PHAT aggregate transformation checks.

No learned teacher, PyTorch, VAE decoder or vocoder is used. Exchange/copy
relations are mathematical sanity checks, not learned localization accuracy.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import subprocess
import time
import numpy as np
import soundfile as sf
from scipy import signal

ROOT=Path(__file__).resolve().parent
DEFAULT_INPUT=ROOT.parent/'new_100_samp_720p'


def describe(values):
    x=np.asarray(values,dtype=float)
    if not len(x):
        return None
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),
                p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),
                min=float(x.min()),max=float(x.max()))


def cosine(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    sa,sb=float(np.max(np.abs(a))),float(np.max(np.abs(b)))
    if sa==0 or sb==0 or not np.isfinite(sa+sb):
        return None
    a,b=a/sa,b/sb
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a,b)/denominator,-1,1))


def quality_gate(wave,silence_db=-45):
    x=wave.astype(np.float64,copy=False)
    power=float(np.mean(x*x))
    if not np.isfinite(x).all():
        return 'nonfinite_samples'
    frame=800
    frames=x[:len(x)//frame*frame].reshape(-1,frame,2)
    active=np.mean(10*np.log10(np.mean(frames*frames,axis=(1,2))+1e-30)>silence_db)
    if 10*np.log10(power+1e-30)<silence_db or active<.25:
        return 'silent_or_low_activity'
    if np.mean(np.abs(x)>=.999)>.01:
        return 'excessive_clipping'
    return ''


def extract_cues(wave,sr=16000,max_itd_ms=1,interp=8):
    """Return 16 nonnegative L/R band energies, 2 total energies and GCC.

    GCC uses L * conj(R); tau = arrival_time_L - arrival_time_R. Negative
    tau means the left channel arrives earlier. This is not an azimuth label.
    """
    x=wave.astype(np.float64,copy=False)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<512:
        raise ValueError('Expected at least 512 samples of stereo waveform.')
    freq,_,z=signal.stft(x,fs=sr,nperseg=512,noverlap=352,axis=0,
                        boundary=None,padded=False)
    edges=np.geomspace(100,min(7500,sr/2-1),9)
    pl,pr=[],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(freq>=lo)&(freq<hi)
        pl.append(float(np.mean(np.sum(np.abs(z[mask,0,:])**2,axis=0))))
        pr.append(float(np.mean(np.sum(np.abs(z[mask,1,:])**2,axis=0))))
    pl,pr=np.asarray(pl),np.asarray(pr)
    band_power=(pl+pr)/2
    band_valid=band_power>=max(float(band_power.max())*1e-4,1e-30)
    left_energy=np.where(band_valid,pl,0)
    right_energy=np.where(band_valid,pr,0)
    energy=np.concatenate((left_energy,right_energy))
    total_energy=np.mean(x*x,axis=0)
    energy_valid=bool(float(energy.sum())>1e-30)
    energy_target=energy/energy.sum() if energy_valid else np.zeros_like(energy)

    n=len(x)
    nfft=1<<(2*n-1).bit_length()
    window=np.hanning(n)
    a=np.fft.rfft(x[:,0]*window,n=nfft)
    b=np.fft.rfft(x[:,1]*window,n=nfft)
    cross=a*np.conj(b)
    amplitude=np.abs(cross)
    cross_valid=bool(float(amplitude.max())>1e-24)
    floor_cross=max(float(amplitude.max())*1e-8,1e-30)
    phat=cross/np.maximum(amplitude,floor_cross)
    cc=np.fft.irfft(phat,n=nfft*interp)
    shift=min(round(sr*interp*max_itd_ms/1000),len(cc)//2-1)
    gcc=np.concatenate((cc[-shift:],cc[:shift+1]))
    norm=float(np.linalg.norm(gcc))
    gcc_valid=cross_valid and norm>1e-12
    gcc=gcc/norm if gcc_valid else np.zeros_like(gcc)
    peak_index=int(np.argmax(np.abs(gcc))) if gcc_valid else shift
    tau_us=(peak_index-shift)/(sr*interp)*1e6 if gcc_valid else None
    peak_ratio=float(np.max(np.abs(gcc))/(np.mean(np.abs(gcc))+1e-30)) if gcc_valid else 0
    return dict(energy=energy,energy_target=energy_target,energy_valid=energy_valid,
                total_energy=total_energy,left_energy=left_energy,right_energy=right_energy,
                left_right_spectral_cosine=cosine(left_energy,right_energy),
                gcc=gcc,gcc_valid=gcc_valid,
                tau_us=tau_us,peak_ratio=peak_ratio,
                zero_lag_index=shift,lag_bin_us=1e6/(sr*interp))


def evaluate_pairs(original,other):
    energy_cos=cosine(original['energy'],other['energy']) if original['energy_valid'] and other['energy_valid'] else None
    gcc_cos=cosine(original['gcc'],other['gcc']) if original['gcc_valid'] and other['gcc_valid'] else None
    return dict(energy_cosine=energy_cos,total_energy_cosine=cosine(original['total_energy'],other['total_energy']),
                gcc_cosine=gcc_cos,
                itd_difference_us=abs(original['tau_us']-other['tau_us']) if original['gcc_valid'] and other['gcc_valid'] else None)


def stronger_ear_mono(wave):
    """Copy the higher full-window waveform-energy ear to both channels."""
    x=wave.astype(np.float64,copy=False)
    powers=np.mean(x*x,axis=0)
    ear=int(powers[1]>powers[0])  # Exact ties choose left, deterministically.
    mono=np.repeat(x[:,ear:ear+1],2,axis=1)
    return mono,ear,powers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--video',default=None)
    p.add_argument('--out',type=Path,default=ROOT/'outputs'/'energy_gcc_100ms_summary')
    args=p.parse_args()
    started=time.perf_counter()
    videos=sorted(f for f in args.input.rglob('*') if f.suffix.lower() in {'.mp4','.mkv','.mov','.avi','.webm'})
    if args.video:
        videos=[f for f in videos if f.stem==args.video or f.name==args.video]
        if len(videos)!=1:
            raise ValueError('The selected video must match exactly one file.')
    counts=Counter()
    metrics={k:[] for k in ('E_exc','E_mono')}
    algebra={k:[] for k in ('energy_swap_channel_reorder_max_error','gcc_swap_reversal_max_error','itd_swap_sign_error_us')}
    valid_energy=valid_gcc=strong_gcc=0
    peak_ratios=[]
    left_right_cosines=[]
    comparison_gaps={k:[] for k in ('energy_cosine','total_energy_cosine','gcc_cosine')}
    mono_checks={k:[] for k in ('left_right_waveform_max_difference','copy_source_max_error',
                                'left_right_band_energy_max_difference','absolute_itd_us',
                                'total_energy_increase_db')}
    mono_ear_counts=Counter()
    mono_unchanged=0
    mono_valid_gcc=0
    for video in videos:
        probe=subprocess.run([str(ROOT/'tools'/'ffprobe.exe'),'-v','error','-select_streams','a',
                              '-show_streams','-of','json',str(video)],capture_output=True,check=True,
                             encoding='utf-8',errors='replace')
        streams=[s for s in json.loads(probe.stdout).get('streams',[]) if s.get('channels')==2]
        if not streams:
            counts['no_native_stereo_video']+=1
            continue
        relative=str(video.relative_to(args.input))
        key=hashlib.sha256(relative.encode()).hexdigest()[:12]
        wav=ROOT/'audio_cache'/f'{video.stem}_{key}.wav'
        if not wav.exists() or wav.stat().st_mtime<video.stat().st_mtime:
            wav.parent.mkdir(exist_ok=True)
            part=wav.with_suffix('.part.wav')
            subprocess.run([str(ROOT/'tools'/'ffmpeg.exe'),'-hide_banner','-loglevel','error',
                            '-nostdin','-y','-i',str(video),'-map',f"0:{streams[0]['index']}",
                            '-vn','-ar','16000','-c:a','pcm_s16le',str(part)],check=True,capture_output=True)
            part.replace(wav)
        wave,sr=sf.read(wav,dtype='float32',always_2d=True)
        if sr!=16000 or wave.shape[1]!=2:
            raise RuntimeError('Invalid audio cache.')
        counts['stereo_videos']+=1
        for offset in range(0,len(wave)-1599,1600):
            counts['complete_windows']+=1
            x=wave[offset:offset+1600].astype(np.float64)
            reason=quality_gate(x)
            if reason:
                counts[reason]+=1
                continue
            counts['evaluated_windows']+=1
            original=extract_cues(x)
            swap=extract_cues(x[:,::-1])
            mono_wave,ear,powers=stronger_ear_mono(x)
            mono=extract_cues(mono_wave)
            mono_ear_counts['equal_energy_choose_left' if powers[0]==powers[1] else ('left_stronger' if ear==0 else 'right_stronger')]+=1
            mono_unchanged+=int(np.array_equal(mono_wave,x))
            mono_valid_gcc+=int(mono['gcc_valid'])
            mono_checks['left_right_waveform_max_difference'].append(float(np.max(np.abs(mono_wave[:,0]-mono_wave[:,1]))))
            mono_checks['copy_source_max_error'].append(float(np.max(np.abs(mono_wave[:,0]-x[:,ear]))))
            mono_checks['left_right_band_energy_max_difference'].append(float(np.max(np.abs(mono['left_energy']-mono['right_energy']))))
            if mono['gcc_valid']:
                mono_checks['absolute_itd_us'].append(abs(mono['tau_us']))
            mono_checks['total_energy_increase_db'].append(float(10*np.log10(np.mean(mono_wave**2)/np.mean(x**2))))
            valid_energy+=int(original['energy_valid'])
            if original['left_right_spectral_cosine'] is not None:
                left_right_cosines.append(original['left_right_spectral_cosine'])
            valid_gcc+=int(original['gcc_valid'])
            strong_gcc+=int(original['gcc_valid'] and original['peak_ratio']>=3)
            if original['gcc_valid']:
                peak_ratios.append(original['peak_ratio'])
            for name,cues in [('E_exc',swap),('E_mono',mono)]:
                metrics[name].append(evaluate_pairs(original,cues))
            reordered=np.concatenate((swap['energy'][8:],swap['energy'][:8]))
            algebra['energy_swap_channel_reorder_max_error'].append(float(np.max(np.abs(original['energy']-reordered))))
            if original['gcc_valid'] and swap['gcc_valid']:
                algebra['gcc_swap_reversal_max_error'].append(float(np.max(np.abs(original['gcc']-swap['gcc'][::-1]))))
                algebra['itd_swap_sign_error_us'].append(abs(original['tau_us']+swap['tau_us']))
            for key in comparison_gaps:
                exc_value=metrics['E_exc'][-1][key]
                mono_value=metrics['E_mono'][-1][key]
                if exc_value is not None and mono_value is not None:
                    comparison_gaps[key].append(mono_value-exc_value)
    def aggregate(values):
        result={}
        for name,rows in values.items():
            result[name]={key:describe([r[key] for r in rows if r[key] is not None])
                          for key in ('energy_cosine','total_energy_cosine','gcc_cosine','itd_difference_us')}
        return result
    report=dict(method='Left/right energy cosine + GCC-PHAT',learned_teacher_used=False,
                gpu_required=False,selected_video=args.video,window_ms=100,hop_ms=100,
                sample_rate=16000,counts=dict(counts),
                filters=['overall silence/low activity','severe clipping'],
                removed_filters=['one_channel_nearly_silent','near_phase_inverted_mono','near_dual_mono','weak_direction_cues'],
                energy=dict(bands_per_ear=8,feature_dimensions=16,
                            definition='[E_L,1,...,E_L,8,E_R,1,...,E_R,8]; no independent ear normalization',
                            total_energy_definition='[mean(L^2),mean(R^2)]',
                            band_edges_hz=np.geomspace(100,7500,9).tolist(),
                            stft_samples=512,hop_samples=160,valid_windows=valid_energy,
                            within_window_left_right_spectral_cosine=describe(left_right_cosines),
                            within_window_left_right_undefined_windows=counts['evaluated_windows']-len(left_right_cosines)),
                gcc=dict(max_itd_ms=1,interpolation=8,lag_bin_us=7.8125,vector_size=257,
                         valid_windows=valid_gcc,peak_ratio_ge_3_windows=strong_gcc,
                         peak_ratio=describe(peak_ratios),
                         sign='tau=t_left-t_right; negative means left arrives earlier, not a guaranteed azimuth'),
                variants=dict(E_exc='[R,L]',E_mono='[X_stronger,X_stronger]',
                              stronger_rule='argmax mean(waveform^2) over full 100 ms; exact ties choose left',
                              waveform_copy=True,per_band_selection=False,loudness_compensation=False),
                comparisons=aggregate(metrics),transformation_consistency={k:describe(v) for k,v in algebra.items()},
                mono_diagnostics=dict(source_ear_counts=dict(mono_ear_counts),
                                      unchanged_original_windows=mono_unchanged,
                                      valid_gcc_windows=mono_valid_gcc,
                                      **{k:describe(v) for k,v in mono_checks.items()}),
                mono_minus_exc_cosine_gap={k:describe(v) for k,v in comparison_gaps.items()},
                training_targets=dict(energy='16 jointly sum-normalized nonnegative L/R band energies',
                                      gcc='257 signed normalized lag correlation values with reliability mask',
                                      note='Keep separate heads; joint normalization retains relative ear energy.'),
                suggested_losses=dict(energy='Cosine and/or regression/KL on joint L/R energy distribution',
                                      gcc='Confidence-weighted vector regression or lag-distribution supervision'),
                limitations=['Exchange/copy relations are mathematical properties, not learned localization accuracy.',
                             'Zero energy vectors have undefined cosine and are counted separately.',
                             'Cosine between E_L and E_R alone measures spectral shape, not relative loudness; use the concatenated vector for exchange.',
                             'One silent channel gives undefined delay; the energy vector remains available.',
                             'Centered/identical stereo is a valid invariance control, not automatically a negative.',
                             'Stronger-ear copying removes interaural differences and can increase total energy; no gain correction is applied.',
                             'E_exc and E_mono are transformations, not universally valid negatives or a positive/negative ranking pair.',
                             'Mixtures and reverberation can produce multiple or ambiguous GCC peaks.',
                             'Neither cue encodes sound identity or audiovisual semantic alignment.'],
                elapsed_seconds=time.perf_counter()-started,per_window_outputs=False,
                dit_training_integration_implemented=False)
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 左右耳能量余弦 / GCC-PHAT 汇总评测','',
           f"双声道视频 {counts['stereo_videos']}；完整窗口 {counts['complete_windows']}；评测窗口 {counts['evaluated_windows']}。",'',
           '每个 100 ms 窗口独立计算；只排除整体静音/低活动和严重削波。无需神经网络或 GPU。','',
           '| 变换 | 拼接分频能量余弦均值（16维） | 总左右能量余弦均值（2维） | GCC-PHAT 余弦均值 | ITD 绝对差均值（μs） |',
           '|---|---:|---:|---:|---:|']
    for name,label in [('E_exc','E_exc：交换 L/R'),('E_mono','E_mono：较强声道复制到两侧')]:
        r=report['comparisons'][name]
        cells=[f"{r[key]['mean']:.8g}" if r[key] else '未定义' for key in ('energy_cosine','total_energy_cosine','gcc_cosine','itd_difference_us')]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    ears=describe(left_right_cosines)
    lines+=['',f'能量余弦有定义的窗口 {valid_energy}；GCC 有定义的窗口 {valid_gcc}。',
            f"同窗左右耳 8 维能量向量余弦均值：{ears['mean'] if ears else '未定义'}。此项仅衡量频谱形状，不能单独判断左右响度差。",
            f"复制来源计数：{dict(mono_ear_counts)}；复制后与原始完全相同的窗口：{mono_unchanged}。",
            '较强侧按整个窗口的波形均方能量选择，复制整条波形，不按频带分别选边，不补偿响度。',
            'E_exc 与 E_mono 分别和原始比较，不设正负排序或通过率。原本居中的样本可能不因复制而改变。',
            '未来 DiT 训练可离线缓存这两类目标，接两个小型预测头；本脚本尚未修改 DiT 训练链路。',
            '能量/GCC 不能独自提供声音语义或视觉对象对应关系。仅保存总统计。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
