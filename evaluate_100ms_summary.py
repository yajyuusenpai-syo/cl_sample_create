"""100 ms evaluation, removed stereo-category filters, aggregate outputs only."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import time
import warnings
import numpy as np
import soundfile as sf
import torch
from validate_spatialclap import ROOT, DEFAULT_INPUT, cue_metrics, load_teacher, summarize, command_json

REMOVED=['one_channel_nearly_silent','near_phase_inverted_mono','near_dual_mono']


def gap_summary(positive,negative):
    x=np.asarray(positive)-np.asarray(negative)
    passed=x>1e-4
    y=x[passed]
    return dict(count=len(x),pass_count=int(passed.sum()),
                pass_fraction=float(passed.mean()) if len(x) else None,
                gap_all=summarize(x.tolist()),gap_passed=summarize(y.tolist()),
                passed_gap_le_001_fraction=float(np.mean(y<=.01)) if len(y) else None,
                passed_gap_gt_005_fraction=float(np.mean(y>.05)) if len(y) else None)


def cohort(mask,ss,sm,sp):
    a,b,c=ss[mask],sm[mask],sp[mask]
    return dict(count=int(mask.sum()),swap_cosine=summarize(a.tolist()),
                minus3db_cosine=summarize(b.tolist()),plus3db_cosine=summarize(c.tolist()),
                swap_cosine_ge_0999_count=int(np.sum(a>=.999)),
                swap_cosine_ge_0999_fraction=float(np.mean(a>=.999)) if len(a) else None,
                minus3db_vs_swap=gap_summary(b,a),plus3db_vs_swap=gap_summary(c,a),
                both_gains_vs_swap=gap_summary(np.minimum(b,c),a))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--video',type=str,default=None,help='Evaluate only this exact video filename or stem.')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights'/'audio_weights.pt')
    p.add_argument('--out',type=Path,default=ROOT/'outputs'/'window_100ms_summary')
    p.add_argument('--batch-windows',type=int,default=4)
    args=p.parse_args()
    if args.batch_windows<1:
        p.error('batch-windows must be positive')
    started=time.perf_counter()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    filters=argparse.Namespace(silence_db=-45,min_side_db=-30,min_ild_db=1,max_itd_ms=1,gcc_frame_ms=25)
    extensions={'.mp4','.mkv','.mov','.webm','.avi'}
    videos=sorted(f for f in args.input.rglob('*') if f.suffix.lower() in extensions)
    if args.video:
        videos=[f for f in videos if f.name==args.video or f.stem==args.video]
        if len(videos)!=1:
            raise ValueError('The selected video must match exactly one input file.')
    reasons,errors=Counter(),Counter()
    eligible,diagnostics=[],[]
    total=tails=processed=0
    for video in videos:
        try:
            probe=command_json([ROOT/'tools'/'ffprobe.exe','-v','error','-select_streams','a','-show_streams','-of','json',video])
            stereo=[s for s in probe.get('streams',[]) if s.get('channels')==2]
            if not stereo:
                errors['no_native_stereo_stream']+=1
                continue
            key=hashlib.sha256(str(video.relative_to(args.input)).encode()).hexdigest()[:12]
            wav=ROOT/'audio_cache'/f'{video.stem}_{key}.wav'
            if not wav.exists() or wav.stat().st_mtime<video.stat().st_mtime:
                import subprocess
                part=wav.with_suffix('.part.wav')
                part.parent.mkdir(exist_ok=True)
                subprocess.run([str(ROOT/'tools'/'ffmpeg.exe'),'-hide_banner','-loglevel','error',
                                '-nostdin','-y','-i',str(video),'-map',f"0:{stereo[0]['index']}",
                                '-vn','-ar','16000','-c:a','pcm_s16le',str(part)],check=True,capture_output=True)
                part.replace(wav)
            wave,sr=sf.read(wav,dtype='float32',always_2d=True)
            if sr!=16000 or wave.shape[1]!=2 or not np.isfinite(wave).all():
                raise RuntimeError('Invalid audio cache.')
            processed+=1
            count=len(wave)//1600
            total+=count
            tails+=len(wave)%1600
            for offset in range(0,count*1600,1600):
                segment=wave[offset:offset+1600]
                metrics,_,reason=cue_metrics(segment,16000,filters)
                if reason:
                    reasons[reason]+=1
                if reason in ('silent_or_low_activity','excessive_clipping'):
                    continue
                eligible.append(segment.copy())
                diagnostics.append(dict(
                    one_channel_nearly_silent=min(metrics['left_db'],metrics['right_db'])<-65,
                    near_phase_inverted_mono=metrics['lr_correlation']<-.95,
                    near_dual_mono=metrics['side_ratio_db']<-30,
                    identical_lr=bool(np.array_equal(segment[:,0],segment[:,1])),
                    passes_remaining_direction_filter=not reason))
        except Exception as exc:
            errors[type(exc).__name__]+=1
    report=dict(window_ms=100,hop_ms=100,sample_rate=16000,videos=len(videos),selected_video=args.video,
                native_stereo_videos=processed,video_error_counts=dict(errors),complete_windows=total,
                removed_rejection_rules=REMOVED,
                remaining_filters=dict(overall_rms_db_min=-45,activity_fraction_min=.25,
                                       clipped_fraction_max=.01,mean_abs_band_ild_db_min=1,
                                       or_reliable_nonzero_delay=True,gcc_subframe_ms=25),
                remaining_filter_rejections=dict(reasons),quality_only_coverage_windows=len(eligible),
                primary_remaining_filter_windows=sum(r['passes_remaining_direction_filter'] for r in diagnostics),
                tail_seconds_not_evaluated=tails/16000,teacher_score_filtering=False,per_window_outputs=False,
                metric_definition='gain/exchange cosine ranking with margin=1e-4; not localization accuracy')
    if not eligible:
        report['status']='no_quality_valid_windows'
        (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
        return
    torch.set_num_threads(6)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    sha=hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            sha.update(block)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='torch.meshgrid:.*')
        model,teacher=load_teacher(args.checkpoint,device)

    @torch.inference_mode()
    def encode(wave):
        x=torch.from_numpy(np.ascontiguousarray(wave.transpose(0,2,1))).to(device)
        torch.cuda.synchronize()
        tic=time.perf_counter()
        features,_,_=model.features(x)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-tic
        a=features.float().cpu().numpy()
        if not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):
            raise RuntimeError('Invalid teacher features.')
        return a,elapsed

    reference,_=encode(np.stack(eligible[:1]))
    ss,sm,sp,times=[],[],[],[]
    drift=None
    for offset in range(0,len(eligible),args.batch_windows):
        wave=np.stack(eligible[offset:offset+args.batch_windows])
        inputs=np.concatenate((wave,wave[:,:,::-1],wave*np.float32(10**(-3/20)),wave*np.float32(10**(3/20))),axis=0)
        features,elapsed=encode(inputs)
        a,b,c,d=np.split(features,4)
        if offset==0:
            drift=float(np.linalg.norm(a[0]-reference[0]))
            if drift>1e-4:
                raise RuntimeError(f'Batch/single inconsistency: {drift}')
        ss.extend(np.clip(np.sum(a*b,axis=1),-1,1).tolist())
        sm.extend(np.clip(np.sum(a*c,axis=1),-1,1).tolist())
        sp.extend(np.clip(np.sum(a*d,axis=1),-1,1).tolist())
        times.append(elapsed)
    ss,sm,sp=map(np.asarray,(ss,sm,sp))
    primary=np.asarray([r['passes_remaining_direction_filter'] for r in diagnostics])
    groups={}
    for name in REMOVED+['identical_lr']:
        mask=np.asarray([r[name] for r in diagnostics])
        groups[name]=cohort(mask,ss,sm,sp)
        groups[name]['passes_remaining_direction_filter']=int(np.sum(mask&primary))
        groups[name]['excluded_only_by_remaining_direction_filter']=int(np.sum(mask&~primary))
    report.update(status='completed',teacher=teacher,checkpoint_sha256=sha.hexdigest(),
                  primary_remaining_filter_cohort=cohort(primary,ss,sm,sp),
                  coverage_quality_only_cohort=cohort(np.ones(len(ss),dtype=bool),ss,sm,sp),
                  diagnostic_groups=groups,diagnostic_groups_can_overlap=True,tf32_enabled=False,
                  batch_windows=args.batch_windows,batch_vs_single_reference_l2=drift,
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  teacher_forward_seconds=sum(times),elapsed_seconds=time.perf_counter()-started,
                  interpretation=['Removed categories are only diagnostic labels, never direct rejection rules.',
                                  'The remaining direction gate can exclude some categories; quality-only coverage reports them separately.',
                                  'Identical L/R is an invariance control: cosine near 1 is expected, not a failure.',
                                  'For centered/phase-inverted sound, exchange is not automatically a valid negative.',
                                  'Teacher reliability for independent 100 ms inputs is not established.'])
    assert total==len(eligible)+reasons['silent_or_low_activity']+reasons['excessive_clipping']
    assert int(primary.sum())==total-sum(reasons.values())
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 100 ms 总评测（移除三项过滤）','',
           '单侧近静音、近反相单声道、近双单声道不再直接过滤，仅作为分组标记。','',
           f'源视频 {len(videos)}；完整窗口 {total}；仅排除静音/低活动和严重削波后 {len(eligible)}。',
           f'应用其余方向条件后的窗口数为 {int(primary.sum())}。','',
           '方向线索不足规则仍可能间接排除近单声道，因此同时列出覆盖组。','',
           '| 统计范围 | 窗口数 | 交换余弦均值 | 中位数 | −3 dB 排序通过 | +3 dB 排序通过 | 两种都通过 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in [('primary_remaining_filter_cohort','保留其余筛选'),('coverage_quality_only_cohort','仅音量/削波质量筛选')]:
        c=report[key]
        if c['count']:
            lines.append(f"| {label} | {c['count']} | {c['swap_cosine']['mean']:.6f} | {c['swap_cosine']['median']:.6f} | {c['minus3db_vs_swap']['pass_fraction']:.2%} | {c['plus3db_vs_swap']['pass_fraction']:.2%} | {c['both_gains_vs_swap']['pass_fraction']:.2%} |")
    lines+=['','以上是配对排序比例，不应把所有覆盖组样本都当作有效交换负例。完全相同 L/R 的高相似度是预期表现。','',
            '| 仅分组，不直接过滤 | 窗口数 | 交换余弦均值 | 仍因方向不足排除 |','|---|---:|---:|---:|']
    labels=dict(one_channel_nearly_silent='单侧近静音',near_phase_inverted_mono='近反相单声道',near_dual_mono='近双单声道',identical_lr='完全相同 L/R')
    for name,label in labels.items():
        c=groups[name]
        mean=f"{c['swap_cosine']['mean']:.6f}" if c['swap_cosine'] else '—'
        lines.append(f"| {label} | {c['count']} | {mean} | {c['excluded_only_by_remaining_direction_filter']} |")
    lines+=['','分组可重叠。更多余弦分布、间隔和显存耗时见 summary.json。',
            '仅保存本报告和总统计 JSON，不保存或打印逐样本、逐窗口、逐视频结果。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
