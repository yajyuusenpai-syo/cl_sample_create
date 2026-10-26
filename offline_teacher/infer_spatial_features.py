"""JSONL -> one [num_chunk,256] PT dictionary per video and transformation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from spatial_pipeline import (PROJECT_ROOT, SAMPLE_RATE, MAX_DURATION_MS, SCHEMA_VERSION,
    logger_for, get_ffmpeg, run_ffmpeg, decode_audio, make_windows, validate_row, atomic_json)


def atomic_pt(torch, payload, path):
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, help='Default: parent of the input video directory.')
    p.add_argument('--device', default='cuda', help='cuda, cuda:0, or cpu; no automatic CPU fallback.')
    p.add_argument('--batch-size', type=int, default=32, help='Valid windows per forward per transformation.')
    p.add_argument('--checkpoint', type=Path, help='Optional spatial or original full audio checkpoint.')
    p.add_argument('--ffmpeg', type=str)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    logger = logger_for('infer_spatial_features', args.verbose)
    try:
        import torch
        from spatial_teacher import SpatialTeacher
        logger.info('Arguments: %s', vars(args))
        rows = []
        seen = set()
        source = args.input_jsonl.expanduser().resolve()
        for line_no, line in enumerate(source.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                valid = validate_row(row)
                video = Path(row['video']).expanduser().resolve()
                if video in seen:
                    raise ValueError(f'Duplicate video: {video}')
                seen.add(video)
                if not valid.any():
                    logger.warning('Line %d has no valid windows; skipped: %s', line_no, video)
                    continue
                row['video'] = str(video)
                rows.append(row)
            except Exception as e:
                raise ValueError(f'Invalid JSONL at line {line_no}: {e}') from e
        if not rows:
            raise ValueError('No records containing valid windows')
        parents = {Path(r['video']).parent for r in rows}
        if len(parents) != 1:
            raise ValueError('All JSONL videos must share the same directory')
        video_dir = next(iter(parents))
        if len({Path(r['video']).stem for r in rows}) != len(rows):
            raise ValueError('Duplicate video stems would overwrite PT outputs')
        output = (args.output_dir or video_dir.parent).expanduser().resolve()
        variants = {'S+': 'original', 'S_exc': 'lr_swap', 'S_mono': 'samplewise_max'}
        folders = {key: output/f'{video_dir.name}_{key}' for key in variants}
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        logger.info('Output folders: %s', {k: str(v) for k, v in folders.items()})
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; pass --device cpu only for explicit CPU testing')
        if device.type not in ['cuda', 'cpu']:
            raise ValueError('Supported devices: cuda[:index] or cpu')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        checkpoint = (args.checkpoint or Path(__file__).resolve().parent/'weights/spatial_encoder.pt').expanduser().resolve()
        model = SpatialTeacher(checkpoint).to(device)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ffmpeg = get_ffmpeg(args.ffmpeg)
        logger.info('Teacher: parameters=%d checkpoint=%s sha256=%s device=%s torch=%s cuda=%s', sum(x.numel() for x in model.parameters()), checkpoint, checkpoint_hash, device, torch.__version__, torch.version.cuda)
        logger.info('FFmpeg: %s', run_ffmpeg([ffmpeg, '-version'], logger).stdout.decode(errors='replace').splitlines()[0])
        logger.info('S_mono uses signed samplewise max(L,R), without gain compensation or per-channel normalization')
        saved, errors = [], []
        for number, row in enumerate(rows, 1):
            video = Path(row['video'])
            try:
                paths = {k: f/f'{video.stem}.pt' for k, f in folders.items()}
                if not args.overwrite and any(p.exists() for p in paths.values()):
                    raise FileExistsError(f'Output exists for {video.stem}; use --overwrite')
                audio = decode_audio(video, ffmpeg, logger)
                windows, complete, real_counts = make_windows(audio, row['chunk_len'])
                valid = validate_row(row)
                if (valid & ~complete).any():
                    raise ValueError('JSONL valid_mask marks a padded chunk as valid; re-run screening')
                if row.get('source_num_samples', len(audio)) != len(audio):
                    raise ValueError('Decoded source length changed since screening; re-run screening')
                if 'real_sample_counts' in row and row['real_sample_counts'] != real_counts.tolist():
                    raise ValueError('Source window timing changed since screening')
                indexes = np.flatnonzero(valid)
                count = row['num_chunk']
                banks = {k: torch.zeros((count, 256), dtype=torch.float32) for k in variants}
                for start in range(0, len(indexes), args.batch_size):
                    selected = indexes[start:start+args.batch_size]
                    original = np.ascontiguousarray(windows[selected].transpose(0, 2, 1))
                    mono = np.maximum(original[:, 0:1], original[:, 1:2])
                    inputs = {'S+': original, 'S_exc': original[:, ::-1], 'S_mono': np.repeat(mono, 2, axis=1)}
                    for key, samples in inputs.items():
                        features = model(torch.from_numpy(np.ascontiguousarray(samples))).cpu().float()
                        if features.shape != (len(selected), 256) or not torch.isfinite(features).all():
                            raise RuntimeError(f'Invalid teacher features for {key}')
                        if not torch.allclose(features.norm(dim=-1), torch.ones(len(selected)), atol=1e-5):
                            raise RuntimeError(f'Non-normalized teacher features for {key}')
                        banks[key][torch.from_numpy(selected)] = features
                    logger.debug('%s valid-window batch %d:%d / %d', video.name, start, start+len(selected), len(indexes))
                common = dict(schema_version=SCHEMA_VERSION, video=str(video), num_chunk=count,
                              chunk_len=row['chunk_len'], sample_rate=SAMPLE_RATE, max_duration_ms=MAX_DURATION_MS,
                              num_samples_per_chunk=windows.shape[1], source_num_samples=len(audio),
                              valid_mask=torch.tensor(valid, dtype=torch.bool), real_sample_counts=torch.from_numpy(real_counts.copy()),
                              start_seconds=torch.arange(count, dtype=torch.float64)*row['chunk_len']/1000,
                              end_seconds=torch.arange(1, count+1, dtype=torch.float64)*row['chunk_len']/1000,
                              feature_dim=256, normalized=True, teacher_checkpoint_sha256=checkpoint_hash,
                              teacher_name='SpatialCLAP.independent_spatial_encoder', screening=row.get('screening', {}),
                              padding_rule=row.get('padding_rule', 'repeat_last_complete_source_chunk'))
                for key, transform in variants.items():
                    if torch.count_nonzero(banks[key][~common['valid_mask']]):
                        raise RuntimeError('Expected exact zero placeholder at valid_mask=false')
                    atomic_pt(torch, dict(common, features=banks[key], transform=transform), paths[key])
                saved.append(dict(video=str(video), shape=[count, 256], valid_chunks=len(indexes), paths={k: str(v) for k, v in paths.items()}))
                logger.info('[%d/%d] %s saved 3 PTs, shape=[%d,256], valid=%d padding=%d', number, len(rows), video.name, count, valid.sum(), (~complete).sum())
            except Exception as e:
                logger.exception('[%d/%d] Failed: %s', number, len(rows), video)
                errors.append(dict(video=str(video), error=str(e)))
        summary = dict(input_jsonl=str(source), video_directory=str(video_dir), output_dir=str(output),
                       device=str(device), teacher_checkpoint_sha256=checkpoint_hash, saved=saved, errors=errors)
        report = output/f'{video_dir.name}_spatial_inference_summary.json'
        atomic_json(report, summary)
        logger.info('DONE successful_videos=%d/%d errors=%d summary=%s', len(saved), len(rows), len(errors), report)
        return 0 if saved and not errors else 2
    except Exception:
        logger.exception('Fatal feature inference error')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
