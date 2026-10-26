"""Screen stereo videos by whole-window power and absolute interaural level difference."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import tempfile
import numpy as np
from spatial_pipeline import (PROJECT_ROOT, SAMPLE_RATE, MAX_DURATION_MS, SCHEMA_VERSION,
    VIDEO_EXTENSIONS, logger_for, get_ffmpeg, run_ffmpeg, decode_audio, window_grid,
    make_windows, energy_statistics, finite_float, atomic_json, atomic_jsonl)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, help='Video directory (non-recursive).')
    p.add_argument('--input-txt', type=Path, help='One video path per line; takes priority over --input-dir.')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--chunk-len', type=int, default=256, help='Window length in ms; default 256.')
    p.add_argument('--min-ild-db', type=finite_float, default=0.5, help='Minimum absolute L/R energy difference in dB.')
    p.add_argument('--min-total-dbfs', type=finite_float, default=-45., help='Minimum 10log10((E_L+E_R)/2), default -45 dBFS.')
    p.add_argument('--sample-count', type=int, default=100, help='Number of globally random valid windows to export.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ffmpeg', type=str)
    p.add_argument('--overwrite', action='store_true', help='Replace matching outputs; unrelated files are preserved.')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    if not args.input_txt and not args.input_dir:
        p.error('Specify --input-dir or --input-txt')
    try:
        window_grid(args.chunk_len)
    except ValueError as e:
        p.error(str(e))
    if args.min_ild_db < 0 or args.sample_count < 0:
        p.error('min-ild-db and sample-count must be nonnegative')
    return args


def find_inputs(args, logger):
    if args.input_txt:
        source = args.input_txt.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if args.input_dir:
            logger.info('--input-txt takes priority; ignoring --input-dir=%s', args.input_dir)
        paths = []
        for line in source.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            path = Path(line).expanduser()
            paths.append((path if path.is_absolute() else source.parent/path).resolve())
        if not paths:
            raise ValueError('TXT contains no video paths')
        if len({p.parent for p in paths}) != 1:
            raise ValueError('TXT videos must share one parent directory, as required by the feature output naming contract')
        directory = paths[0].parent
    else:
        directory = args.input_dir.expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        paths = sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    unique = list(dict.fromkeys(paths))
    if len(unique) != len(paths):
        logger.warning('Removed %d duplicate input paths', len(paths)-len(unique))
    stems = {}
    for path in unique:
        if path.stem in stems and stems[path.stem] != path:
            raise ValueError(f'Duplicate video stem would collide: {stems[path.stem]} and {path}')
        stems[path.stem] = path
    return unique, directory


def export_clip(ffmpeg, video, index, chunk_len, out, logger, overwrite):
    if out.exists() and not overwrite:
        raise FileExistsError(f'{out} exists; use --overwrite')
    fd, temporary = tempfile.mkstemp(prefix=f'.{out.stem}.', suffix='.mp4', dir=out.parent)
    os.close(fd)
    try:
        command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                   '-ss', f'{index*chunk_len/1000:.6f}', '-i', str(video), '-t', f'{chunk_len/1000:.6f}',
                   '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', temporary]
        run_ffmpeg(command, logger)
        if Path(temporary).stat().st_size == 0:
            raise RuntimeError('FFmpeg created an empty clip')
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    args = parse_args()
    logger = logger_for('filter_spatial_audio', args.verbose)
    try:
        logger.info('Arguments: %s', vars(args))
        paths, directory = find_inputs(args, logger)
        if not paths:
            raise ValueError(f'No supported videos found in {directory}')
        output = (args.output_dir or PROJECT_ROOT/f'output_{directory.name}').expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        jsonl = output/'filtered.jsonl'
        if jsonl.exists() and not args.overwrite:
            raise FileExistsError(f'{jsonl} already exists; use --overwrite')
        ffmpeg = get_ffmpeg(args.ffmpeg)
        logger.info('FFmpeg: %s', run_ffmpeg([ffmpeg, '-version'], logger).stdout.decode(errors='replace').splitlines()[0])
        count, width = window_grid(args.chunk_len)
        logger.info('Grid: %d chunks x %d ms (%d samples), cap=%d ms; padding chunks have valid_mask=false', count, args.chunk_len, width, MAX_DURATION_MS)
        rows, valid_windows, errors = [], [], []
        stats = dict(input_videos=len(paths), retained_videos=0, no_valid_window_videos=0,
                     valid_windows=0, complete_windows=0, padding_windows=0,
                     below_total_threshold_complete_windows=0, below_ild_threshold_complete_windows=0)
        for number, video in enumerate(paths, 1):
            try:
                audio = decode_audio(video, ffmpeg, logger)
                windows, complete, real_counts = make_windows(audio, args.chunk_len)
                total, ild = energy_statistics(windows)
                valid = complete & (total >= args.min_total_dbfs) & (abs(ild) >= args.min_ild_db)
                stats['complete_windows'] += int(complete.sum())
                stats['padding_windows'] += int((~complete).sum())
                stats['below_total_threshold_complete_windows'] += int((complete & (total < args.min_total_dbfs)).sum())
                stats['below_ild_threshold_complete_windows'] += int((complete & (abs(ild) < args.min_ild_db)).sum())
                logger.info('[%d/%d] %s decoded=%.6fs valid=%d/%d padding=%d', number, len(paths), video.name, len(audio)/SAMPLE_RATE, valid.sum(), count, (~complete).sum())
                logger.debug('%s total_dbfs=%s signed_ild_db=%s valid_mask=%s', video, total.tolist(), ild.tolist(), valid.tolist())
                if not valid.any():
                    stats['no_valid_window_videos'] += 1
                    continue
                row = dict(video=str(video), num_chunk=count, chunk_len=args.chunk_len, valid_mask=valid.tolist(),
                           schema_version=SCHEMA_VERSION, sample_rate=SAMPLE_RATE, max_duration_ms=MAX_DURATION_MS,
                           num_samples_per_chunk=width, source_num_samples=len(audio), real_sample_counts=real_counts.tolist(),
                           screening=dict(min_ild_db=args.min_ild_db, min_total_dbfs=args.min_total_dbfs,
                                          total_definition='10log10((mean(L^2)+mean(R^2))/2)',
                                          total_dbfs=total.tolist(), signed_ild_db=ild.tolist()),
                           padding_rule='repeat_last_complete_source_chunk; every partially/fully padded chunk has valid_mask=false')
                rows.append(row)
                valid_windows.extend((video, int(i)) for i in np.flatnonzero(valid))
            except Exception as e:
                logger.exception('[%d/%d] Failed: %s', number, len(paths), video)
                errors.append(dict(video=str(video), error=str(e)))
        atomic_jsonl(jsonl, rows, args.overwrite)
        stats.update(retained_videos=len(rows), valid_windows=len(valid_windows))
        sampled = random.Random(args.seed).sample(valid_windows, min(args.sample_count, len(valid_windows)))
        if len(valid_windows) < args.sample_count:
            logger.info('Only %d valid windows; exporting all without replacement', len(valid_windows))
        exports = []
        for number, (video, index) in enumerate(sampled, 1):
            out = output/f'{video.stem}_{index}.mp4'
            record = dict(video=str(video), window_index=index, start_seconds=index*args.chunk_len/1000,
                          chunk_len=args.chunk_len, output=str(out))
            try:
                export_clip(ffmpeg, video, index, args.chunk_len, out, logger, args.overwrite)
                record['status'] = 'saved'
                logger.info('Sample [%d/%d] %s', number, len(sampled), out.name)
            except Exception as e:
                record.update(status='failed', error=str(e))
                logger.exception('Sample export failed: %s', out)
            exports.append(record)
        stats.update(jsonl=str(jsonl), seed=args.seed, chunk_len=args.chunk_len, num_chunk=count,
                     sample_exports=exports, errors=errors, video_directory=str(directory),
                     min_ild_db=args.min_ild_db, min_total_dbfs=args.min_total_dbfs)
        atomic_json(output/'filter_summary.json', stats)
        logger.info('DONE JSONL=%s retained=%d/%d valid_windows=%d exported=%d/%d errors=%d', jsonl, len(rows), len(paths), len(valid_windows), sum(x['status']=='saved' for x in exports), len(sampled), len(errors))
        return 0 if rows and not errors and all(x['status']=='saved' for x in exports) else 2
    except Exception:
        logger.exception('Fatal screening error')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
