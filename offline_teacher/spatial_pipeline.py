"""Shared decoding, fixed-grid windowing, logging and JSONL contracts."""
from __future__ import annotations
import io
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
import warnings
import numpy as np
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
MAX_DURATION_MS = 10060
SCHEMA_VERSION = 1
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v', '.mpeg', '.mpg'}


def logger_for(name: str, verbose: bool = False):
    folder = PROJECT_ROOT / 'running_log'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log'
    logger = logging.getLogger(f'{name}.{os.getpid()}')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    file = logging.FileHandler(path, encoding='utf-8')
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    logger.handlers = [console, file]
    logger.propagate = False
    logger.info('Log file: %s', path)
    return logger


def get_ffmpeg(explicit=None):
    if explicit:
        candidate = shutil.which(str(explicit)) or str(Path(explicit).expanduser())
        if not Path(candidate).is_file():
            raise FileNotFoundError(f'FFmpeg not found: {explicit}')
        return str(Path(candidate).resolve())
    configured = os.environ.get('SPATIAL_CLAP_FFMPEG')
    if configured:
        return get_ffmpeg(configured)
    if os.name == 'nt' and (PROJECT_ROOT / 'tools/ffmpeg.exe').is_file():
        return str(PROJECT_ROOT / 'tools/ffmpeg.exe')
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        candidate = shutil.which('ffmpeg')
        if candidate:
            return candidate
        raise RuntimeError('No FFmpeg found. Install the offline wheel bundle or pass --ffmpeg.')


def run_ffmpeg(command, logger, timeout=180):
    logger.debug('FFmpeg argv: %r', command)
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    if result.returncode:
        error = result.stderr.decode('utf-8', errors='replace')[-8000:]
        raise RuntimeError(f'FFmpeg exit {result.returncode}: {error}')
    return result


def decode_audio(video: Path, ffmpeg: str, logger):
    if not video.is_file():
        raise FileNotFoundError(video)
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(video),
               '-map', '0:a:0', '-vn', '-t', '10.06', '-ar', str(SAMPLE_RATE),
               '-c:a', 'pcm_f32le', '-f', 'wav', 'pipe:1']
    result = run_ffmpeg(command, logger)
    # A piped WAV has unknown RIFF length; FFmpeg success and decoded sample count
    # establish completeness. Do not use '-ac 2', which would silently upmix mono.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavfile.WavFileWarning)
        sr, audio = wavfile.read(io.BytesIO(result.stdout))
    if sr != SAMPLE_RATE or audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f'Native stereo required; decoded shape={audio.shape}, sample_rate={sr}')
    audio = np.asarray(audio[:MAX_DURATION_MS * SAMPLE_RATE // 1000], dtype=np.float32)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite decoded audio')
    return audio


def window_grid(chunk_len: int):
    if isinstance(chunk_len, bool) or not isinstance(chunk_len, int) or chunk_len <= 0:
        raise ValueError('chunk_len must be a positive integer in milliseconds')
    samples = SAMPLE_RATE * chunk_len // 1000
    if samples <= 512:
        raise ValueError('chunk_len must be >=33 ms: the teacher requires >512 samples at 16 kHz')
    count = math.ceil(MAX_DURATION_MS / chunk_len)
    return count, samples


def make_windows(audio, chunk_len):
    count, width = window_grid(chunk_len)
    audio = audio[:MAX_DURATION_MS * SAMPLE_RATE // 1000]
    n = len(audio)
    if n == 0:
        raise ValueError('Empty source audio')
    output = np.empty((count, width, 2), dtype=np.float32)
    full = n // width
    # Repeat the last complete source chunk; a sub-chunk source repeats itself.
    repeat_source = audio[(full-1)*width:full*width] if full else audio
    real_counts = np.zeros(count, dtype=np.int64)
    for i in range(count):
        start = i * width
        available = max(0, min(width, n-start))
        real_counts[i] = available
        if available:
            output[i, :available] = audio[start:start+available]
        missing = width-available
        if missing:
            output[i, available:] = np.tile(repeat_source, (math.ceil(missing/len(repeat_source)), 1))[:missing]
    complete = real_counts == width
    return output, complete, real_counts


def energy_statistics(windows):
    power = np.mean(windows.astype(np.float64)**2, axis=1)
    eps = 1e-12
    total_dbfs = 10 * np.log10(np.maximum(power.mean(axis=1), eps))
    signed_ild_db = 10 * np.log10((power[:, 0]+eps)/(power[:, 1]+eps))
    return total_dbfs, signed_ild_db


def finite_float(value):
    import argparse
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError('Threshold must be finite')
    return number


def atomic_json(path, data, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists; use --overwrite')
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_jsonl(path, rows, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists; use --overwrite')
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_row(row):
    if not isinstance(row, dict):
        raise ValueError('JSONL record must be an object')
    for key in ['video', 'num_chunk', 'chunk_len', 'valid_mask']:
        if key not in row:
            raise ValueError(f'Missing JSONL key: {key}')
    if not isinstance(row['video'], str) or not Path(row['video']).is_absolute():
        raise ValueError('video must be an absolute path string on the current OS')
    count, width = window_grid(row['chunk_len'])
    if isinstance(row['num_chunk'], bool) or not isinstance(row['num_chunk'], int) or row['num_chunk'] != count:
        raise ValueError(f'num_chunk must equal ceil(10060/chunk_len)={count}')
    mask = row['valid_mask']
    if not isinstance(mask, list) or len(mask) != count or any(type(v) not in [bool, int] or v not in [0, 1] for v in mask):
        raise ValueError('valid_mask must contain num_chunk boolean/0/1 values')
    if row.get('sample_rate', SAMPLE_RATE) != SAMPLE_RATE or row.get('max_duration_ms', MAX_DURATION_MS) != MAX_DURATION_MS:
        raise ValueError('Unsupported sample_rate/max_duration_ms in JSONL')
    if row.get('num_samples_per_chunk', width) != width:
        raise ValueError('num_samples_per_chunk does not match chunk_len')
    if row.get('schema_version', SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError('Unsupported JSONL schema version')
    return np.asarray(mask, dtype=bool)
