"""Aggregate valid-window cosine similarities for S+/S_exc and S+/S_mono."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from spatial_pipeline import logger_for, atomic_jsonl


def find_pairs(root):
    suffixes = ['_S+', '_S_exc', '_S_mono']
    prefixes = sorted({folder.name[:-len(suffix)] for folder in root.iterdir() if folder.is_dir()
                       for suffix in suffixes if folder.name.endswith(suffix)})
    if not prefixes:
        raise ValueError('No immediate subdirectories ending in _S+, _S_exc or _S_mono found')
    groups = {}
    for negative in ['S_exc', 'S_mono']:
        counts = dict(positive_files_total=0, negative_files_total=0, matched_pairs=0,
                      positive_only_files=0, negative_only_files=0)
        pairs = []
        for prefix in prefixes:
            positive_dir, negative_dir = root/f'{prefix}_S+', root/f'{prefix}_{negative}'
            positive = {p.name: p for p in positive_dir.glob('*.pt') if p.is_file()}
            other = {p.name: p for p in negative_dir.glob('*.pt') if p.is_file()}
            matched = sorted(positive.keys() & other.keys())
            counts['positive_files_total'] += len(positive)
            counts['negative_files_total'] += len(other)
            counts['matched_pairs'] += len(matched)
            counts['positive_only_files'] += len(positive.keys()-other.keys())
            counts['negative_only_files'] += len(other.keys()-positive.keys())
            pairs.extend((prefix, positive[name], other[name]) for name in matched)
        groups[negative] = dict(counts=counts, pairs=pairs)
    return groups


def validate_payload(torch, data):
    if not isinstance(data, dict):
        raise ValueError('PT must contain a dictionary')
    features, valid = data.get('features'), data.get('valid_mask')
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 256:
        raise ValueError('features must be a [num_chunk,256] Tensor')
    if not features.is_floating_point():
        raise ValueError('features must have floating dtype')
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != (len(features),):
        raise ValueError('valid_mask must be a bool Tensor with shape [num_chunk]')
    if 'num_chunk' in data and data['num_chunk'] != len(features):
        raise ValueError('num_chunk does not match feature shape')
    return features, valid


def pair_cosines(torch, positive, negative, expected_transform):
    a, valid = validate_payload(torch, positive)
    b, other_valid = validate_payload(torch, negative)
    if a.shape != b.shape:
        raise ValueError('Feature shapes differ between paired files')
    if not torch.equal(valid, other_valid):
        raise ValueError('valid_mask differs between paired files')
    for key in ['video', 'num_chunk', 'chunk_len', 'sample_rate', 'max_duration_ms',
                'num_samples_per_chunk', 'source_num_samples', 'teacher_checkpoint_sha256']:
        if positive.get(key) != negative.get(key):
            raise ValueError(f'Paired metadata differs: {key}')
    for key in ['start_seconds', 'end_seconds', 'real_sample_counts']:
        x, y = positive.get(key), negative.get(key)
        if x is None and y is None:
            continue
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or not torch.equal(x, y):
            raise ValueError(f'Paired metadata differs: {key}')
    if positive.get('transform', 'original') != 'original' or negative.get('transform', expected_transform) != expected_transform:
        raise ValueError('Unexpected transformation metadata for directory pair')
    a, b = a[valid].double(), b[valid].double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Non-finite feature at valid_mask=true')
    if len(a) == 0:
        return np.empty(0, dtype=np.float64)
    na, nb = torch.linalg.vector_norm(a, dim=1), torch.linalg.vector_norm(b, dim=1)
    if (na <= 1e-12).any() or (nb <= 1e-12).any():
        raise ValueError('Zero/near-zero feature at valid_mask=true; cosine undefined')
    return torch.clamp((a*b).sum(dim=1)/(na*nb), -1., 1.).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='The previous inference script output-dir.')
    parser.add_argument('--output-jsonl', type=Path, help='Default: <output-dir>/teacher_similarity_summary.jsonl')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    logger = logger_for('evaluate_teacher_similarity', args.verbose)
    try:
        root = args.output_dir.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        output = (args.output_jsonl or root/'teacher_similarity_summary.jsonl').expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f'{output} already exists; use --overwrite')
        groups = find_pairs(root)
        # Print both match summaries before loading any feature tensors.
        for key, label in [('S_exc', '+/exc'), ('S_mono', '+/mono')]:
            c = groups[key]['counts']
            logger.info('%s: %d total, %d matched; negative total=%d, positive-only=%d, negative-only=%d',
                        label, c['positive_files_total'], c['matched_pairs'], c['negative_files_total'],
                        c['positive_only_files'], c['negative_only_files'])
        import torch
        results = []
        for key, label, transform in [('S_exc', 'S+/S_exc', 'lr_swap'), ('S_mono', 'S+/S_mono', 'samplewise_max')]:
            group = groups[key]
            scores = []
            errors = Counter()
            processed = empty = 0
            for prefix, pos_file, neg_file in group['pairs']:
                try:
                    positive = torch.load(pos_file, map_location='cpu', weights_only=True)
                    negative = torch.load(neg_file, map_location='cpu', weights_only=True)
                    values = pair_cosines(torch, positive, negative, transform)
                    processed += 1
                    empty += int(len(values) == 0)
                    if len(values):
                        scores.append(values)
                except Exception as e:
                    errors[f'{type(e).__name__}: {e}'] += 1
                    # Details stay in the log; console has only aggregate counts.
                    logger.debug('Pair failed dataset=%s positive=%s negative=%s', prefix, pos_file, neg_file, exc_info=True)
            values = np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)
            result = dict(comparison=label, **group['counts'], processed_pairs=processed,
                          failed_pairs=sum(errors.values()), pairs_with_no_valid_windows=empty,
                          valid_windows=len(values), mean=None, median=None, p90=None, p95=None,
                          weighting='one vote per aligned valid window across all matched videos',
                          percentile_method='linear', error_counts=dict(errors),
                          status='complete' if len(values) and not errors else ('partial' if len(values) else 'no_valid_windows'))
            if len(values):
                result.update(mean=float(values.mean()), median=float(np.median(values)),
                              p90=float(np.percentile(values, 90, method='linear')),
                              p95=float(np.percentile(values, 95, method='linear')))
                logger.info('%s: processed=%d failed=%d valid_windows=%d mean=%.8f median=%.8f P90=%.8f P95=%.8f',
                            label, processed, sum(errors.values()), len(values), result['mean'], result['median'], result['p90'], result['p95'])
            else:
                logger.warning('%s: processed=%d failed=%d valid_windows=0; statistics=null', label, processed, sum(errors.values()))
            if errors:
                logger.warning('%s aggregate errors: %s (details in running_log)', label, dict(errors))
            results.append(result)
        atomic_jsonl(output, results, overwrite=args.overwrite)
        logger.info('Summary JSONL: %s', output)
        return 0 if all(r['status'] == 'complete' for r in results) else 2
    except Exception:
        logger.exception('Fatal teacher similarity evaluation error')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
