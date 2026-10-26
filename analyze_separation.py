"""Measure cosine separation on correctly ranked gain/exchange examples."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def stats(values):
    x = np.asarray(values,dtype=float)
    if not len(x):
        return None
    return dict(mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),
                p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),max=float(x.max()))


def summarize(rows, positive_keys):
    positive = np.asarray([[float(r[k]) for k in positive_keys] for r in rows])
    # Both-gain cohort uses the less favorable positive similarity for each row.
    positive = positive.min(axis=1) if len(rows) else np.asarray([])
    negative = np.asarray([float(r['swap_cosine']) for r in rows])
    gap = positive-negative
    edges = [0,.001,.01,.05,.1,float('inf')]
    bins = []
    for lo,hi in zip(edges[:-1],edges[1:]):
        count = int(np.sum((gap>lo)&(gap<=hi)))
        bins.append(dict(lower_exclusive=lo,upper_inclusive=hi if np.isfinite(hi) else None,
                         count=count,fraction=count/len(rows) if rows else None))
    return dict(count=len(rows),videos=len(set(r['video'] for r in rows)),
                positive_cosine=stats(positive),swap_cosine=stats(negative),
                cosine_gap=stats(gap),gap_bins=bins,
                swap_cosine_ge_099=int(np.sum(negative>=.99)),
                swap_cosine_ge_095=int(np.sum(negative>=.95)),
                swap_cosine_lt_090=int(np.sum(negative<.9)))


def write_csv(path,rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs'/'gain_plus3db')
    args=p.parse_args()
    with (args.source/'gain_comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    margin=1e-4
    minus=[r for r in rows if float(r['minus3db_minus_swap'])>margin]
    plus=[r for r in rows if float(r['plus3db_minus_swap'])>margin]
    both=[r for r in rows if min(float(r['minus3db_minus_swap']),float(r['plus3db_minus_swap']))>margin]
    groups=dict(minus3db_correct=(minus,['minus3db_cosine']),
                plus3db_correct=(plus,['plus3db_cosine']),
                both_correct_conservative=(both,['minus3db_cosine','plus3db_cosine']))
    result=dict(total_windows=len(rows),correctness_margin=margin,
                definition='cosine_gap = cosine(original,gain) - cosine(original,exchange)',
                both_definition='min(cosine(original,-3dB),cosine(original,+3dB))-cosine(original,exchange)',
                notes=['Positive gaps are expected because this is conditioned on correctly ranked examples.',
                       'The numeric bins describe gap size; they are not validated perceptual or localization thresholds.',
                       'Means can be dominated by strongly separated examples; inspect medians and small-gap fractions.',
                       'Multiple windows from one video are not statistically independent.'])
    for name,(subset,keys) in groups.items():
        result[name]=summarize(subset,keys)
        ordered=[]
        for row in subset:
            r=dict(row)
            r['cosine_gap']=min(float(row[k]) for k in keys)-float(row['swap_cosine'])
            ordered.append(r)
        ordered.sort(key=lambda r:r['cosine_gap'])
        write_csv(args.source/f'{name}_ranked.csv',ordered)
    (args.source/'separation_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 排序正确样本的余弦区分度','',
           'Δ = cos(原始, 增益) − cos(原始, 交换)。Δ 越大，教师对交换的相对区分越明显。','',
           '仅统计 Δ > 0.0001 的窗口。两种增益都通过时，以两者中较小的 Δ 作为保守间隔。','',
           '| 样本组 | 窗口数 | 正例余弦均值 | 交换余弦均值 | Δ 均值 | Δ 中位数 | Δ P10 | Δ P90 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels=dict(minus3db_correct='−3 dB 排序正确',plus3db_correct='+3 dB 排序正确',both_correct_conservative='±3 dB 都正确（保守）')
    for key,label in labels.items():
        r=result[key]
        if not r['count']:
            continue
        g=r['cosine_gap']
        lines.append(f"| {label} | {r['count']} | {r['positive_cosine']['mean']:.6f} | {r['swap_cosine']['mean']:.6f} | {g['mean']:.6f} | {g['median']:.6f} | {g['p10']:.6f} | {g['p90']:.6f} |")
    lines+=['','## 间隔分布','','数值分档仅用于描述，不代表公认的空间定位或听觉阈值。','',
            '| Δ 区间 | −3 dB 正确组 | +3 dB 正确组 | ±3 dB 都正确（保守） |',
            '|---|---:|---:|---:|']
    binlabels=['(0, 0.001]','(0.001, 0.01]','(0.01, 0.05]','(0.05, 0.1]','> 0.1']
    for i,label in enumerate(binlabels):
        cells=[]
        for key in labels:
            b=result[key]['gap_bins'][i]
            cells.append(f"{b['count']}（{b['fraction']:.2%}）" if b['fraction'] is not None else '0')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','各组 *_ranked.csv 按区分间隔从小到大排列，可定位到原视频及起止时间。',
            '仅统计正确样本会偏向表现较好的一侧；需与总通过率一起阅读。余弦间隔不能直接换算为方向角误差。']
    (args.source/'SEPARATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
