import os, sys, json, math
from typing import List, Dict, Any

ENTITY = 'kjust-pinduoduo'
PROJECT = 'lerobot'
RUNS = ['ufiskzhp','7xv5wewo','uazctjvr']
METRICS = {
    'train_ot': 'train/ot_cost/action_lbl',
    'eval_loss': 'eval/offline_eval/avg_loss',
}

# Try imports, install hint if missing
try:
    import wandb
except Exception as e:
    print('ERR: wandb not installed', file=sys.stderr)
    sys.exit(3)

api = wandb.Api()

results: Dict[str, Any] = {}

for rid in RUNS:
    key = f"{ENTITY}/{PROJECT}/{rid}"
    try:
        run = api.run(key)
    except Exception as e:
        results[rid] = {'error': f'cannot access run: {e}'}
        continue

    # Fetch limited keys; use scan_history to stream
    rows = []
    try:
        # Only request the keys we need; some runs may not log global_step
        for row in run.scan_history(keys=['_step', METRICS['train_ot'], METRICS['eval_loss']], page_size=1000):
            rows.append(row)
    except Exception as e:
        results[rid] = {'error': f'history fetch failed: {e}'}
        continue

    def series(key: str):
        xs = []
        ys = []
        for r in rows:
            y = r.get(key)
            if y is None:
                continue
            step = r.get('_step')
            xs.append(step if step is not None else len(xs))
            ys.append(y)
        return xs, ys

    import numpy as np

    def jitter_stats(y: List[float]):
        arr = np.asarray(y, dtype=float)
        if arr.size == 0:
            return None
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        cv = float(std / (abs(mean) + 1e-8))
        # Median absolute relative change
        if arr.size >= 2:
            prev = arr[:-1]
            cur = arr[1:]
            marr = np.median(np.abs(cur - prev) / (np.abs(prev) + 1e-8))
        else:
            marr = float('nan')
        # Rolling std (window ~2% of series length, min 5, max 200)
        w = int(max(5, min(200, round(len(arr) * 0.02))))
        rs = []
        if len(arr) >= w:
            for i in range(w, len(arr)+1):
                rs.append(np.std(arr[i-w:i]))
            roll_std_mean = float(np.mean(rs))
        else:
            roll_std_mean = float('nan')
        return {
            'count': int(arr.size),
            'last': float(arr[-1]),
            'mean': mean,
            'std': std,
            'cv': cv,
            'median_abs_rel_change': float(marr),
            'roll_std_mean': roll_std_mean,
        }

    def trend_stats(x: List[float], y: List[float]):
        xarr = np.asarray(x, dtype=float)
        yarr = np.asarray(y, dtype=float)
        if yarr.size == 0:
            return None
        minv = float(np.nanmin(yarr))
        last = float(yarr[-1])
        delta = float(last - minv)
        # slope on last 20% (>=20 points) else last 100 points
        n = len(yarr)
        k = max(20, int(n * 0.2))
        idx0 = max(0, n - k)
        xs = xarr[idx0:]
        ys = yarr[idx0:]
        if xs.size >= 2:
            # normalize x for numerical stability
            xs0 = xs - xs[0]
            try:
                m = float(np.polyfit(xs0, ys, 1)[0])
            except Exception:
                m = float('nan')
        else:
            m = float('nan')
        return {
            'count': int(n),
            'last': last,
            'min': minv,
            'delta_last_minus_min': delta,
            'slope_recent': m,
        }

    tx, ty = series(METRICS['train_ot'])
    ex, ey = series(METRICS['eval_loss'])

    results[rid] = {
        'train_ot': jitter_stats(ty),
        'eval_loss': trend_stats(ex, ey),
        'points_train_ot': len(ty),
        'points_eval_loss': len(ey),
    }

# Persist
os.makedirs('/workspace/.codex_tmp/wandb_runs', exist_ok=True)
with open('/workspace/.codex_tmp/wandb_runs/dp_ot_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Also emit a markdown summary fragment for direct append
lines = []
for rid, val in results.items():
    lines.append(f"### Run {rid}")
    if 'error' in val:
        lines.append(f"- Error: {val['error']}")
        continue
    to = val.get('train_ot')
    el = val.get('eval_loss')
    if to:
        lines.append("- train/ot_cost/action_lbl: "
                     f"n={to['count']}, last={to['last']:.4g}, mean={to['mean']:.4g}, std={to['std']:.4g}, cv={to['cv']:.3g}, "
                     f"med_abs_rel_change={to['median_abs_rel_change']:.3g}")
    else:
        lines.append("- train/ot_cost/action_lbl: no data")
    if el:
        lines.append("- eval/offline_eval/avg_loss: "
                     f"n={el['count']}, last={el['last']:.4g}, min={el['min']:.4g}, delta(last-min)={el['delta_last_minus_min']:.4g}, "
                     f"slope_recent={el['slope_recent']:.4g}")
    else:
        lines.append("- eval/offline_eval/avg_loss: no data")
    lines.append("")

with open('/workspace/.codex_tmp/wandb_runs/dp_ot_summary.md', 'w') as f:
    f.write('\n'.join(lines))

print('OK')
