# -*- coding: utf-8 -*-
"""Step 2. Uncertainty and validation of the reference.

  * expected positions fitted without the series concerned (leave-one-series-out);
  * spread of the same target across the baseline series of a session (drive + tracker + overlay exchange);
  * Monte Carlo propagation of the geometry uncertainties to the reported angles;
  * angles referred to the centre of rotation (12 mm behind the corneal apex) instead of the apex;
  * moving-block bootstrap of the filter shifts;
  * bootstrap confidence intervals of the rotation response;
  * slopes of the pupil-centre regression with t-based 95 % intervals for the first drooping-eyelid series.

The random draws are made in a fixed order from one seeded generator, so the results are reproducible.
Inputs: results/series.csv, results/cells.csv, results/summary.json. Output: results/uncertainty.json"""
import json
import math
import os

import numpy as np
import pandas as pd

from common import (RESULTS, GEOM, MARGIN, BOUND_DEG, MIN_CELL, MIN_MEAN, N_TARGETS, BASELINE, DROOPING,
                    read_gaze, window, out_of_bounds, angle, fit_affine, apply_affine)

rng = np.random.default_rng(20261109)
S = pd.read_csv(os.path.join(RESULTS, "series.csv")); K = pd.read_csv(os.path.join(RESULTS, "cells.csv"))
L = json.load(open(os.path.join(RESULTS, "summary.json"), encoding="utf-8"))
ses = S.set_index("run_id").session
BASE = S[S.group == BASELINE].run_id.tolist()
U = {}

DAT = {n: read_gaze(n) for n in S.run_id}
CELL = {}   # (series, k) -> (column, row, median x, median y) for cells with >= 30 BPOGV-valid samples
for n in S.run_id:
    D = DAT[n]
    for k in range(N_TARGETS):
        s = D["BPOGV"] & window(D["t"], k)
        if s.sum() >= MIN_MEAN:
            CELL[(n, k)] = (k % 5, k // 5, float(np.median(D["x"][s])), float(np.median(D["y"][s])))
MAP = {d: fit_affine([v for (n, k), v in CELL.items() if n in BASE and ses[n] == d])[:2] for d in sorted(set(ses))}
expected = lambda d, c, r: apply_affine(MAP[d], c, r)

# ---------------------------------------------------------------- leave-one-series-out
e_in, e_out = [], []
for n in BASE:
    mo = fit_affine([v for (m, k), v in CELL.items() if m in BASE and ses[m] == ses[n] and m != n])[:2]
    for (m, k), (c, r, mx, my) in CELL.items():
        if m != n:
            continue
        e_in.append(float(angle(mx, my, *expected(ses[n], c, r)))); e_out.append(float(angle(mx, my, *apply_affine(mo, c, r))))
p99_out = float(np.percentile(e_out, 99))
kd = K[K.group == DROOPING]
q = lambda a: dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), max=float(max(a)))
U["leave_one_series_out"] = dict(n=len(e_in), full_fit=q(e_in), without_series=q(e_out),
                                 drooping_cells_above_p99_full_fit=int((kd.err_median > np.percentile(e_in, 99)).sum()),
                                 drooping_cells_above_p99_without_series=int((kd.err_median > p99_out).sum()), drooping_cells=int(len(kd)),
                                 baseline_cells_per_session={d: int(sum(1 for (n, k) in CELL if n in BASE and ses[n] == d)) for d in sorted(set(ses))})

# ---------------------------------------------------------------- spread of the same target across baseline series
per_ses = {}; allv = []
for d in sorted(set(ses)):
    dev = []
    for k in range(N_TARGETS):
        P = np.array([CELL[(n, k)][2:] for n in BASE if ses[n] == d and (n, k) in CELL])
        if len(P) < 3:
            continue
        dev += list(angle(P[:, 0], P[:, 1], np.median(P[:, 0]), np.median(P[:, 1])))
    dev = np.array(dev); allv += list(dev)
    per_ses[d] = dict(n_series=int(sum(1 for n in BASE if ses[n] == d)), n=int(len(dev)), median=float(np.median(dev)),
                      p95=float(np.percentile(dev, 95)), rms=float(np.sqrt(np.mean(dev ** 2))))
allv = np.array(allv)
U["target_spread_across_series"] = dict(sessions=per_ses, all=dict(n=int(len(allv)), median=float(np.median(allv)),
                                                                   p95=float(np.percentile(allv, 95)), rms=float(np.sqrt(np.mean(allv ** 2)))))

# ---------------------------------------------------------------- Monte Carlo of the geometry
UG = dict(W=0.5, H=0.5, D=5.0, dx=2.0, dy=2.0)   # standard uncertainties [mm]
U["geometry_standard_uncertainties_mm"] = UG
kb = [(n, k) for (n, k) in CELL if n in BASE]
X1 = np.array([CELL[c][2:] for c in kb]); E1 = np.array([expected(ses[c[0]], *CELL[c][:2]) for c in kb])
FK = L["filter_cells"]


def cell_samples(n, k):
    D = DAT[n]; s = D["BPOGV"] & window(D["t"], k); x, y = D["x"][s], D["y"][s]
    return x, y, out_of_bounds(x, y), angle(x, y, np.median(x), np.median(y)) <= BOUND_DEG


def quantities(g):
    e = angle(X1[:, 0], X1[:, 1], E1[:, 0], E1[:, 1], g)
    out = dict(cell_error_median=float(np.median(e)), cell_error_p99=float(np.percentile(e, 99)),
               margin_horizontal=float(angle(1.0, 0.5, 1 + MARGIN, 0.5, g)), margin_vertical=float(angle(0.5, 1.0, 0.5, 1 + MARGIN, g)))
    for f in FK:
        n, k = f["run_id"], f["target"] - 1
        x, y, oob, near = cell_samples(n, k); ex, ey = expected(ses[n], k % 5, k // 5)
        m0 = (x.mean(), y.mean()); mk = (x[~oob].mean(), y[~oob].mean())
        out["shift_boundary_%s_%d" % (n, k + 1)] = float(angle(*m0, *mk, g))
    return out


nom = quantities(GEOM); mc = []
for _ in range(300):
    mc.append(quantities({k: GEOM[k] + UG[k] * rng.standard_normal() for k in GEOM}))
MC = pd.DataFrame(mc)
U["geometry_monte_carlo"] = {k: dict(nominal=nom[k], u=float(MC[k].std()), u_rel_pct=float(100 * MC[k].std() / abs(nom[k])))
                             for k in ["cell_error_median", "cell_error_p99", "margin_horizontal", "margin_vertical"]}
U["geometry_monte_carlo_filter_shifts_u_rel_pct_max"] = float(max(100 * MC[c].std() / abs(nom[c]) for c in MC.columns
                                                                  if c.startswith("shift_boundary") and abs(nom[c]) > 0.1))

# ---------------------------------------------------------------- moving-block bootstrap of the filter shifts
def block_indices(nn, block=30):
    st = rng.integers(0, max(nn - block, 1), size=int(math.ceil(nn / block)))
    return np.concatenate([np.arange(s, min(s + block, nn)) for s in st])[:nn]


ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) > 50 else None
bt = {}
for f in FK:
    n, k = f["run_id"], f["target"] - 1
    x, y, oob, near = cell_samples(n, k); ex, ey = expected(ses[n], k % 5, k // 5)
    sb, db, si, di = [], [], [], []
    for _ in range(1000):
        i = block_indices(len(x)); xi, yi, oi, ni = x[i], y[i], oob[i], near[i]
        m0 = (xi.mean(), yi.mean())
        if (~oi).sum() >= MIN_MEAN:
            mk = (xi[~oi].mean(), yi[~oi].mean()); sb.append(float(angle(*m0, *mk))); db.append(float(angle(*mk, ex, ey) - angle(*m0, ex, ey)))
        if ni.sum() >= MIN_MEAN:
            mo = (xi[ni].mean(), yi[ni].mean()); si.append(float(angle(*m0, *mo))); di.append(float(angle(*mo, ex, ey) - angle(*m0, ex, ey)))
    have_b = f["err_mean_boundary"] == f["err_mean_boundary"]   # not NaN
    bt["%s_%d" % (n, k + 1)] = dict(shift_boundary=f["shift_boundary"], shift_boundary_ci=ci(sb),
                                    d_error_boundary=f["err_mean_boundary"] - f["err_mean"] if have_b else None, d_error_boundary_ci=ci(db),
                                    shift_isolated=f["shift_isolated"], shift_isolated_ci=ci(si),
                                    d_error_isolated=f["err_mean_isolated"] - f["err_mean"], d_error_isolated_ci=ci(di))
U["filter_block_bootstrap"] = bt


# ---------------------------------------------------------------- bootstrap of the rotation response
def pupil_cells(n):
    D = DAT[n]; P = []
    for k in range(N_TARGETS):
        s = D["LPV"] & window(D["t"], k)
        if s.sum() >= MIN_CELL:
            P.append((k % 5, k // 5, D["lpcx"][s].mean(), D["lpcy"][s].mean()))
    return np.array(P)


slopes = lambda P: (np.polyfit(P[:, 0], P[:, 2], 1)[0], np.polyfit(P[:, 1], P[:, 3], 1)[0])
resp = {}
for n in ["2_config_03_calib_OEMI_orig", "2_conf_03_calib_3mm", "2_config_03_calib_OEMI gliceryna", "powieka_opadajaca_1"]:
    Bs = [m for m in BASE if ses[m] == ses[n]]; PB = {m: pupil_cells(m) for m in Bs}; P = pupil_cells(n)
    R0 = np.median([slopes(PB[m]) for m in Bs], axis=0); b0 = np.array(slopes(P))
    vals = []
    for _ in range(2000):
        Pi = P[rng.integers(0, len(P), len(P))]
        if len(set(Pi[:, 0])) < 2 or len(set(Pi[:, 1])) < 2:
            continue
        Ri = np.median([slopes(PB[m][rng.integers(0, len(PB[m]), len(PB[m]))]) for m in rng.choice(Bs, len(Bs))], axis=0)
        vals.append(float(np.array(slopes(Pi)) @ Ri / (Ri @ Ri)))
    resp[n] = dict(response=float(b0 @ R0 / (R0 @ R0)), ci95=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                   n_cells=int(len(P)), n_reference_series=len(Bs))
U["rotation_response_bootstrap"] = resp

# ---------------------------------------------------------------- slopes of the first drooping-eyelid series
T975 = {21: 2.0796, 22: 2.0739, 23: 2.0687}   # Student t, 0.975 quantile, by degrees of freedom
n = "powieka_opadajaca_1"; P = pupil_cells(n); out = dict(n_cells=int(len(P)))
Bs = [m for m in BASE if ses[m] == ses[n]]
out["session_norm"] = [float(np.median([S.set_index("run_id").slope_x[m] for m in Bs])), float(np.median([S.set_index("run_id").slope_y[m] for m in Bs]))]
for name, xi, yi in (("horizontal", 0, 2), ("vertical", 1, 3)):
    x, y = P[:, xi], P[:, yi]; b1, b0 = np.polyfit(x, y, 1)
    se = math.sqrt(np.sum((y - (b0 + b1 * x)) ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    out[name] = dict(slope=float(b1), ci95=[float(b1 - T975[len(x) - 2] * se), float(b1 + T975[len(x) - 2] * se)])
U["drooping_1_slopes"] = out

# ---------------------------------------------------------------- angles referred to the centre of rotation
g12 = dict(GEOM, D=GEOM["D"] + 12.0)
red = 100 * (1 - angle(X1[:, 0], X1[:, 1], E1[:, 0], E1[:, 1], g12) / angle(X1[:, 0], X1[:, 1], E1[:, 0], E1[:, 1]))
gx, gy = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21)); gx, gy = gx.ravel(), gy.ravel()
i, j = np.triu_indices(len(gx), 1)
a0 = angle(gx[i], gy[i], gx[j], gy[j]); a12 = angle(gx[i], gy[i], gx[j], gy[j], g12)
red_disp = 100 * (1 - a12 / a0)
U["centre_of_rotation_12mm_reduction_pct"] = dict(baseline_cell_errors=[float(red.min()), float(red.max())],
                                                   point_pairs_within_display=[float(red_disp.min()), float(red_disp.max())])

json.dump(U, open(os.path.join(RESULTS, "uncertainty.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: v for k, v in U.items() if k != "filter_block_bootstrap"}, indent=1, ensure_ascii=False, default=float))
