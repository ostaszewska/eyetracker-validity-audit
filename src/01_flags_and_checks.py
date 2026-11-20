# -*- coding: utf-8 -*-
"""Step 1. Validity flags and the five checks for every series and every cell.

  * flag proportions (FPOGV, BPOGV, LPV) over the 250 s sweep;
  * out-of-bounds samples, impossible motion (jumps), isolated deviations, pupil-scale value;
  * response of the pupil image to the commanded rotation and jitter of the pupil centre;
  * expected positions (affine map per session), cell errors, cell means before and after the two filters;
  * sensitivity to the margin and to the flag (BPOGV main analysis, FPOGV sensitivity);
  * episodes in which the reported pupil centre lies on the second model eye;
  * timing of the pupil movement relative to the target windows.

Outputs: results/series.csv, results/cells.csv, results/challenge_series.csv, results/summary.json"""
import json
import os

import numpy as np
import pandas as pd

from common import (RESULTS, MARGIN, BOUND_DEG, MIN_CELL, MIN_MEAN, N_TARGETS, T_SWEEP, NEGATIVE_CONTROLS,
                    BASELINE, DROOPING, CONTROL, runs, read_gaze, window, out_of_bounds, angle, fit_affine, apply_affine)

DAT = {n: read_gaze(n) for n in runs()}
TARGETS = [(k, (k % 5, k // 5)) for k in range(N_TARGETS)]

# ---------------------------------------------------------------- expected positions: affine map per session
MAP = {}
for ses in sorted({D["session"] for D in DAT.values()}):
    rows = []
    for D in DAT.values():
        if D["session"] != ses or D["group"] != BASELINE:
            continue
        for k, (c, r) in TARGETS:
            s = D["BPOGV"] & window(D["t"], k)
            if s.sum() >= MIN_MEAN:
                rows.append((c, r, np.median(D["x"][s]), np.median(D["y"][s])))
    bx, by, n_cells, n_kept = fit_affine(rows)
    MAP[ses] = dict(coef=(bx, by), n_cells=n_cells, n_retained=n_kept)


def expected(ses, c, r):
    return apply_affine(MAP[ses]["coef"], c, r)


# ---------------------------------------------------------------- gaze checks for one validity flag
def gaze_checks(flag, margin=MARGIN):
    ser, cel = [], []
    for n, D in DAT.items():
        v = D[flag]; x, y = D["x"], D["y"]; n_sweep = int(D["sweep"].sum())
        oob = v & out_of_bounds(x, y, margin)
        jumps = pairs = 0; iso = {3.0: 0, 5.0: 0, 10.0: 0}; n_win = 0; prec = []
        for k, (c, r) in TARGETS:
            s = v & window(D["t"], k)
            idx = np.where(s)[0]
            if len(idx) >= 2:                       # impossible motion: adjacent samples only (counter + 1)
                adj = np.diff(D["cnt"][idx]) == 1
                dd = angle(x[idx[:-1]], y[idx[:-1]], x[idx[1:]], y[idx[1:]])
                jumps += int((dd[adj] > BOUND_DEG).sum()); pairs += int(adj.sum())
            if s.sum() < MIN_CELL:
                continue
            n_win += int(s.sum())
            mx, my = np.median(x[s]), np.median(y[s])
            dist = angle(x[s], y[s], mx, my)        # distance of each sample from the cell median
            for b in iso:
                iso[b] += int((dist > b).sum())
            ex, ey = expected(D["session"], c, r)
            o = oob[s]
            row = dict(run_id=n, group=D["group"], session=D["session"], target=k + 1, n=int(s.sum()), n_oob=int(o.sum()),
                       oob_dist_from_median=float(np.median(dist[o])) if o.any() else np.nan,
                       n_oob_near=int((dist[o] <= BOUND_DEG).sum()), n_oob_far=int((dist[o] > BOUND_DEG).sum()),
                       err_median=float(angle(mx, my, ex, ey)))
            if s.sum() >= MIN_MEAN:
                xs, ys = x[s], y[s]; keep_b = ~o; keep_i = dist <= BOUND_DEG
                m0 = (xs.mean(), ys.mean())
                row["err_mean"] = float(angle(*m0, ex, ey)); row["mean_to_median"] = float(angle(*m0, mx, my))
                if keep_b.sum() >= MIN_MEAN:        # boundary filter: out-of-bounds samples removed
                    m1 = (xs[keep_b].mean(), ys[keep_b].mean())
                    row["shift_boundary"] = float(angle(*m0, *m1)); row["err_mean_boundary"] = float(angle(*m1, ex, ey))
                    row["mean_boundary_to_median"] = float(angle(*m1, mx, my))
                if keep_i.sum() >= MIN_MEAN:        # cluster-based filter: isolated deviations removed
                    m2 = (xs[keep_i].mean(), ys[keep_i].mean())
                    row["shift_isolated"] = float(angle(*m0, *m2)); row["err_mean_isolated"] = float(angle(*m2, ex, ey))
                    row["mean_isolated_to_median"] = float(angle(*m2, mx, my))
                    rr = angle(xs[keep_i], ys[keep_i], *m2); prec.append(float(np.sqrt(np.mean(rr ** 2))))
            cel.append(row)
        ser.append(dict(run_id=n, group=D["group"], session=D["session"], n_sweep=n_sweep, n_valid=int(v.sum()),
                        pct_valid=100 * v.sum() / n_sweep, n_oob=int(oob.sum()), pct_oob=100 * oob.sum() / max(v.sum(), 1),
                        jumps=jumps, adjacent_pairs=pairs, pct_jumps=100 * jumps / max(pairs, 1),
                        **{"pct_isolated_%g" % b: 100 * iso[b] / max(n_win, 1) for b in iso}, n_in_windows=n_win,
                        precision=float(np.mean(prec)) if prec else np.nan))
    return pd.DataFrame(ser), pd.DataFrame(cel)


S, K = gaze_checks("BPOGV")      # main analysis
SF, KF = gaze_checks("FPOGV")    # sensitivity analysis


# ---------------------------------------------------------------- pupil checks (LPV-valid samples, camera-image coordinates)
def pupil_slopes(D):
    """Regression of the cell means of the pupil centre on the column / row index, and the jitter."""
    P, jit = [], []
    for k, (c, r) in TARGETS:
        s = D["LPV"] & window(D["t"], k)
        if s.sum() < MIN_CELL:
            continue
        i = np.where(s)[0]; adj = np.diff(D["cnt"][i]) == 1
        if adj.any():
            jit.append(np.median(np.hypot(np.diff(D["lpcx"][i]), np.diff(D["lpcy"][i]))[adj]))
        P.append((c, r, D["lpcx"][s].mean(), D["lpcy"][s].mean()))
    P = np.array(P); sx = sy = np.nan
    if len(P) >= 10:
        sx = np.polyfit(P[:, 0], P[:, 2], 1)[0]; sy = np.polyfit(P[:, 1], P[:, 3], 1)[0]
    return sx, sy, float(np.median(jit)) if jit else np.nan


for n, D in DAT.items():
    i = S.run_id == n; lpv = D["LPV"]; sx, sy, jit = pupil_slopes(D)
    S.loc[i, "slope_x"] = sx; S.loc[i, "slope_y"] = sy; S.loc[i, "jitter"] = jit
    S.loc[i, "n_FPOGV"] = int(D["FPOGV"].sum()); S.loc[i, "pct_FPOGV"] = 100 * D["FPOGV"].sum() / D["sweep"].sum()
    S.loc[i, "n_LPV"] = int(lpv.sum()); S.loc[i, "pct_LPV"] = 100 * lpv.sum() / D["sweep"].sum()
    S.loc[i, "LPS"] = float(np.median(D["LPS"][lpv])) if lpv.any() else np.nan
    S.loc[i, "LPD"] = float(np.median(D["LPD"][lpv])) if lpv.any() else np.nan
    S.loc[i, "n_RPV"] = int(D["RPV"].sum())
    S.loc[i, "FB_11"] = int((D["FPOGV"] & D["BPOGV"]).sum()); S.loc[i, "FB_10"] = int((D["FPOGV"] & ~D["BPOGV"]).sum())
    S.loc[i, "FB_01"] = int((~D["FPOGV"] & D["BPOGV"] & D["sweep"]).sum())
    S.loc[i, "FB_00"] = int((~D["FPOGV"] & ~D["BPOGV"] & D["sweep"]).sum())
# response = (b . r) / (r . r), r = median slope vector of the baseline series of the session; jitter relative to the same
B = S[S.group == BASELINE]
NX, NY, NJ = (B.groupby("session")[c].median() for c in ("slope_x", "slope_y", "jitter"))
S["response"] = [(r.slope_x * NX[r.session] + r.slope_y * NY[r.session]) / (NX[r.session] ** 2 + NY[r.session] ** 2) for r in S.itertuples()]
S["jitter_rel"] = [r.jitter / NJ[r.session] for r in S.itertuples()]
S["LPS_departure"] = (S.LPS - 1).abs()
S["pct_isolated_FPOGV"] = SF.set_index("run_id").pct_isolated_5[S.run_id].to_numpy()
S["pct_oob_FPOGV"] = SF.set_index("run_id").pct_oob[S.run_id].to_numpy()
S.to_csv(os.path.join(RESULTS, "series.csv"), index=False, float_format="%.6g", encoding="utf-8")
K.to_csv(os.path.join(RESULTS, "cells.csv"), index=False, float_format="%.6g", encoding="utf-8")

# ---------------------------------------------------------------- challenge series and extremes over the baseline series
CH_ORDER = ["2_config_03_calib_OEMI gliceryna", "2_conf_03_calib_3mm", "2_config_03_calib_OEMI_orig",
            "powieka_opadajaca_1", "powieka_opadajaca_2", "2_conf_02_calib_pow_opad_01"]
TCOLS = ["pct_FPOGV", "pct_valid", "pct_LPV", "pct_oob", "n_oob", "pct_jumps", "LPS", "LPD", "response"]
TCH = S.set_index("run_id").loc[CH_ORDER, ["group"] + TCOLS].reset_index()
b = S[S.group == BASELINE]
TCH = pd.concat([TCH, pd.DataFrame([dict(run_id="baseline min", group=BASELINE, **b[TCOLS].min()),
                                  dict(run_id="baseline max", group=BASELINE, **b[TCOLS].max())])], ignore_index=True)
TCH = TCH.rename(columns={"pct_FPOGV": "FPOGV_pct", "pct_valid": "BPOGV_pct", "pct_LPV": "LPV_pct", "pct_oob": "out_of_bounds_pct",
                        "n_oob": "out_of_bounds_n", "pct_jumps": "jumps_pct", "LPD": "diameter_px", "response": "rotation_response"})
TCH.to_csv(os.path.join(RESULTS, "challenge_series.csv"), index=False, float_format="%.4f", encoding="utf-8")


# ---------------------------------------------------------------- campaign-level quantities
L = {}
ch = S[S.group != BASELINE]; kb = K[K.group == BASELINE]; kd = K[K.group == DROOPING]; kc = K[K.group == CONTROL]
L["campaign"] = dict(n_series=int(len(S)), groups=S.group.value_counts().to_dict(),
                     samples_per_sweep=sorted(set(S.n_sweep.astype(int).tolist())),
                     samples_per_second=float(np.median(S.n_sweep) / T_SWEEP), RPV_valid_total=int(S.n_RPV.sum()),
                     FPOGV_valid_BPOGV_invalid=dict(total=int(S.FB_10.sum()), baseline=int(b.FB_10.sum())))
L["expected_positions"] = {s: dict(n_cells=M["n_cells"], n_retained=M["n_retained"]) for s, M in MAP.items()}
L["expected_positions_total"] = dict(n_cells=sum(M["n_cells"] for M in MAP.values()), n_retained=sum(M["n_retained"] for M in MAP.values()))
L["margin_deg"] = dict(horizontal=float(angle(1.0, 0.5, 1 + MARGIN, 0.5)), vertical=float(angle(0.5, 1.0, 0.5, 1 + MARGIN)))
dist = []
for ses in MAP:   # expanded boundary beyond the outermost expected positions (middle of each edge)
    for c, r in [(0, 2), (4, 2), (2, 0), (2, 4)]:
        ex, ey = expected(ses, c, r)
        gx = -MARGIN if c == 0 else (1 + MARGIN if c == 4 else ex); gy = -MARGIN if r == 0 else (1 + MARGIN if r == 4 else ey)
        dist.append(float(angle(ex, ey, gx, gy)))
L["boundary_beyond_outer_targets_deg"] = [min(dist), max(dist)]
L["baseline_cell_error_deg"] = dict(n=int(len(kb)), median=float(kb.err_median.median()), p95=float(kb.err_median.quantile(.95)),
                                    p99=float(kb.err_median.quantile(.99)), max=float(kb.err_median.max()),
                                    cells_without_mean=int(kb.err_mean.isna().sum()))
p99 = L["baseline_cell_error_deg"]["p99"]

# negative controls and drooping eyelid
L["controls"] = {n: dict(n_cells=int((kc.run_id == n).sum()),
                         cell_error_deg=[float(kc[kc.run_id == n].err_median.min()), float(kc[kc.run_id == n].err_median.median()),
                                         float(kc[kc.run_id == n].err_median.max())] if (kc.run_id == n).any() else None,
                         jitter_rel=float(S.set_index("run_id").jitter_rel[n])) for n in NEGATIVE_CONTROLS}
L["baseline_jitter_rel_range"] = [float(b.jitter_rel.min()), float(b.jitter_rel.max())]
m7 = S[S.run_id.str.contains("PowNorm7mm")]   # 7 mm pupil, normal eyelid, same day as the controls
L["pupil_7mm_same_day"] = dict(series=m7.run_id.tolist(), diameter_px=float(m7.LPD.mean()), expected_3mm_px=float(m7.LPD.mean() * 3 / 7),
                               pct_FPOGV=float(m7.pct_FPOGV.mean()))
L["drooping"] = {n: dict(cell_error_median_deg=float(kd[kd.run_id == n].err_median.median())) for n in kd.run_id.unique()}
L["drooping_cells"] = dict(n=int(len(kd)), max_error_deg=float(kd.err_median.max()), n_above_baseline_p99=int((kd.err_median > p99).sum()))
# reported pupil centre on the second model eye (right half of the camera image) while LPV = 1
second = {}
for n, D in DAT.items():
    m = D["LPV"] & (D["lpcx"] > 0.5)
    if m.any():
        t = D["t"][m]
        second[n] = dict(n=int(m.sum()), t_first=float(t.min()), t_last=float(t.max()),
                         BPOGV_valid=int((m & D["BPOGV"]).sum()), FPOGV_valid=int((m & D["FPOGV"]).sum()),
                         LPD_median=float(np.median(D["LPD"][m])), LPD_median_rest=float(np.median(D["LPD"][D["LPV"] & ~m])),
                         LPCX_median=float(np.median(D["lpcx"][m])), LPCX_median_rest=float(np.median(D["lpcx"][D["LPV"] & ~m])),
                         targets=sorted({int(k) + 1 for k in (t // 10)}))
L["pupil_on_second_eye"] = second

# out-of-bounds samples and isolated deviations in the baseline series
ko = kb[kb.n_oob > 0]
L["baseline_out_of_bounds"] = dict(max_pct=float(b.pct_oob.max()), n=int(b.n_oob.sum()), in_windows=int(kb.n_oob.sum()),
                                   between_targets=int(b.n_oob.sum() - kb.n_oob.sum()), cells=int(len(ko)),
                                   near=int(kb.n_oob_near.sum()), cells_with_near=int((ko.n_oob_near > 0).sum()),
                                   far=int(kb.n_oob_far.sum()), cells_with_far=int((ko.n_oob_far > 0).sum()),
                                   cells_list=ko[["run_id", "target", "n", "n_oob", "n_oob_near", "n_oob_far", "oob_dist_from_median",
                                                  "err_median"]].to_dict("records"))
phase = []   # out-of-bounds samples outside the analysis windows: time since the start of their 10 s target window
for n in b.run_id:
    D = DAT[n]; p = D["t"][D["BPOGV"] & out_of_bounds(D["x"], D["y"])] % 10
    phase += list(p[~((p > 5.0) & (p <= 9.5))])
phase = np.array(phase)
L["baseline_out_of_bounds"]["outside_windows_by_time"] = {"0-3.5 s": int((phase <= 3.5).sum()), "3.5-5 s": int(((phase > 3.5) & (phase <= 5)).sum()),
                                                          "9.5-10 s": int((phase > 9.5).sum())}
L["baseline_isolated_pct"] ={r.run_id: r.pct_isolated_5 for r in b.itertuples() if r.pct_isolated_5 > 0}
L["baseline_isolated_max_pct_by_bound"] = {"%g" % t: float(b["pct_isolated_%g" % t].max()) for t in (3.0, 5.0, 10.0)}
L["challenge_isolated_pct"] = {r.run_id: r.pct_isolated_5 for r in ch.itertuples()}

# effect of the two filters on the cell means of the baseline series
f = kb.dropna(subset=["err_mean"]).copy()
moved = f[(f.shift_boundary > 0.1) | (f.shift_isolated > 0.1)]
L["filter_cells"] = moved[["run_id", "target", "n", "n_oob", "shift_boundary", "shift_isolated", "err_median", "err_mean",
                           "err_mean_boundary", "err_mean_isolated", "mean_to_median", "mean_boundary_to_median",
                           "mean_isolated_to_median"]].to_dict("records")
ob = f[(f.shift_boundary > 0.1) & (f.shift_isolated <= 0.1)]; oi = f[f.shift_isolated > 0.1]
L["boundary_filter_only"] = dict(n=int(len(ob)), shift_deg=[float(ob.shift_boundary.min()), float(ob.shift_boundary.max())],
                                 error_reduction_deg=[float((ob.err_mean - ob.err_mean_boundary).min()), float((ob.err_mean - ob.err_mean_boundary).max())],
                                 all_closer=bool((ob.err_mean_boundary < ob.err_mean).all()),
                                 below_median_deg=[float((ob.err_median - ob.err_mean_boundary).min()), float((ob.err_median - ob.err_mean_boundary).max())])
L["isolated_filter"] = dict(n=int(len(oi)), shift_deg=[float(oi.shift_isolated.min()), float(oi.shift_isolated.max())],
                            mean_to_median_before_max=float(oi.mean_to_median.max()), mean_to_median_after_max=float(oi.mean_isolated_to_median.max()),
                            error_before_deg=[float(oi.err_mean.min()), float(oi.err_mean.max())],
                            error_after_deg=[float(oi.err_mean_isolated.min()), float(oi.err_mean_isolated.max())],
                            n_closer=int((oi.err_mean_isolated < oi.err_mean).sum()), n_further=int((oi.err_mean_isolated > oi.err_mean).sum()))

# separation of the series by the out-of-bounds fraction: flag x margin
sep = {}
for flag in ("BPOGV", "FPOGV"):
    for m in (0.05, 0.10, 0.20):
        SS = S if (flag, m) == ("BPOGV", MARGIN) else (SF if (flag, m) == ("FPOGV", MARGIN) else gaze_checks(flag, m)[0])
        bb = SS[SS.group == BASELINE]; cc = SS[SS.group != BASELINE]
        sep["%s_%d" % (flag, round(100 * m))] = dict(baseline_max_pct=float(bb.pct_oob.max()),
                                                     detected=int((cc.pct_oob > bb.pct_oob.max()).sum()),
                                                     challenge_pct=dict(zip(cc.run_id, cc.pct_oob.astype(float))))
L["separation_out_of_bounds"] = sep
L["FPOGV_sensitivity"] = dict(baseline_oob=int(SF[SF.group == BASELINE].n_oob.sum()),
                              baseline_oob_in_windows=int(KF[KF.group == BASELINE].n_oob.sum()),
                              baseline_oob_near=int(KF[KF.group == BASELINE].n_oob_near.sum()),
                              baseline_series_with_isolated=int((SF[SF.group == BASELINE].pct_isolated_5 > 0).sum()),
                              challenge=dict(zip(SF[SF.group != BASELINE].run_id, SF[SF.group != BASELINE][["n_valid", "n_oob", "pct_oob"]].to_dict("records"))))

# timing of the pupil movement relative to the target windows: first time after the window boundary at which the
# reported pupil centre has covered half of its displacement to the next target (baseline series, targets 2-25)
lag = []
for n in b.run_id:
    D = DAT[n]; t, lx, ly = D["t"], D["lpcx"], D["lpcy"]
    for k in range(1, N_TARGETS):
        s = D["LPV"] & (t > k * 10 - 4) & (t <= k * 10 + 4)
        if s.sum() < 50:
            continue
        pre = s & (t <= k * 10 - 1)
        x0 = np.median(lx[pre]) if pre.any() else lx[s][0]; y0 = np.median(ly[pre]) if pre.any() else ly[s][0]
        z = np.hypot(lx[s] - x0, ly[s] - y0); ts = t[s]
        zk = np.median(z[ts > k * 10 + 3])
        if zk < 0.005:
            continue
        i = np.where(z > zk / 2)[0]
        if len(i):
            lag.append(float(ts[i[0]] - k * 10))
lag = np.array(lag)
L["pupil_half_displacement_after_boundary_s"] = dict(n=int(len(lag)), min=float(lag.min()), p5=float(np.percentile(lag, 5)),
                                                     median=float(np.median(lag)), p95=float(np.percentile(lag, 95)), max=float(lag.max()))

json.dump(L, open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)

pd.set_option("display.width", 250)
print("Series:", L["campaign"]["groups"], "| samples per sweep:", L["campaign"]["samples_per_sweep"])
print("\nChallenge series and baseline extremes\n" + TCH.round(3).to_string(index=False))
print("\nBaseline cell error [deg]:", {k: round(v, 3) for k, v in L["baseline_cell_error_deg"].items()})
print("Out-of-bounds separation:", {k: (round(v["baseline_max_pct"], 2), v["detected"]) for k, v in sep.items()})
print("Baseline out-of-bounds:", {k: v for k, v in L["baseline_out_of_bounds"].items() if k != "cells_list"})
print("Boundary filter only:", L["boundary_filter_only"]); print("Isolated-deviation filter:", L["isolated_filter"])
