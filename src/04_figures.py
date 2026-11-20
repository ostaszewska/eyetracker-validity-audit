# -*- coding: utf-8 -*-
"""Step 4. Figures from the results of steps 1-2 (the eye-image figure also from the raw data and the video):
  rotation_response      rotation response vs reported pupil diameter;
  series_checks          series-level checks;
  eye_images             the tracker's eye imagery;
  out_of_bounds_cells    (left) cells with out-of-bounds samples, (right) effect of the two filters on cell means.
Outputs: figures/<name>.{pdf,svg,png}, results/eye_image_values.json"""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from common import RESULTS, FIGURES, BASELINE, DROOPING, CONTROL, BOUND_DEG, N_TARGETS, read_gaze, window, out_of_bounds, video_path

plt.rcParams.update({"svg.hashsalt": "figures", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.edgecolor": "#333333",
                     "axes.linewidth": 0.7, "figure.dpi": 300, "xtick.direction": "in", "ytick.direction": "in"})


def save(fig, name):
    kw = dict(bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "%s.pdf" % name), metadata={"CreationDate": None, "ModDate": None}, **kw)
    fig.savefig(os.path.join(FIGURES, "%s.svg" % name), metadata={"Date": None}, **kw)
    fig.savefig(os.path.join(FIGURES, "%s.png" % name), dpi=400, **kw)
    plt.close(fig)


S = pd.read_csv(os.path.join(RESULTS, "series.csv")); K = pd.read_csv(os.path.join(RESULTS, "cells.csv"))
L = json.load(open(os.path.join(RESULTS, "summary.json"), encoding="utf-8"))
U = json.load(open(os.path.join(RESULTS, "uncertainty.json"), encoding="utf-8"))
CLR = {BASELINE: "#0d6b70", DROOPING: "#8f6200", CONTROL: "#b5461b"}
MK = {BASELINE: "o", DROOPING: "s", CONTROL: "^"}
N = S.group.value_counts(); LB = {e: "%s (%d)" % (e, N.get(e, 0)) for e in CLR}

# ---------------- rotation response
fig, ax = plt.subplots(figsize=(3.5, 3.3))
for e, g in S.dropna(subset=["response", "LPD"]).groupby("group"):
    ax.scatter(g.LPD, g.response, s=34, c=CLR[e], marker=MK[e], edgecolors="k", linewidths=0.4, label=LB[e], zorder=3)
ax.axhline(1.0, color="#999999", lw=0.9, ls=(0, (4, 3))); ax.axhline(0.0, color="#333333", lw=0.7)
for r in S[S.group == CONTROL].itertuples():      # 95 % bootstrap confidence intervals of the controls
    ci = U["rotation_response_bootstrap"].get(r.run_id, {}).get("ci95")
    if ci:
        ax.plot([r.LPD, r.LPD], ci, color=CLR[CONTROL], lw=0.9, zorder=2)
ax.set_ylim(-1.6, 4.4)
ax.set_xlabel("reported pupil diameter [px]")
ax.set_ylabel("signed response of the pupil image\nto commanded rotation (1 = session norm)")
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handletextpad=0.3)
fig.tight_layout(); save(fig, "rotation_response")

# ---------------- series-level checks (zero values drawn at 0.008 % and 3e-4)
ZX, ZL = 0.008, 3e-4
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
for e, g in S.groupby("group"):
    ax[0].scatter(np.clip(g.pct_oob, ZX, None), g.pct_valid, s=30, c=CLR[e], marker=MK[e], edgecolors="k", linewidths=0.4, label=LB[e], zorder=3)
    ax[1].scatter(np.clip(g.LPS_departure, ZL, None), np.clip(g.pct_oob, ZX, None), s=30, c=CLR[e], marker=MK[e], edgecolors="k", linewidths=0.4, zorder=3)
bmax = L["separation_out_of_bounds"]["BPOGV_10"]["baseline_max_pct"]
ax[0].set_xscale("log"); ax[0].axvspan(bmax * 1.05, 150, color="#b5461b", alpha=0.07, zorder=1)
ax[0].set_xlim(0.006, 150); ax[0].set_ylim(-4, 104)
ax[0].set_xlabel("BPOGV-valid samples out of bounds [%]"); ax[0].set_ylabel("samples with BPOGV valid [%]")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("|LPS − 1|, departure of the pupil-scale value"); ax[1].set_ylabel("out of bounds [%]")
ax[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); save(fig, "series_checks")

# ---------------- eye images (control-panel video, session of 16 February, central target)
plt.rcParams.update({"axes.grid": False})
OK, BAD, INK = "#0d6b70", "#b5461b", "#333333"


def window_values(run_id):
    """LPV-valid share, out-of-bounds share of BPOGV-valid samples and median diameter over the analysis windows."""
    D = read_gaze(run_id); sel = np.zeros(len(D["t"]), bool)
    for k in range(N_TARGETS):
        sel |= window(D["t"], k)
    val = sel & D["BPOGV"]; pup = sel & D["LPV"]; oob = val & out_of_bounds(D["x"], D["y"])
    return dict(lpv=100 * pup.sum() / sel.sum(), oob=100 * oob.sum() / max(val.sum(), 1), lpd=float(np.median(D["LPD"][pup])))


def crop(run_id, k=12):
    c = cv2.VideoCapture(video_path(run_id)); fps = c.get(cv2.CAP_PROP_FPS)
    c.set(cv2.CAP_PROP_POS_FRAMES, int((k * 10 + 7.0) * fps)); ok, fr = c.read(); c.release()
    return cv2.cvtColor(fr[86:144, 0:96], cv2.COLOR_BGR2RGB) if ok else np.zeros((58, 96, 3), np.uint8)


CASES = [("bez_powieki", "no lid"), ("powieka_norm_1", "normal eyelid"), ("rzesy_dlugie_1", "long eyelashes"), ("powieka_opadajaca_1", "drooping eyelid")]
fig, ax = plt.subplots(1, 4, figsize=(5.2, 2.75)); F3 = {}
for a, (run, lab) in zip(ax, CASES):
    m = window_values(run); F3[lab] = m
    a.imshow(crop(run)); a.set_xticks([]); a.set_yticks([])
    a.set_title(lab, fontsize=7, pad=4)
    a.text(0.5, -0.13, "LPV valid %.1f %%" % m["lpv"], transform=a.transAxes, ha="center", va="top", fontsize=7, color=INK)
    a.text(0.5, -0.33, "out of bounds %.1f %%" % m["oob"], transform=a.transAxes, ha="center", va="top",
           fontsize=7, fontweight="bold" if m["oob"] > 5 else "normal", color=BAD if m["oob"] > 5 else OK)
    a.text(0.5, -0.54, "pupil %.1f px" % m["lpd"], transform=a.transAxes, ha="center", va="top",
           fontsize=7, fontweight="bold" if m["lpd"] < 18 else "normal", color=BAD if m["lpd"] < 18 else OK)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.30, wspace=0.06); save(fig, "eye_images")
plt.rcParams.update({"axes.grid": True})
json.dump(F3, open(os.path.join(RESULTS, "eye_image_values.json"), "w", encoding="utf-8"), indent=1)

# ---------------- out-of-bounds cells and filters
p99 = L["baseline_cell_error_deg"]["p99"]
PK = K[K.n_oob > 0]
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw=dict(width_ratios=[1.0, 1.2]))
for e, g in PK.groupby("group"):
    ax[0].scatter(np.clip(g.oob_dist_from_median, 0.05, None), np.clip(g.err_median, 0.1, None), s=8 + 2.2 * np.sqrt(g.n_oob),
                  c=CLR[e], marker=MK[e], edgecolors="k", linewidths=0.4, alpha=0.85, label=LB[e], zorder=3)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].axvline(BOUND_DEG, color="#999999", lw=0.9, ls=(0, (4, 3))); ax[0].axhline(p99, color="#999999", lw=0.9, ls=(0, (1, 2)))
ax[0].set_xlabel("out-of-bounds samples: median distance\nfrom the cell median [deg]"); ax[0].set_ylabel("cell median: distance from\nexpected position [deg]")
ax[0].set_xlim(0.1, 60); ax[0].set_ylim(0.1, 400)
for a_, tk in ((ax[0].xaxis, [0.1, 0.3, 1, 3, 10, 30]), (ax[0].yaxis, [0.1, 0.3, 1, 3, 10, 30, 100])):
    a_.set_major_locator(FixedLocator(tk)); a_.set_major_formatter(FixedFormatter([("%g" % v) for v in tk])); a_.set_minor_locator(NullLocator())
kw = dict(fontsize=7, transform=ax[0].transAxes)
ax[0].text(0.03, 0.02, "edge of a cluster", color="#0d6b70", ha="left", va="bottom", **kw)
ax[0].text(0.97, 0.02, "isolated\ndeviations", color="#0d6b70", ha="right", va="bottom", **kw)
ax[0].text(0.03, 0.97, "whole cell displaced", color="#b5461b", ha="left", va="top", **kw)
ax[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2, columnspacing=0.8, handletextpad=0.2)
NAMES = {"XL_pupil_powieka_norm_3": "7 mm, normal lid 3", "bez_powieki": "no lid", "rzesy_dlugie_1": "long lashes 1",
         "rzesy_dugie_2": "long lashes 2", "rzesy_dlugie_3": "long lashes 3"}
X = pd.DataFrame(L["filter_cells"]); X["label"] = ["%s, target %d" % (NAMES.get(a, a), c) for a, c in zip(X.run_id, X.target)]
X = X.sort_values("err_median"); yy = np.arange(len(X))
for y_, r in zip(yy, X.itertuples()):
    v = [r.err_mean, r.err_mean_boundary, r.err_mean_isolated, r.err_median]
    ax[1].plot([min(v), max(v)], [y_, y_], color="#bbbbbb", lw=0.8, zorder=1)
ax[1].scatter(X.err_median, yy, marker="|", s=110, c="k", linewidths=1.3, zorder=4, label="cell median")
ax[1].scatter(X.err_mean, yy, marker="o", s=26, facecolors="white", edgecolors="k", linewidths=0.7, zorder=3, label="mean, unfiltered")
ax[1].scatter(X.err_mean_boundary, yy + 0.18, marker="v", s=24, c="#b5461b", edgecolors="k", linewidths=0.4, zorder=3, label="mean, out-of-bounds samples removed")
ax[1].scatter(X.err_mean_isolated, yy - 0.18, marker="^", s=24, c="#0d6b70", edgecolors="k", linewidths=0.4, zorder=3, label="mean, isolated deviations (> 5°) removed")
ax[1].set_xscale("log"); ax[1].set_xlim(0.1, 10)
ax[1].xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10])); ax[1].xaxis.set_major_formatter(FixedFormatter(["0.1", "0.3", "1", "3", "10"])); ax[1].xaxis.set_minor_locator(NullLocator())
ax[1].set_yticks(yy); ax[1].set_yticklabels(X.label, fontsize=7)
ax[1].set_xlabel("distance from expected position [deg]")
ax[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=1, handletextpad=0.4)
fig.tight_layout(); save(fig, "out_of_bounds_cells")
print("eye-image values:", json.dumps(F3)); print("saved figures/")
