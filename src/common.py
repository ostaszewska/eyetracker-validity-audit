# -*- coding: utf-8 -*-
"""Paths, set-up geometry and helpers shared by all analysis steps.

The raw files are read only through data/manifest.csv. Their location is the data root:
data/ in this repository by default, or another folder given by the environment variable DATA_DIR. Paths in the manifest are relative to the data root (raw/gaze/..., raw/video/...)."""
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA_DIR", os.path.join(REPO, "data"))
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ---- set-up
W_MM, H_MM = 345.4, 194.3       # display, 15.6 inch
D_MM = 800.0                    # corneal apex of the model eye to the display
IPD_MM = 63.0                   # distance between the two model eyes; the measured (left) eye is IPD/2 left of centre
GEOM = dict(W=W_MM, H=H_MM, D=D_MM, dx=IPD_MM / 2, dy=0.0)

# ---- protocol and checks
T_SWEEP = 250.0                 # 25 targets x 10 s
N_TARGETS = 25                  # 5 x 5 grid in row order from the upper left; target k -> column k % 5, row k // 5
WIN_START, WIN_END = 5.0, 9.5   # analysis window within each 10 s target window [s]
MARGIN = 0.10                   # display expanded by 10 % on each side
BOUND_DEG = 5.0                 # isolated-deviation bound and impossible-motion bound [deg]
MIN_CELL = 10                   # samples needed for a cell (median, isolated deviations, pupil regression)
MIN_MEAN = 30                   # samples needed for a cell mean (before and after filtering) and for the expected-position fit

NEGATIVE_CONTROLS = {"2_config_03_calib_OEMI_orig": "original retina + water",
                     "2_config_03_calib_OEMI gliceryna": "original retina + glycerine",
                     "2_conf_03_calib_3mm": "3 mm pupil"}
BASELINE, DROOPING, CONTROL = "baseline", "drooping eyelid", "negative control"

MANIFEST = pd.read_csv(os.path.join(REPO, "data", "manifest.csv"))
_GAZE = MANIFEST[MANIFEST.kind == "gaze"].set_index("run_id")
_VIDEO = MANIFEST[MANIFEST.kind == "video"].set_index("run_id")


def runs():
    """The 36 analysed series, in manifest order."""
    return list(_GAZE.index)


def session(run_id):
    return _GAZE.session[run_id]


def group(run_id):
    if run_id in NEGATIVE_CONTROLS:
        return CONTROL
    if "opadajaca" in run_id or "pow_opad" in run_id:
        return DROOPING
    return BASELINE


def load(run_id):
    """A Gazepoint *_all_gaze.csv export as a DataFrame."""
    return pd.read_csv(os.path.join(DATA, _GAZE.path[run_id]))


def video_path(run_id):
    """Control-panel video (192 x 144 px, 10 fps) of a series of the session of 16 February 2024."""
    return os.path.join(DATA, _VIDEO.path[run_id])


def window(t, k):
    """Analysis window of target k (0-based): 5.0 < t <= 9.5 s after the start of its 10 s window."""
    return (t > k * 10 + WIN_START) & (t <= k * 10 + WIN_END)


def out_of_bounds(x, y, m=MARGIN):
    return (x < -m) | (x > 1 + m) | (y < -m) | (y > 1 + m)


def angle(x1, y1, x2, y2, g=GEOM):
    """Angle [deg] between the directions from the measured eye to two display points
    (coordinates as fractions of the display, BPOGX/BPOGY convention); vectorised."""
    eye = np.array([g["W"] / 2 - g["dx"], g["H"] / 2 + g["dy"], -g["D"]])
    x1, y1, x2, y2 = (np.asarray(v, float) for v in (x1, y1, x2, y2))
    a = np.stack([x1 * g["W"], y1 * g["H"], np.zeros_like(x1)], -1) - eye
    b = np.stack([x2 * g["W"], y2 * g["H"], np.zeros_like(x2)], -1) - eye
    c = np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fit_affine(rows):
    """Affine map (column, row) -> (x, y), least squares with three rounds of rejecting cells whose residual
    exceeds the median residual by more than 3 scaled MADs. rows: iterable of (column, row, x, y).
    Returns (coefficients for x, coefficients for y, number of cells, number retained)."""
    A = np.array(list(rows), float)
    X = np.column_stack([A[:, 0], A[:, 1], np.ones(len(A))])
    keep = np.ones(len(A), bool)
    for _ in range(3):
        bx = np.linalg.lstsq(X[keep], A[keep, 2], rcond=None)[0]
        by = np.linalg.lstsq(X[keep], A[keep, 3], rcond=None)[0]
        res = np.hypot(A[:, 2] - X @ bx, A[:, 3] - X @ by)
        med = np.median(res[keep]); mad = np.median(np.abs(res[keep] - med)) * 1.4826
        keep = res <= med + 3 * max(mad, 1e-9)
    return bx, by, len(A), int(keep.sum())


def apply_affine(coef, col, row):
    v = np.array([col, row, 1.0])
    return float(v @ coef[0]), float(v @ coef[1])


def read_gaze(run_id):
    """Columns used by the analysis as arrays; validity flags restricted to the 250 s sweep."""
    d = load(run_id)
    t = d.iloc[:, 3].to_numpy()          # TIME column
    sweep = t <= T_SWEEP
    return dict(run_id=run_id, session=session(run_id), group=group(run_id), t=t, sweep=sweep,
                cnt=d.CNT.to_numpy(), x=d.BPOGX.to_numpy(), y=d.BPOGY.to_numpy(),
                lpcx=d.LPCX.to_numpy(), lpcy=d.LPCY.to_numpy(), LPS=d.LPS.to_numpy(), LPD=d.LPD.to_numpy(),
                FPOGV=(d.FPOGV.to_numpy() != 0) & sweep, BPOGV=(d.BPOGV.to_numpy() != 0) & sweep,
                LPV=(d.LPV.to_numpy() == 1) & sweep, RPV=(d.RPV.to_numpy() != 0) & sweep)
