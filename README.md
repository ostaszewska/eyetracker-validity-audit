# Validity flags of a video eye-tracker: data, analysis code and figures

Raw recordings of a Gazepoint GP3 HD video eye-tracker aimed at a model eye moved by a two-axis robotic drive,
and the code that computes, from these recordings, the checks of the tracker's validity flags and the figures.

## Contents

| Path | Contents |
|---|---|
| `data/raw/gaze/<session>/<series>_all_gaze.csv` | 36 recording series exported by Gazepoint Analysis (5 sessions, February–April 2024) |
| `data/raw/video/*-face.avi` | control-panel video (192 × 144 px, 10 fps) of four series of the session of 16 February 2024 |
| `data/manifest.csv` | series and video files with their SHA-256 checksums; the scripts read the raw files only through it |
| `src/` | analysis steps (below) |
| `LICENSE`, `data/LICENSE` | licences of the code (MIT) and of the data and figures (CC BY 4.0) |
| `figures/` | the four figures, as PDF, SVG and PNG (400 dpi), as produced by `src/04_figures.py` |

Series identifiers are the original file names. The 30 baseline series carry different eyelid and eyelash overlays and
5 or 7 mm pupils; `powieka_opadajaca_*` and `2_conf_02_calib_pow_opad_01` have a drooping eyelid; the three negative
controls are `2_config_03_calib_OEMI_orig` (original retina element, water filling), `2_config_03_calib_OEMI gliceryna`
(original retina element, glycerine filling) and `2_conf_03_calib_3mm` (3 mm pupil insert).

## Running

Tested with Python 3.13.

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
python verify_data.py                                    # files present, checksums match
python run_all.py                                        # about 1 min
```

`run_all.py` writes `results/` and `figures/` and saves the log of each step as `results/log_*.txt`. The bootstrap and
Monte Carlo draws come from one seeded generator, so a rerun gives identical numbers and byte-identical figures.

## Set-up and protocol (`src/common.py`)

* display 345.4 × 194.3 mm; measured (left) model eye 800 mm from it, 31.5 mm left of the display centre, at its height;
  a second model eye, 63 mm to the right, is moved by the same drive;
* 25 targets on a 5 × 5 grid, 10 s each, in row order from the upper left; the analysis window of each target is
  5.0–9.5 s; flag proportions refer to the 250 s sweep;
* out of bounds: outside the display expanded by 10 % on each side; isolated deviation: more than 5° from the median of
  its cell; impossible motion: more than 5° between adjacent samples (counters differing by one);
* cells need at least 10 valid samples, cell means at least 30 (before and after filtering).

## Analysis steps

| Script | What it computes | Outputs (`results/`) |
|---|---|---|
| `01_flags_and_checks.py` | flag proportions (FPOGV, BPOGV, LPV); out-of-bounds samples, jumps, isolated deviations, pupil-scale value; response of the pupil image to the commanded rotation and jitter; expected positions (affine map per session) and cell errors; boundary and cluster-based filters; sensitivity to the margin and the flag; pupil centre reported on the second model eye; timing of the pupil movement | `series.csv`, `cells.csv`, `challenge_series.csv`, `summary.json` |
| `02_uncertainty.py` | expected positions fitted without the series concerned; spread of a target across series; Monte Carlo propagation of the geometry; angles referred to the centre of rotation; moving-block bootstrap of the filter shifts; bootstrap intervals of the rotation response; slopes of the first drooping-eyelid series | `uncertainty.json` |
| `03_video_check.py` | position of both model eyes in the video frames against the pupil-centre coordinate LPCX | `video_check.json` |
| `04_figures.py` | figures `rotation_response`, `series_checks`, `eye_images`, `out_of_bounds_cells` | `figures/`, `eye_image_values.json` |

`challenge_series.csv` lists the six challenge series and the extremes over the baseline series; `series.csv` gives,
for every series, the counts and denominators of all proportions.

## Licence

* Code (`src/`, `run_all.py`, `verify_data.py`): MIT License, see `LICENSE`.
* Data (`data/`) and figures (`figures/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `data/LICENSE`.
