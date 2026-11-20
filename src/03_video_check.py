# -*- coding: utf-8 -*-
"""Step 3. Control-panel video against the pupil-centre coordinate LPCX.

In the first drooping-eyelid series, the horizontal position of each model eye in the video frame (centre of the
largest bright object in each half of the wide camera view) is compared frame by frame with the LPCX values
recorded within +/- 0.05 s of the frame (video start = recording start):
  * measured eye (left half) vs LPCX < 0.5: do the frame and LPCX share one horizontal coordinate?
  * second model eye (right half) vs LPCX > 0.5: where is the second eye while the reported pupil centre jumps to it?
Output: results/video_check.json"""
import json
import os

import cv2
import numpy as np

from common import RESULTS, read_gaze, video_path

RUN = "powieka_opadajaca_1"
c = cv2.VideoCapture(video_path(RUN)); fps = c.get(cv2.CAP_PROP_FPS); pos = []; nr = -1
while True:
    ok, fr = c.read(); nr += 1
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(float); h, w = g.shape
    top = g[int(0.05 * h):int(0.58 * h), :]          # wide camera view, without the status bar and the magnified panel
    nlab, lab, st, cen = cv2.connectedComponentsWithStats((top > 60).astype(np.uint8))
    best = {}                                       # largest object in each half of the frame = model eye
    for i in range(1, nlab):
        side = 0 if cen[i][0] < w / 2 else 1
        if st[i, cv2.CC_STAT_AREA] > best.get(side, (0, 0))[0]:
            best[side] = (st[i, cv2.CC_STAT_AREA], cen[i][0] / w)
    if 0 in best and 1 in best and min(best[0][0], best[1][0]) > 30:
        pos.append((nr / fps, best[0][1], best[1][1]))
c.release()

D = read_gaze(RUN); m = D["LPV"]
measured, second = [], []
for tk, xm, xs in pos:
    s = m & (np.abs(D["t"] - tk) <= 0.05)
    if not s.any():
        continue
    lx = D["lpcx"][s]
    if (lx < 0.5).any():
        measured.append((xm, float(np.median(lx[lx < 0.5]))))
    if (lx > 0.5).any():
        second.append((xs, float(np.median(lx[lx > 0.5]))))
pm = np.array(measured); ps = np.array(second) if second else np.zeros((0, 2))
agree = [float(v) for v, l in zip(ps[:, 0], ps[:, 1]) if abs(v - l) < 0.02]   # second eye in the frame within 0.02 of LPCX
V = dict(fps=float(fps), frames=int(len(pos)),
         measured_eye=dict(frames=int(len(pm)), median_difference=float(np.median(pm[:, 0] - pm[:, 1])),
                           p95_abs_difference=float(np.percentile(np.abs(pm[:, 0] - pm[:, 1]), 95)),
                           correlation=float(np.corrcoef(pm[:, 0], pm[:, 1])[0, 1])),
         second_eye_during_jumps=dict(frames=int(len(ps)), frame_positions=[float(v) for v in ps[:, 0]],
                                      LPCX=[float(v) for v in ps[:, 1]], frames_within_0_02=len(agree),
                                      range_within_0_02=[min(agree), max(agree)] if agree else None))
json.dump(V, open(os.path.join(RESULTS, "video_check.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in V.items() if k != "second_eye_during_jumps"}, indent=1))
print({k: v for k, v in V["second_eye_during_jumps"].items() if k not in ("frame_positions", "LPCX")})
