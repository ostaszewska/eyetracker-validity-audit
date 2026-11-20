# -*- coding: utf-8 -*-
"""Checks that every raw file listed in data/manifest.csv is present and has the listed SHA-256."""
import hashlib
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(REPO, "data"))
bad = 0
for r in pd.read_csv(os.path.join(REPO, "data", "manifest.csv")).itertuples():
    p = os.path.join(DATA, r.path)
    if not os.path.exists(p):
        print("missing:", r.path); bad += 1; continue
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != r.sha256:
        print("checksum mismatch:", r.path); bad += 1
print("OK" if not bad else "%d problem(s)" % bad)
sys.exit(1 if bad else 0)
