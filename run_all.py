# -*- coding: utf-8 -*-
"""Recomputes all results and figures from the raw data.  Usage:  python run_all.py"""
import os
import subprocess
import sys

STEPS = ["01_flags_and_checks.py", "02_uncertainty.py", "03_video_check.py", "04_figures.py"]
REPO = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
for s in STEPS:
    print("\n# " + s, flush=True)
    p = subprocess.run([sys.executable, os.path.join(REPO, "src", s)], cwd=os.path.join(REPO, "src"),
                       env=env, capture_output=True, text=True, encoding="utf-8")
    print(p.stdout, end="")
    with open(os.path.join(REPO, "results", "log_" + s.replace(".py", ".txt")), "w", encoding="utf-8") as f:
        f.write(p.stdout + p.stderr)
    if p.returncode:
        print(p.stderr)
        sys.exit("stopped at " + s)
print("\nDone. Results in results/, figures in figures/.")
