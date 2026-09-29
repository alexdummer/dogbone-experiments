"""Size-effect check for the A0V100 (pure Vero) 3-point-bending FE
predictions: load-time curves with the load scaled by crack length SQUARED
(F/a^2, N/mm^2), one line per crack-length variant (a5/a10/a20mm).
Companion to plot_bending_size_effect_A0V100.py (F/a instead) -- same data,
different normalization.

NOT the continuum-beam prediction -- see plot_bending_size_effect_A0V100.py's
docstring for the actual derivation: since only ONE cross-section dimension
(depth, in the bending direction) scales with a here, not both (the 5mm
unit-cell thickness stays fixed), an equivalent solid beam of this specimen's
shape predicts F ~ a, not F ~ a^2 (F ~ a^2 would be the prediction only for
a SQUARE cross-section scaling as a x a, which this specimen doesn't have).
F/a^2 is kept here purely as a deliberately-wrong comparison baseline -- it
diverges far more sharply across sizes than F/a does, which is exactly the
point: it demonstrates that F/a (not F/a^2) is the normalization actually
motivated by this specimen's real geometry.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors  # noqa: E402

HERE = Path(__file__).parent
JOB_BASE = HERE.parent / "jobs" / "edelweissfe_bending_A0V100"
SIZES = [5, 10, 20]
TOTAL_TIME = 600.0  # RAMP_TIME + HOLD_TIME, generate_bending_jobs_A0V100.py
# Row-count thresholds are NOT reliable here: EdelweissFE writes duplicate
# t=0 rows at each step boundary (2 steps -> 3 zero rows, not 1), so a
# completed run has 44 rows, not "2*20+1=41" -- checking the final TIME
# value instead of a row count avoids re-learning this the hard way.


def load_size(a):
    path = JOB_BASE / f"a{a}mm" / "loadForce.csv"
    if not path.exists():
        return None
    data = np.loadtxt(path)
    t, idx = np.unique(data[:, 0], return_index=True)
    force = data[idx, 1]
    complete = len(t) > 0 and t[-1] >= TOTAL_TIME - 1.0
    return t, force, complete


fig, ax = plt.subplots(figsize=(7, 5))
for i, a in enumerate(SIZES):
    result = load_size(a)
    if result is None:
        print(f"a{a}mm: no loadForce.csv yet, skipping")
        continue
    t, force, complete = result
    ls = "-" if complete else "--"
    label = f"a={a}mm" if complete else f"a={a}mm (in progress, t={t[-1]:.0f}/{TOTAL_TIME:.0f}s)"
    ax.plot(t, force / a**2, color=colors[i], linestyle=ls, linewidth=1.3, marker="o",
            markersize=2.5, markevery=2, label=label)

ax.set_xlabel("Time in s")
ax.set_ylabel(r"Load / a$^2$ in N/mm$^2$")
ax.set_title(r"A0V100 (pure Vero) cubic-lattice 3-point-bending -- size effect (F/a$^2$)")
ax.legend(fontsize=9)
fig.tight_layout()
out_path = HERE / "bending_size_effect_a2_A0V100.png"
fig.savefig(out_path, dpi=200)
print(f"saved {out_path}")
