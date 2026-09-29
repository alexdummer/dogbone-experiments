"""Preliminary load-time curves from the A0V100 (pure Vero) 3-point-bending
FE predictions, one line per crack-length variant (a5/a10/a20mm), reading
each size's loadForce.csv directly (written every increment by the
*fieldOutput export= mechanism in edelweissfe_bending_A0V100/generate_bending_jobs_A0V100.py's
generated .inp files). Companion to dogbone-lattices/plot_lattice_fe_results.py
-- same "(in progress)"/dashed convention for a run not yet complete.

loadForce is the reaction force summed over the tied midspanCrossSection at
the load point (see generate_bending_jobs_A0V100.py's docstring for the full
rigid-body-constraint setup) -- i.e. the load a 3-point-bend test's load
cell would read, no symmetry-factor scaling needed (unlike the tension
dogbone-lattice jobs): this mesh is the full specimen cross-section, not a
quarter/octant.
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
    ax.plot(t, force, color=colors[i], linestyle=ls, linewidth=1.3, marker="o",
            markersize=2.5, markevery=2, label=label)

ax.set_xlabel("Time in s")
ax.set_ylabel("Load in N")
ax.set_title("A0V100 (pure Vero) cubic-lattice 3-point-bending")
ax.legend(fontsize=9)
fig.tight_layout()
out_path = HERE / "bending_load_time_A0V100.png"
fig.savefig(out_path, dpi=200)
print(f"saved {out_path}")
