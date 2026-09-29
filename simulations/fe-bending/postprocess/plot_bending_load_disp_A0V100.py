"""Load-deflection curves from the A0V100 (pure Vero) 3-point-bending FE
predictions, one line per crack-length variant (a5/a10/a20mm), reading each
size's loadDisp.csv/loadForce.csv directly. Companion to
plot_bending_load_time_A0V100.py (same data, time axis instead).

loadDisp/loadForce are indexed by the same increments (both written every
increment by the same *fieldOutput export= mechanism), so they're joined by
row position, not by matching time values -- same convention as
dogbone-lattices/plot_lattice_model_comparison.py.
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


def load_size(a):
    disp_path = JOB_BASE / f"a{a}mm" / "loadDisp.csv"
    force_path = JOB_BASE / f"a{a}mm" / "loadForce.csv"
    if not (disp_path.exists() and force_path.exists()):
        return None
    disp_data = np.loadtxt(disp_path)
    force_data = np.loadtxt(force_path)
    t, idx = np.unique(disp_data[:, 0], return_index=True)
    disp = disp_data[idx, 1]
    force = force_data[idx, 1]
    complete = len(t) > 0 and t[-1] >= TOTAL_TIME - 1.0
    return disp, force, t, complete


fig, ax = plt.subplots(figsize=(7, 5))
for i, a in enumerate(SIZES):
    result = load_size(a)
    if result is None:
        print(f"a{a}mm: no data yet, skipping")
        continue
    disp, force, t, complete = result
    ls = "-" if complete else "--"
    label = f"a={a}mm" if complete else f"a={a}mm (in progress, t={t[-1]:.0f}/{TOTAL_TIME:.0f}s)"
    ax.plot(disp, force, color=colors[i], linestyle=ls, linewidth=1.3, marker="o",
            markersize=2.5, markevery=2, label=label)

ax.set_xlabel("Midspan deflection in mm")
ax.set_ylabel("Load in N")
ax.set_title("A0V100 (pure Vero) cubic-lattice 3-point-bending -- load-deflection")
ax.legend(fontsize=9)
fig.tight_layout()
out_path = HERE / "bending_load_disp_A0V100.png"
fig.savefig(out_path, dpi=200)
print(f"saved {out_path}")
