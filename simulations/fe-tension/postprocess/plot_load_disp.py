"""Force-displacement (load-displacement) curves from the FE relaxation
simulations, reading each material's gripDisp.csv/gripForce.csv directly
(written every increment by the *fieldOutput export= mechanism in the
generated .inp files -- see generate_jobs.py). Both are indexed by the same
increments, so they're joined by row position (not by matching time values)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dogbone-relaxation"))
from plotstyle import colors  # noqa: E402

HERE = Path(__file__).parent
MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]

# The FE model only meshes the y>=0, z<=T/2 quarter of the cross-section (on
# top of the x=0 grip-to-grip symmetry, which needs no such factor: force is
# a "through" quantity, not split by that symmetry). gripForce as computed
# is therefore only 1/4 of the actual specimen's total reaction -- scale up
# to get the force a real testing machine/load cell would read.
CROSS_SECTION_SYMMETRY_FACTOR = 4.0

fig, ax = plt.subplots(figsize=(8, 5.5))

for i, material in enumerate(MATERIALS):
    disp_path = HERE / material / "gripDisp.csv"
    force_path = HERE / material / "gripForce.csv"
    if not (disp_path.exists() and force_path.exists()):
        print(f"{material}: no CSVs yet, skipping")
        continue
    disp_data = np.loadtxt(disp_path)
    force_data = np.loadtxt(force_path)
    _, idx = np.unique(disp_data[:, 0], return_index=True)
    # gripDisp is the octant model's own u_final/2 grip displacement; report
    # the physical (full-specimen) elongation, 2x that, for the load-
    # displacement curve.
    disp = 2.0 * disp_data[idx, 1]
    force = force_data[idx, 1] * CROSS_SECTION_SYMMETRY_FACTOR
    n_rows = len(disp)
    label = material if n_rows >= 41 else f"{material} (in progress, {n_rows} pts)"
    ls = "-" if n_rows >= 41 else "--"
    ax.plot(disp, force, color=colors[i], linestyle=ls, marker="o", markersize=2.5,
            linewidth=1.2, label=label)

ax.set_xlabel("Grip displacement in mm")
ax.set_ylabel("Full-specimen axial force in N")
ax.set_title("Force-displacement (dogbone-lattice FE relaxation simulations)")
ax.legend(fontsize=8)
fig.tight_layout()
out_path = HERE / "load_disp.png"
fig.savefig(out_path, dpi=200)
print(f"saved {out_path}")
