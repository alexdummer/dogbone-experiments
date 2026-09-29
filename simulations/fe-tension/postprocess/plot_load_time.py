"""Force-time (load-time) curves from the FE relaxation simulations, reading
each material's gripForce.csv directly (written every increment by the
*fieldOutput export= mechanism in the generated .inp files -- see
generate_jobs.py)."""
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
    csv_path = HERE / material / "gripForce.csv"
    if not csv_path.exists():
        print(f"{material}: no gripForce.csv yet, skipping")
        continue
    data = np.loadtxt(csv_path)
    t, force = data[:, 0], data[:, 1]
    # drop the duplicate t=0 row EdelweissFE writes at initializeStep
    t, idx = np.unique(t, return_index=True)
    force = force[idx] * CROSS_SECTION_SYMMETRY_FACTOR
    n_rows = len(t)
    label = material if n_rows >= 41 else f"{material} (in progress, {n_rows} pts)"
    ls = "-" if n_rows >= 41 else "--"
    ax.plot(t, force, color=colors[i], linestyle=ls, marker="o", markersize=2.5,
            linewidth=1.2, label=label)

ax.set_xlabel("Time in s")
ax.set_ylabel("Full-specimen axial force in N")
ax.set_title("Force-time (dogbone-lattice FE relaxation simulations)")
ax.legend(fontsize=8)
fig.tight_layout()
out_path = HERE / "load_time.png"
fig.savefig(out_path, dpi=200)
print(f"saved {out_path}")
