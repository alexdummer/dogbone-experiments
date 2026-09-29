"""Force-time and force-displacement curves from the predictive FE
relaxation simulations of the lattice-infill dogbone specimens (see
edelweissfe/), for the paper's "Validation and
predictive capabilities" subsection. Follows this project's plotstyle.py
conventions (colorblind palette, LaTeX text, "Quantity in UNIT" axis labels).

One column per material (not sharey: peak forces span ~300N down to <1N
across the five blends, so a shared axis would flatten the softer materials
to invisibility -- the same reasoning as standup_slides.py's slide 5).

Reads each material's gripForce.csv/gripDisp.csv directly (written every
increment by the *fieldOutput export= mechanism in the FE job decks -- see
dogbone-lattices/edelweissfe/generate_jobs.py). The FE model exploits three
mirror symmetries (x=0 grip-to-grip, y=0 and z=T/2 cross-section quarters);
gripDisp is doubled (x=0 symmetry: force is a "through" quantity, so gripForce
needs no such factor from x=0, only from the cross-section quarter) and
gripForce is scaled by 4 (y=0, z=T/2 cross-section quarters) to recover the
full-specimen quantities a real testing machine/load cell would read.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors, figsize_double  # noqa: E402

FE_BASE = REPO_ROOT / "simulations" / "fe-tension" / "jobs"
OUT_BASE = Path(__file__).parent
MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]

X_SYMMETRY_FACTOR = 2.0  # gripDisp -> full grip-to-grip elongation
CROSS_SECTION_SYMMETRY_FACTOR = 4.0  # gripForce -> full-specimen axial force
N_EXPECTED = 41  # 2 steps x 20 increments + 1 initial zero increment


def load_material(material):
    disp_path = FE_BASE / material / "gripDisp.csv"
    force_path = FE_BASE / material / "gripForce.csv"
    if not (disp_path.exists() and force_path.exists()):
        return None
    disp_data = np.loadtxt(disp_path)
    force_data = np.loadtxt(force_path)
    t, idx = np.unique(disp_data[:, 0], return_index=True)
    disp = X_SYMMETRY_FACTOR * disp_data[idx, 1]
    force = CROSS_SECTION_SYMMETRY_FACTOR * force_data[idx, 1]
    complete = len(t) >= N_EXPECTED
    return t, disp, force, complete


def save(fig, name):
    for ext in ("pdf", "png"):
        out_path = OUT_BASE / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def grid_plot(xkey, xlabel, out_name):
    n = len(MATERIALS)
    center = n // 2
    fig, axes = plt.subplots(1, n, figsize=figsize_double, sharey=False)
    for i, (ax, material) in enumerate(zip(axes, MATERIALS)):
        result = load_material(material)
        if result is None:
            ax.set_title(f"{material}\n(not yet run)", fontsize=9)
            if i == 0:
                ax.set_ylabel("Axial force in N")
            if i == center:
                ax.set_xlabel(xlabel)
            continue
        t, disp, force, complete = result
        x = t if xkey == "t" else disp
        ls = "-" if complete else "--"
        ax.plot(x, force, color=colors[i], linestyle=ls, linewidth=1.3, marker="o",
                markersize=2.2, markevery=2)
        ax.set_title(material if complete else f"{material} (in progress)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Axial force in N")
        if i == center:
            ax.set_xlabel(xlabel)
    fig.tight_layout()
    save(fig, out_name)


grid_plot("t", "Time in s", "lattice_fe_force_time")
grid_plot("disp", "Grip displacement in mm", "lattice_fe_force_disp")
