"""Direct QLV-Marmot vs. Bergström-Boyce comparison, one panel per material,
for the paper's "Validation and predictive capabilities" subsection.
Companion to plot_lattice_fe_results.py / plot_lattice_fe_results_bb.py --
same simulations, same scaling conventions, but overlaid per material instead
of shown in two separate figures, to make the two models' relative agreement
directly visible without flipping between figures. Follows plotstyle.py
conventions.

One column per material (not sharey: peak forces span two orders of
magnitude across the five blends, same reasoning as the individual-model
figures). Both models are complete for all five materials, so linestyle is
free to encode model (solid=QLV-Marmot, dashed=Bergström-Boyce) without the
"dashed=incomplete" ambiguity used elsewhere in this project.
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

MODELS = [
    ("", "GM", "-", "o"),
    ("_bb", "BB", "--", "^"),
]


def load_material(material, suffix):
    disp_path = FE_BASE / f"{material}{suffix}" / "gripDisp.csv"
    force_path = FE_BASE / f"{material}{suffix}" / "gripForce.csv"
    if not (disp_path.exists() and force_path.exists()):
        return None
    disp_data = np.loadtxt(disp_path)
    force_data = np.loadtxt(force_path)
    t, idx = np.unique(disp_data[:, 0], return_index=True)
    disp = X_SYMMETRY_FACTOR * disp_data[idx, 1]
    force = CROSS_SECTION_SYMMETRY_FACTOR * force_data[idx, 1]
    return t, disp, force


def save(fig, name):
    for ext in ("pdf", "png"):
        out_path = OUT_BASE / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def grid_plot(xkey, xlabel, out_name):
    n = len(MATERIALS)
    center = n // 2
    fig, axes = plt.subplots(1, n, figsize=figsize_double, sharey=False)
    model_handles = []
    for i, (ax, material) in enumerate(zip(axes, MATERIALS)):
        for suffix, model_label, ls, marker in MODELS:
            result = load_material(material, suffix)
            if result is None:
                continue
            t, disp, force = result
            x = t if xkey == "t" else disp
            line, = ax.plot(x, force, color=colors[i], linestyle=ls, linewidth=1.3,
                             marker=marker, markersize=2.2, markevery=2)
            if i == 0:
                model_handles.append((line, model_label))
        ax.set_title(material, fontsize=9)
        if i == 0:
            ax.set_ylabel("Axial force in N")
        if i == center:
            ax.set_xlabel(xlabel)
    handles = [plt.Line2D([], [], color="black", linestyle=ls, marker=marker, markersize=4)
               for _, _, ls, marker in MODELS]
    labels = [label for _, label, _, _ in MODELS]
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08),
               frameon=False)
    fig.tight_layout()
    save(fig, out_name)


grid_plot("t", "Time in s", "lattice_model_comparison_force_time")
grid_plot("disp", "Grip displacement in mm", "lattice_model_comparison_force_disp")
