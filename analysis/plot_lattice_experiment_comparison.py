"""QLV-Marmot vs. Bergström-Boyce FE predictions vs. the first physical
dogbone-lattice validation tests (2026-09-14), one panel per material.
Companion to plot_lattice_model_comparison.py -- same simulations, same
scaling conventions, same grid layout, with the measured force-time/
force-displacement curve added as a third series. Follows plotstyle.py
conventions.

Experimental data: dogbone-lattices/experiments/clean_data.py, one test per
material at that material's own "slow"-tier ramp/hold protocol -- the same
protocol the FE job decks target (see generate_jobs.py), so no rate
mismatch is expected between experiment and simulation.

One column per material (not sharey: peak forces span two orders of
magnitude across the five blends, same reasoning as plot_lattice_model_
comparison.py). Cleaned force_N is negative-in-tension (project convention);
flipped sign here to match the FE gripForce convention (positive in
tension, see edelweissfe/plot_load_time.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors, figsize_double  # noqa: E402

FE_BASE = REPO_ROOT / "simulations" / "fe-tension" / "jobs"
EXP_BASE = REPO_ROOT / "data" / "tensile-lattice" / "cleaned"
OUT_BASE = Path(__file__).parent
MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]

X_SYMMETRY_FACTOR = 2.0  # gripDisp -> full grip-to-grip elongation
CROSS_SECTION_SYMMETRY_FACTOR = 4.0  # gripForce -> full-specimen axial force

MODELS = [
    ("", "GM", "-", "o"),
    ("_bb", "BB", "--", "^"),
]


def load_fe(material, suffix):
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


def load_experiment(material):
    path = EXP_BASE / f"{material}_cleaned.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy()
    disp = df["position_mm"].to_numpy()
    force = -df["force_N"].to_numpy()
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
        exp = load_experiment(material)
        if exp is not None:
            t, disp, force = exp
            x = t if xkey == "t" else disp
            line, = ax.plot(x, force, color="0.5", linestyle="-", linewidth=1.0,
                             zorder=1)
            if i == 0:
                model_handles.append((line, "Experiment"))
        for suffix, model_label, ls, marker in MODELS:
            result = load_fe(material, suffix)
            if result is None:
                continue
            t, disp, force = result
            x = t if xkey == "t" else disp
            line, = ax.plot(x, force, color=colors[i], linestyle=ls, linewidth=1.3,
                             marker=marker, markersize=2.2, markevery=2, zorder=2)
            if i == 0:
                model_handles.append((line, model_label))
        ax.set_title(material, fontsize=9)
        if i == 0:
            ax.set_ylabel("Axial force in N")
        if i == center:
            ax.set_xlabel(xlabel)
    legend_entries = [("0.5", "-", None, "Experiment")] + [
        ("black", ls, marker, label) for _, label, ls, marker in MODELS
    ]
    handles = [plt.Line2D([], [], color=c, linestyle=ls, marker=marker, markersize=4)
               for c, ls, marker, _ in legend_entries]
    labels = [label for _, _, _, label in legend_entries]
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08),
               frameon=False)
    fig.tight_layout()
    save(fig, out_name)


grid_plot("t", "Time in s", "lattice_experiment_comparison_force_time")
grid_plot("disp", "Grip displacement in mm", "lattice_experiment_comparison_force_disp")
