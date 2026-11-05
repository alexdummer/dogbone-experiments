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
    ("", "QLV-Marmot", "-", "o"),
    ("_bb", "Bergström-Boyce", "--", "^"),
    ("_fractional_schapery", "Fractional Schapery", "-.", "s"),
]
DISPLAY_NAMES = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}


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


PAPER_FIGS = Path(__file__).resolve().parents[2] / "paper" / "figures"


def load_replicates(material):
    reps = []
    for r in (1, 2, 3):
        path = EXP_BASE / f"{material}-{r}_cleaned.csv"
        if path.exists():
            df = pd.read_csv(path)
            reps.append((df["time_s"].to_numpy(), df["position_mm"].to_numpy(), -df["force_N"].to_numpy()))
    return reps


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
        if PAPER_FIGS.exists():
            fig.savefig(PAPER_FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def grid_plot(xkey, xlabel, out_name):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    n = len(MATERIALS)
    center = n // 2
    fig, axes = plt.subplots(1, n, figsize=figsize_double, sharey=False)
    for i, (ax, material) in enumerate(zip(axes, MATERIALS)):
        disp_name = DISPLAY_NAMES.get(material, material)

        # Plot shaded experimental scatter band across replicates and mean curve
        reps = load_replicates(material)
        if reps:
            if xkey == "t":
                t_max = min(t_r.max() for t_r, _, _ in reps)
                x_grid = np.linspace(0, t_max, 1000)
                forces_interp = [np.interp(x_grid, t_r, force_r) for t_r, _, force_r in reps]
            else:
                disp_max = min(disp_r.max() for _, disp_r, _ in reps)
                x_grid = np.linspace(0, disp_max, 1000)
                forces_interp = [np.interp(x_grid, disp_r, force_r) for _, disp_r, force_r in reps]
            f_min = np.min(forces_interp, axis=0)
            f_max = np.max(forces_interp, axis=0)
            f_mean = np.mean(forces_interp, axis=0)
            ax.fill_between(x_grid, f_min, f_max, color="0.78", alpha=0.55, zorder=1)
            ax.plot(x_grid, f_mean, color="0.25", linestyle="-", linewidth=1.1, zorder=2)
        else:
            exp = load_experiment(material)
            if exp is not None:
                t, disp, force = exp
                x = t if xkey == "t" else disp
                ax.plot(x, force, color="0.25", linestyle="-", linewidth=1.1, zorder=2)

        # Plot FE models
        for suffix, model_label, ls, marker in MODELS:
            result = load_fe(material, suffix)
            if result is None:
                continue
            t, disp, force = result
            x = t if xkey == "t" else disp
            ax.plot(x, force, color=colors[i], linestyle=ls, linewidth=1.2,
                    marker=marker, markersize=2.2, markevery=2, zorder=3)

        ax.set_title(disp_name, fontsize=8.5)
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.set_ylabel("Axial force in N", fontsize=8.5)
        if i == center:
            ax.set_xlabel(xlabel, fontsize=8.5)

    legend_handles = [
        Patch(facecolor="0.78", edgecolor="none", alpha=0.55, label="Experiment (scatter)"),
        Line2D([], [], color="0.25", linestyle="-", linewidth=1.1, label="Experiment (mean)"),
        Line2D([], [], color="black", linestyle="-", marker="o", markersize=3, label="QLV-Marmot"),
        Line2D([], [], color="black", linestyle="--", marker="^", markersize=3, label="Bergström-Boyce"),
        Line2D([], [], color="black", linestyle="-.", marker="s", markersize=3, label="Fractional Schapery"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08),
               frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, out_name)



grid_plot("t", "Time in s", "lattice_experiment_comparison_force_time")
grid_plot("disp", "Grip displacement in mm", "lattice_experiment_comparison_force_disp")
