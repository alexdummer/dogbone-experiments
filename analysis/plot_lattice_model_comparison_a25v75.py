"""Direct GM (QLV-Marmot) vs. BB (Bergström-Boyce) FE predictions vs. experimental data
for the A25V75 dogbone-lattice tensile specimen, in a single-column width
(figsize_single wide) layout.

Follows plotstyle.py conventions:
  - Experiment: charcoal ("0.2"), solid
  - GM: colors[0] (blue), solid, circle markers
  - BB: colors[1] (orange), dashed, triangle markers
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors, figsize_single  # noqa: E402

FE_BASE = REPO_ROOT / "simulations" / "fe-tension" / "jobs"
EXP_BASE = REPO_ROOT / "data" / "tensile-lattice" / "cleaned"
OUT_BASE = Path(__file__).parent
PAPER_FIGS = Path(__file__).resolve().parents[2] / "paper" / "figures"
MATERIAL = "A25V75"

X_SYMMETRY_FACTOR = 2.0  # gripDisp -> full grip-to-grip elongation
CROSS_SECTION_SYMMETRY_FACTOR = 4.0  # gripForce -> full-specimen axial force

# (suffix, label, color, ls, marker)
MODELS = [
    ("", "GM", colors[0], "-", "o"),
    ("_bb", "BB", colors[1], "--", "^"),
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
    force = -df["force_N"].to_numpy()  # negative-in-tension -> positive
    return t, disp, force


def save(fig, name):
    for ext in ("pdf", "png"):
        out_path = OUT_BASE / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if PAPER_FIGS.exists():
            fig.savefig(PAPER_FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def plot_single(xkey, xlabel, out_name):
    fig, ax = plt.subplots(figsize=figsize_single)

    # Experimental curve
    exp = load_experiment(MATERIAL)
    if exp is not None:
        t, disp, force = exp
        x = t if xkey == "t" else disp
        ax.plot(x, force, color="0.2", linestyle="-", linewidth=1.4,
                label="Experiment", zorder=1)

    # FE model curves
    for suffix, model_label, color, ls, marker in MODELS:
        result = load_fe(MATERIAL, suffix)
        if result is None:
            continue
        t, disp, force = result
        x = t if xkey == "t" else disp
        ax.plot(x, force, color=color, linestyle=ls, linewidth=1.3,
                marker=marker, markersize=2.5, markevery=2, label=model_label, zorder=2)

    ax.set_title(MATERIAL, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Axial force in N")
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out_name)
    plt.close(fig)


def plot_combined():
    fig, (ax_time, ax_disp) = plt.subplots(
        2, 1, figsize=(figsize_single[0], 1.6 * figsize_single[1])
    )

    exp = load_experiment(MATERIAL)
    t_exp, disp_exp, force_exp = exp if exp is not None else (None, None, None)

    # Top panel: Force vs. Time
    if exp is not None:
        ax_time.plot(t_exp, force_exp, color="0.2", linestyle="-", linewidth=1.4,
                     label="Experiment", zorder=1)
    for suffix, model_label, color, ls, marker in MODELS:
        result = load_fe(MATERIAL, suffix)
        if result is not None:
            t, disp, force = result
            ax_time.plot(t, force, color=color, linestyle=ls, linewidth=1.3,
                         marker=marker, markersize=2.5, markevery=2, label=model_label, zorder=2)

    ax_time.set_title(MATERIAL, fontsize=9)
    ax_time.set_xlabel("Time in s")
    ax_time.set_ylabel("Axial force in N")
    ax_time.grid(True, alpha=0.4)
    ax_time.legend(frameon=False, fontsize=8)

    # Bottom panel: Force vs. Displacement
    if exp is not None:
        ax_disp.plot(disp_exp, force_exp, color="0.2", linestyle="-", linewidth=1.4,
                     label="Experiment", zorder=1)
    for suffix, model_label, color, ls, marker in MODELS:
        result = load_fe(MATERIAL, suffix)
        if result is not None:
            t, disp, force = result
            ax_disp.plot(disp, force, color=color, linestyle=ls, linewidth=1.3,
                         marker=marker, markersize=2.5, markevery=2, label=model_label, zorder=2)

    ax_disp.set_xlabel("Grip displacement in mm")
    ax_disp.set_ylabel("Axial force in N")
    ax_disp.grid(True, alpha=0.4)

    fig.tight_layout()
    save(fig, "lattice_model_comparison_A25V75_combined")
    plt.close(fig)


def main():
    plot_single("t", "Time in s", "lattice_model_comparison_A25V75_force_time")
    plot_single("disp", "Grip displacement in mm", "lattice_model_comparison_A25V75_force_disp")
    plot_combined()


if __name__ == "__main__":
    main()
