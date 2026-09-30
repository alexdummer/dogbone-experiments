"""Plot experimental repeatability across all 3 physical validation replicates
for the 5 tensile lattice dogbones (A0V100 through A100V0).

Generates:
- lattice_replicates_force_time.pdf/.png (Force vs. Time)
- lattice_replicates_force_disp.pdf/.png (Force vs. Grip Displacement)
- lattice_replicates_combined.pdf/.png (2-row grid: Time on top, Disp on bottom)

Follows plotstyle.py publication styling conventions.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors, figsize_double  # noqa: E402

EXP_BASE = REPO_ROOT / "data" / "tensile-lattice" / "cleaned"
OUT_BASE = Path(__file__).parent
PAPER_FIGS = Path(__file__).resolve().parents[2] / "paper" / "figures"
MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]

REP_STYLES = [
    ("Rep 1 (09-14)", "--", 0.9),
    ("Rep 2 (09-30)", "-.", 0.9),
    ("Rep 3 (09-30)", ":", 1.0),
]


def load_replicate(material, rep):
    path = EXP_BASE / f"{material}-{rep}_cleaned.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["time_s"].to_numpy(), df["position_mm"].to_numpy(), -df["force_N"].to_numpy()


def load_mean(material):
    path = EXP_BASE / f"{material}_cleaned.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["time_s"].to_numpy(), df["position_mm"].to_numpy(), -df["force_N"].to_numpy()


def save(fig, name):
    for ext in ("pdf", "png"):
        out_path = OUT_BASE / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if PAPER_FIGS.exists():
            fig.savefig(PAPER_FIGS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def grid_plot(xkey, xlabel, out_name):
    n = len(MATERIALS)
    center = n // 2
    fig, axes = plt.subplots(1, n, figsize=figsize_double, sharey=False)

    for i, (ax, material) in enumerate(zip(axes, MATERIALS)):
        reps = [load_replicate(material, r) for r in (1, 2, 3)]
        valid_reps = [r for r in reps if r is not None]

        # Individual replicates
        for r_idx, rep_data in enumerate(reps):
            if rep_data is None:
                continue
            t, disp, force = rep_data
            x = t if xkey == "t" else disp
            lbl, ls, lw = REP_STYLES[r_idx]
            ax.plot(x, force, color="0.45", linestyle=ls, linewidth=lw,
                    alpha=0.85, zorder=2)

        # Mean curve
        mean_data = load_mean(material)
        if mean_data is not None:
            t, disp, force = mean_data
            x = t if xkey == "t" else disp
            ax.plot(x, force, color="0.1", linestyle="-", linewidth=1.2, zorder=3)

        ax.set_title(material, fontsize=9)
        if i == 0:
            ax.set_ylabel("Axial force in N")
        if i == center:
            ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    legend_entries = [("0.1", "-", 1.2, "Mean (n=3)")] + [
        ("0.45", ls, lw, lbl) for lbl, ls, lw in REP_STYLES
    ]
    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=lw)
               for c, ls, lw, _ in legend_entries]
    labels = [label for _, _, _, label in legend_entries]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08),
               frameon=False)
    fig.tight_layout()
    save(fig, out_name)
    plt.close(fig)


def combined_grid_plot(out_name):
    n = len(MATERIALS)
    center = n // 2
    fig, axes = plt.subplots(2, n, figsize=(figsize_double[0], 1.8 * figsize_double[1]), sharey=False)

    for i, material in enumerate(MATERIALS):
        ax_t = axes[0, i]
        ax_d = axes[1, i]

        reps = [load_replicate(material, r) for r in (1, 2, 3)]

        # Time row (relaxation)
        for r_idx, rep_data in enumerate(reps):
            if rep_data is None:
                continue
            t, disp, force = rep_data
            lbl, ls, lw = REP_STYLES[r_idx]
            ax_t.plot(t, force, color="0.45", linestyle=ls, linewidth=lw, alpha=0.85, zorder=2)
            ax_d.plot(disp, force, color="0.45", linestyle=ls, linewidth=lw, alpha=0.85, zorder=2)

        # Mean curve
        mean_data = load_mean(material)
        if mean_data is not None:
            t, disp, force = mean_data
            ax_t.plot(t, force, color="0.1", linestyle="-", linewidth=1.2, zorder=3)
            ax_d.plot(disp, force, color="0.1", linestyle="-", linewidth=1.2, zorder=3)

        ax_t.set_title(material, fontsize=9)
        if i == 0:
            ax_t.set_ylabel("Axial force in N")
            ax_d.set_ylabel("Axial force in N")
        if i == center:
            ax_t.set_xlabel("Time in s")
            ax_d.set_xlabel("Grip displacement in mm")
        ax_t.grid(True, alpha=0.3)
        ax_d.grid(True, alpha=0.3)

    legend_entries = [("0.1", "-", 1.2, "Mean (n=3)")] + [
        ("0.45", ls, lw, lbl) for lbl, ls, lw in REP_STYLES
    ]
    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=lw)
               for c, ls, lw, _ in legend_entries]
    labels = [label for _, _, _, label in legend_entries]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04),
               frameon=False)
    fig.tight_layout()
    save(fig, out_name)
    plt.close(fig)


def main():
    grid_plot("t", "Time in s", "lattice_replicates_force_time")
    grid_plot("disp", "Grip displacement in mm", "lattice_replicates_force_disp")
    combined_grid_plot("lattice_replicates_combined")


if __name__ == "__main__":
    main()
