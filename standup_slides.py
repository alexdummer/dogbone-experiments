"""Four standalone, single-panel plots summarizing the key findings of the
digital-composite relaxation study, for the standup meeting.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plotstyle import colors, figsize_single  # noqa: E402

BASE = Path(__file__).parent
AREA_MM2 = 6.0 * 2.0

MATERIALS = {
    "A0V100": {"dir": BASE / "A0V100-relax" / "cleaned", "slow": [1, 2, 3]},
    "A25V75": {"dir": BASE / "A25V75-relax" / "cleaned", "slow": [1, 2, 3]},
    "A50V50": {"dir": BASE / "A50V50-relax-retest" / "cleaned", "slow": [4, 5, 6]},
}


def find_ramp_bounds(position):
    baseline = position.iloc[0]
    deviation = (position - baseline).abs()
    sustained = deviation.rolling(5).min() > 0.0002
    onset = sustained.to_numpy().nonzero()[0][0] - 4

    n = len(position)
    hold_value = position.iloc[-max(1, n // 10):].median()
    reached = (position >= 0.99 * hold_value).to_numpy().nonzero()[0]
    ramp_end = reached[reached > onset][0]
    return onset, ramp_end


def plot_modulus_vs_composition():
    df = pd.read_csv(BASE / "material_comparison_summary.csv")
    fig, ax = plt.subplots(figsize=figsize_single)
    for rate_group, marker, ls in [("slow", "o", "-"), ("fast", "s", "--")]:
        s = df[df["rate_group"] == rate_group].sort_values("a_fraction")
        ax.errorbar(s["a_fraction"], s["secant_modulus_mean"], yerr=s["secant_modulus_std"],
                    marker=marker, linestyle=ls, color=colors[0], capsize=3,
                    label=f"{rate_group} rate")
    ax.set_xlabel("A fraction (\\%)")
    ax.set_ylabel("Secant modulus (MPa)")
    ax.set_title("Composition sets stiffness:\n$\\sim$5x range across the series", fontsize=8)
    ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "slide1_modulus_vs_composition")


def plot_rate_sensitivity_vs_composition():
    df = pd.read_csv(BASE / "material_rate_sensitivity.csv")
    fig, ax = plt.subplots(figsize=figsize_single)
    ax.plot(df["a_fraction"], df["rate_sensitivity_m"], marker="o", color=colors[2])
    ax.set_xlabel("A fraction (\\%)")
    ax.set_ylabel(r"Rate sensitivity $m$")
    ax.set_title("More flexible resin =\nmore rate-sensitive", fontsize=8)
    fig.tight_layout()
    _save(fig, "slide2_rate_sensitivity_vs_composition")


def plot_relaxation_vs_composition():
    df = pd.read_csv(BASE / "material_comparison_summary.csv")
    fig, ax = plt.subplots(figsize=figsize_single)
    for rate_group, marker, ls in [("slow", "o", "-"), ("fast", "s", "--")]:
        s = df[df["rate_group"] == rate_group].sort_values("a_fraction")
        ax.errorbar(s["a_fraction"], s["pct_relaxation_mean"], yerr=s["pct_relaxation_std"],
                    marker=marker, linestyle=ls, color=colors[1], capsize=3,
                    label=f"{rate_group} rate")
    ax.set_xlabel("A fraction (\\%)")
    ax.set_ylabel("Stress relaxed by 590s (\\%)")
    ax.set_title("Composition controls how\nmuch stress relaxes away", fontsize=8)
    ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "slide3_relaxation_vs_composition")


def plot_relaxation_shape_differences():
    fig, ax = plt.subplots(figsize=figsize_single)
    for i, (material, cfg) in enumerate(MATERIALS.items()):
        color = colors[i + 3]
        for j, test_id in enumerate(cfg["slow"]):
            df = pd.read_csv(cfg["dir"] / f"{material}-{test_id}_cleaned.csv")
            stress = -df["force_N"] / AREA_MM2
            onset, ramp_end = find_ramp_bounds(df["position_mm"])
            peak_idx = stress.iloc[ramp_end:].to_numpy().argmax() + ramp_end
            peak_stress = stress.iloc[peak_idx]
            t_peak = df["time_s"].iloc[peak_idx]

            t = df["time_s"].to_numpy() - t_peak
            norm = stress.to_numpy() / peak_stress
            mask = t > 0
            ax.plot(t[mask], norm[mask], color=color, linewidth=0.8,
                    label=material if j == 0 else None)

    ax.set_xscale("log")
    ax.set_xlabel("Time since peak stress (s)")
    ax.set_ylabel("Normalized stress")
    ax.set_title("Composition changes the shape of\nrelaxation, not just its extent", fontsize=8)
    ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "slide4_relaxation_shape_differences")


def _save(fig, name):
    out_path = BASE / f"{name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    plot_modulus_vs_composition()
    plot_rate_sensitivity_vs_composition()
    plot_relaxation_vs_composition()
    plot_relaxation_shape_differences()
