"""Four standalone, single-panel plots summarizing the key findings of the
digital-composite relaxation study, for the standup meeting.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plotstyle import colors, figsize_double, figsize_single  # noqa: E402
# The authoritative, current material/rate-group set (fit_qlv_model.py's own
# MATERIALS, kept in sync with the rest of the project) -- this module's own
# MATERIALS dict below predates A75V25/A100V0 having real data and is now
# stale for slide 5, which is updated to use all 5 materials; slides 1-4
# still use the local (3-material) dict, unchanged.
from fit_qlv_model import MATERIALS as MATERIALS_ALL  # noqa: E402

BASE = Path(__file__).parent
AREA_MM2 = 6.0 * 2.0

MATERIALS = {
    "A0V100": {
        "dir": BASE / "A0V100-relax" / "cleaned",
        "rate_groups": {
            "very slow": [10, 11, 12],
            "slow": [1, 3, 13],  # 2 excluded, outlier
            "fast": [4, 5, 6],
        },
    },
    "A25V75": {
        "dir": BASE / "A25V75-relax" / "cleaned",
        "rate_groups": {
            "very slow": [10, 11, 12],
            "slow": [1, 2, 3, 9],
            "fast": [4, 5, 6],
        },
    },
    "A50V50": {
        "dir": BASE / "A50V50-relax-retest" / "cleaned",
        "rate_groups": {
            "very slow": [13, 14, 15],
            "slow": [4, 5, 6],  # 16 excluded, outlier
            "fast": [7, 11, 12],
        },
    },
}

RATE_GROUPS = [("very slow", "o", ":"), ("slow", "o", "-"), ("fast", "s", "--")]
INDIVIDUAL_ALPHA = 0.3


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
    for rate_group, marker, ls in RATE_GROUPS:
        s = df[df["rate_group"] == rate_group].sort_values("a_fraction")
        ax.errorbar(s["a_fraction"], s["secant_modulus_mean"], yerr=s["secant_modulus_std"],
                    marker=marker, markersize=4, linestyle=ls, color=colors[0], capsize=3,
                    label=f"{rate_group} rate")
    ax.set_xlabel("Agilus fraction (\\%)")
    ax.set_ylabel("Secant modulus (MPa)")
    ax.set_title("Composition sets stiffness:\n$\\sim$5x range across the series", fontsize=8)
    ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "slide1_modulus_vs_composition")


def plot_rate_sensitivity_vs_composition():
    df = pd.read_csv(BASE / "material_rate_sensitivity.csv")
    fig, ax = plt.subplots(figsize=figsize_single)
    ax.plot(df["a_fraction"], df["rate_sensitivity_m"], marker="o", color=colors[2])
    ax.set_xlabel("Agilus fraction (\\%)")
    ax.set_ylabel(r"Rate sensitivity $m$")
    ax.set_title("More flexible resin =\nmore rate-sensitive", fontsize=8)
    fig.tight_layout()
    _save(fig, "slide2_rate_sensitivity_vs_composition")


def plot_relaxation_vs_composition():
    df = pd.read_csv(BASE / "material_comparison_summary.csv")
    fig, ax = plt.subplots(figsize=figsize_single)
    for rate_group, marker, ls in RATE_GROUPS:
        s = df[df["rate_group"] == rate_group].sort_values("a_fraction")
        ax.errorbar(s["a_fraction"], s["pct_relaxation_mean"], yerr=s["pct_relaxation_std"],
                    marker=marker, markersize=4, linestyle=ls, color=colors[1], capsize=3,
                    label=f"{rate_group} rate")
    ax.set_xlabel("Agilus fraction (\\%)")
    ax.set_ylabel("Stress relaxed by 590s (\\%)")
    ax.set_title("Composition controls how\nmuch stress relaxes away", fontsize=8)
    ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "slide3_relaxation_vs_composition")


def normalized_curve(cfg, material, test_id):
    df = pd.read_csv(cfg["dir"] / f"{material}-{test_id}_cleaned.csv")
    stress = -df["force_N"] / AREA_MM2
    onset, ramp_end = find_ramp_bounds(df["position_mm"])
    peak_idx = stress.iloc[ramp_end:].to_numpy().argmax() + ramp_end
    peak_stress = stress.iloc[peak_idx]
    t_peak = df["time_s"].iloc[peak_idx]

    t = df["time_s"].to_numpy() - t_peak
    norm = stress.to_numpy() / peak_stress
    mask = t > 0
    return t[mask], norm[mask]


def mean_norm_curve(curve_list):
    """Mean of several (t, norm_stress) curves on a common log-time grid,
    NaN-masking each curve past its own duration so a shorter curve
    shrinks the mean's effective N near the tail instead of truncating
    the whole mean early."""
    log_min = min(np.log10(t.min()) for t, _ in curve_list)
    log_max = max(np.log10(t.max()) for t, _ in curve_list)
    grid = np.logspace(log_min, log_max, 300)
    values = []
    for t, norm in curve_list:
        v = np.interp(grid, t, norm)
        v[(grid < t.min() * (1 - 1e-9)) | (grid > t.max() * (1 + 1e-9))] = np.nan
        values.append(v)
    return grid, np.nanmean(values, axis=0)


def plot_relaxation_shape_differences():
    fig, axes = plt.subplots(1, 3, figsize=(figsize_double[0], figsize_double[1]), sharey=True)

    for ax, (rate_group, _, ls) in zip(axes, RATE_GROUPS):
        for i, (material, cfg) in enumerate(MATERIALS.items()):
            color = colors[i + 3]
            ids = cfg["rate_groups"][rate_group]
            curves = [normalized_curve(cfg, material, test_id) for test_id in ids]
            for t, norm in curves:
                ax.plot(t, norm, color=color, linestyle=ls, linewidth=0.6, alpha=INDIVIDUAL_ALPHA)

            grid, mean_norm = mean_norm_curve(curves)
            ax.plot(grid, mean_norm, color=color, linestyle=ls, linewidth=1.5, label=material)

        ax.set_xscale("log")
        ax.set_xlabel("Time since peak stress (s)")
        ax.set_title(f"{rate_group} rate", fontsize=9)
        ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)

    axes[0].set_ylabel("Normalized stress")
    fig.suptitle("Composition changes the shape of relaxation, not just its extent", fontsize=10)
    fig.tight_layout()
    _save(fig, "slide4_relaxation_shape_differences")


def load_test(cfg, material, test_id):
    df = pd.read_csv(cfg["dir"] / f"{material}-{test_id}_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


def mean_stress_time(dfs):
    """Mean stress-vs-time curve on a common linear time grid, NaN-masking
    each specimen past its own duration so a shorter test shrinks the
    mean's effective N near the tail instead of truncating the whole mean
    early."""
    common_time = np.linspace(0, max(df["time_s"].max() for df in dfs), 5000)
    stresses = []
    for df in dfs:
        v = np.interp(common_time, df["time_s"], df["stress"])
        v[common_time > df["time_s"].max() * (1 + 1e-9)] = np.nan
        stresses.append(v)
    return common_time, np.nanmean(stresses, axis=0)


def plot_stress_time_by_composition():
    # All 5 materials now have real data (A75V25/A100V0 were still pending
    # when this slide -- and this module's own local 3-material MATERIALS --
    # were first written). Narrower per-panel width than a straight 3-panel
    # scale-up, and NOT sharey: stress levels differ by >20x across
    # compositions (A0V100 ~25 MPa vs A100V0 ~1 MPa), so a shared axis would
    # flatten the softer materials' curves to invisibility.
    n = len(MATERIALS_ALL)
    fig, axes = plt.subplots(1, n, figsize=(figsize_double[0] / 4.2 * n, figsize_double[1]), sharey=False)

    for ax, (material, cfg) in zip(axes, MATERIALS_ALL.items()):
        for i, (rate_group, _, ls) in enumerate(RATE_GROUPS):
            color = colors[i]
            ids = cfg["rate_groups"][rate_group]
            dfs = [load_test(cfg, material, test_id) for test_id in ids]
            for df in dfs:
                ax.plot(df["time_s"], df["stress"], color=color, linestyle=ls,
                        linewidth=0.6, alpha=INDIVIDUAL_ALPHA)

            t, mean_stress = mean_stress_time(dfs)
            ax.plot(t, mean_stress, color=color, linestyle=ls, linewidth=1.5,
                    label=f"{rate_group} rate")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(material, fontsize=9)
        ax.legend(fontsize=7, handlelength=1.5, borderpad=0.3)


    fig.suptitle("Stress relaxation curves by composition and rate", fontsize=10)
    fig.tight_layout()
    _save(fig, "slide5_stress_time_by_composition")


def _save(fig, name):
    out_path = BASE / f"{name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot_modulus_vs_composition()
    plot_rate_sensitivity_vs_composition()
    plot_relaxation_vs_composition()
    plot_relaxation_shape_differences()
    plot_stress_time_by_composition()
