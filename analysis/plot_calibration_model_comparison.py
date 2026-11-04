"""Direct comparison of calibrated QLV-Marmot (Mooney-Rivlin), Bergström-Boyce
(Arruda-Boyce), and Fractional Schapery models against experimental
stress-relaxation test results across all 5 photopolymer blends.

Generates:
  - Full 5-material x 3-rate comparison grid (calibration_model_comparison_grid.pdf/.png)
  - Full 5-material x 3-rate ramp stress-strain grid (calibration_model_comparison_ramp_grid.pdf/.png)
  - Per-material 1-row x 3-rate overview and ramp figures.
Follows plotstyle.py conventions.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "bergstrom-boyce"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "qlv-marmot"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "fractional-schapery"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "1d-qlv"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "prf"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plotstyle import colors, figsize_double
from fit_bb_model import BASE, MATERIALS, PLACEHOLDER_MATERIALS, validate_bb, validate_bb_ramp
from fit_qlv_marmot_model import validate as validate_qlv, validate_ramp as validate_qlv_ramp
from fit_fractional_schapery_model import validate as validate_fs, validate_ramp as validate_fs_ramp
from fit_qlv_model import load, mean_strain_rates

PARAMS_DIR = REPO_ROOT / "simulations" / "material-calibration" / "parameters"
OUT_DIRS = [REPO_ROOT / "analysis", REPO_ROOT / "simulations" / "material-calibration"]

RATE_GROUPS = ["very slow", "slow", "fast"]
RATE_TITLES = ["Very slow rate", "Slow rate", "Fast rate"]


def load_parameters():
    params_qlv = pd.read_csv(PARAMS_DIR / "qlv_marmot_params.csv")
    params_bb = pd.read_csv(PARAMS_DIR / "bb_params.csv")
    params_fs = pd.read_csv(PARAMS_DIR / "fractional_schapery_params.csv")
    return params_qlv, params_bb, params_fs


def simulate_all():
    params_qlv, params_bb, params_fs = load_parameters()
    qlv_time_data = {}
    qlv_ramp_data = {}
    bb_time_data = {}
    bb_ramp_data = {}
    fs_time_data = {}
    fs_ramp_data = {}

    for material, cfg in MATERIALS.items():
        print(f"Simulating {material}...", flush=True)

        # 1. QLV-Marmot (Mooney-Rivlin)
        r_qlv = params_qlv[
            (params_qlv["material"] == material) & (params_qlv["hyperelastic_base"] == "MooneyRivlin")
        ].iloc[0]
        _, qlv_time = validate_qlv(
            material,
            cfg,
            "MooneyRivlin",
            r_qlv["C1"],
            r_qlv["C2"],
            r_qlv["C3"],
            [r_qlv["gamma1"], r_qlv["gamma2"], r_qlv["gamma3"]],
            [r_qlv["tau1"], r_qlv["tau2"], r_qlv["tau3"]],
        )
        qlv_ramp = validate_qlv_ramp(
            material,
            cfg,
            "MooneyRivlin",
            r_qlv["C1"],
            r_qlv["C2"],
            r_qlv["C3"],
            [r_qlv["gamma1"], r_qlv["gamma2"], r_qlv["gamma3"]],
            [r_qlv["tau1"], r_qlv["tau2"], r_qlv["tau3"]],
        )
        qlv_time_data[material] = qlv_time
        qlv_ramp_data[material] = qlv_ramp

        # 2. Bergström-Boyce (Arruda-Boyce)
        r_bb = params_bb[
            (params_bb["material"] == material) & (params_bb["hyperelastic_base"] == "ArrudaBoyce")
        ].iloc[0]
        _, bb_time = validate_bb(
            material,
            cfg,
            "ArrudaBoyce",
            (r_bb["A1"], r_bb["A2"], r_bb["A3"]),
            (r_bb["B1"], r_bb["B2"], r_bb["B3"]),
            r_bb["c1"],
            r_bb["c2"],
            r_bb["c3"],
        )
        bb_ramp = validate_bb_ramp(
            material,
            cfg,
            "ArrudaBoyce",
            (r_bb["A1"], r_bb["A2"], r_bb["A3"]),
            (r_bb["B1"], r_bb["B2"], r_bb["B3"]),
            r_bb["c1"],
            r_bb["c2"],
            r_bb["c3"],
        )
        bb_time_data[material] = bb_time
        bb_ramp_data[material] = bb_ramp

        # 3. Fractional Schapery (NeoHooke base)
        r_fs = params_fs[
            (params_fs["material"] == material) & (params_fs["hyperelastic_base"] == "NeoHooke")
        ].iloc[0]
        _, fs_time = validate_fs(
            material,
            cfg,
            "NeoHooke",
            r_fs["C1"],
            r_fs["C2"],
            r_fs["C3"],
            r_fs["alpha"],
            r_fs["tau0"],
            r_fs["g_inf"],
            r_fs["a1"],
            r_fs["a2"],
            r_fs["b_S"],
        )
        fs_ramp = validate_fs_ramp(
            material,
            cfg,
            "NeoHooke",
            r_fs["C1"],
            r_fs["C2"],
            r_fs["C3"],
            r_fs["alpha"],
            r_fs["tau0"],
            r_fs["g_inf"],
            r_fs["a1"],
            r_fs["a2"],
            r_fs["b_S"],
        )
        fs_time_data[material] = fs_time
        fs_ramp_data[material] = fs_ramp

    return qlv_time_data, qlv_ramp_data, bb_time_data, bb_ramp_data, fs_time_data, fs_ramp_data


def save_multidir(fig, base_name):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        p = d / base_name
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"saved {p.with_suffix('.pdf')} and .png")


def plot_material_overview(material, qlv_time, bb_time, fs_time, strain_rates):
    """Generate 1-row x 3-rate figsize_double comparison plot for a specific material."""
    cfg = MATERIALS[material]
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=True)

    for col, (ax, rate_group, title_prefix) in enumerate(zip(axes, RATE_GROUPS, RATE_TITLES)):
        ids = cfg["rate_groups"].get(rate_group, [])
        for test_id in ids:
            df = load(cfg, material, test_id)
            ax.plot(
                df["time_s"],
                df["stress"],
                color="0.75",
                linewidth=0.7,
                alpha=0.5,
                zorder=1,
            )

        if rate_group in qlv_time and rate_group in bb_time and rate_group in fs_time:
            t_full, measured, t_ds_qlv, pred_qlv = qlv_time[rate_group]
            _, _, t_ds_bb, pred_bb = bb_time[rate_group]
            _, _, t_ds_fs, pred_fs = fs_time[rate_group]

            # Measured representative test
            ax.plot(t_full, measured, color="black", linewidth=1.2, alpha=0.45, zorder=2)

            # QLV-Marmot (Mooney-Rivlin)
            ax.plot(t_ds_qlv, pred_qlv, color=colors[0], linestyle="-", linewidth=1.2,
                    marker="o", markersize=2.0, markevery=4, zorder=3)

            # Bergström-Boyce (Arruda-Boyce)
            ax.plot(t_ds_bb, pred_bb, color=colors[1], linestyle="--", linewidth=1.2,
                    marker="^", markersize=2.0, markevery=4, zorder=4)

            # Fractional Schapery (NeoHooke)
            ax.plot(t_ds_fs, pred_fs, color=colors[2], linestyle="-.", linewidth=1.2,
                    marker="s", markersize=2.0, markevery=4, zorder=5)

        rate = strain_rates.get(material, {}).get(rate_group)
        rate_str = rf" ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)" if rate else ""
        ax.set_title(f"{title_prefix}{rate_str}", fontsize=8)
        ax.set_xlabel("Time in s")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("Engineering stress in MPa")

    handles = [
        Line2D([0], [0], color="0.75", linewidth=1.5, label="replicates"),
        Line2D([0], [0], color="black", linewidth=1.2, alpha=0.6, label="measured"),
        Line2D([0], [0], color=colors[0], linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot"),
        Line2D([0], [0], color=colors[1], linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce"),
        Line2D([0], [0], color=colors[2], linestyle="-.", marker="s", markersize=3.5, label="Fractional Schapery"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7.5)

    save_multidir(fig, f"calibration_model_comparison_{material}")
    plt.close(fig)


def plot_material_ramp(material, qlv_ramp, bb_ramp, fs_ramp, strain_rates):
    """Generate 1-row x 3-rate figsize_double ramp stress-strain comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=True)

    for col, (ax, rate_group, title_prefix) in enumerate(zip(axes, RATE_GROUPS, RATE_TITLES)):
        if rate_group in qlv_ramp and rate_group in bb_ramp and rate_group in fs_ramp:
            strain_full, measured, strain_ds_qlv, pred_qlv = qlv_ramp[rate_group]
            _, _, strain_ds_bb, pred_bb = bb_ramp[rate_group]
            _, _, strain_ds_fs, pred_fs = fs_ramp[rate_group]

            # Measured ramp curve
            ax.plot(strain_full, measured, color="black", linewidth=1.2, alpha=0.45, zorder=2)

            # QLV-Marmot
            ax.plot(strain_ds_qlv, pred_qlv, color=colors[0], linestyle="-", linewidth=1.2,
                    marker="o", markersize=2.0, markevery=2, zorder=3)

            # Bergström-Boyce
            ax.plot(strain_ds_bb, pred_bb, color=colors[1], linestyle="--", linewidth=1.2,
                    marker="^", markersize=2.0, markevery=2, zorder=4)

            # Fractional Schapery
            ax.plot(strain_ds_fs, pred_fs, color=colors[2], linestyle="-.", linewidth=1.2,
                    marker="s", markersize=2.0, markevery=2, zorder=5)

        rate = strain_rates.get(material, {}).get(rate_group)
        rate_str = rf" ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)" if rate else ""
        ax.set_title(f"{title_prefix}{rate_str}", fontsize=8)
        ax.set_xlabel(r"Engineering strain in \%")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("Engineering stress in MPa")

    handles = [
        Line2D([0], [0], color="black", linewidth=1.2, alpha=0.6, label="measured"),
        Line2D([0], [0], color=colors[0], linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot"),
        Line2D([0], [0], color=colors[1], linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce"),
        Line2D([0], [0], color=colors[2], linestyle="-.", marker="s", markersize=3.5, label="Fractional Schapery"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=7.5)

    save_multidir(fig, f"calibration_model_comparison_{material}_ramp")
    plt.close(fig)


def plot_combined_grid(qlv_time_data, bb_time_data, fs_time_data, strain_rates):
    """Full 5-material x 3-rate comparison grid for stress relaxation (width = figsize_double[0])."""
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    n_real = len(MATERIALS)
    height_ratios = [1.0] * n_real + [0.4] * len(PLACEHOLDER_MATERIALS)
    MAX_TOTAL_HEIGHT_IN = 7.5
    per_row_scale = min(0.8, MAX_TOTAL_HEIGHT_IN / (figsize_double[1] * sum(height_ratios)))

    fig, axes = plt.subplots(
        len(materials),
        3,
        figsize=(figsize_double[0], figsize_double[1] * per_row_scale * sum(height_ratios)),
        sharey="row",
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    for row, material in enumerate(materials):
        is_placeholder = material not in qlv_time_data
        for col, rate_group in enumerate(RATE_GROUPS):
            ax = axes[row, col]
            if is_placeholder:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=7,
                        color="grey", style="italic", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                if row == 0:
                    ax.set_title(rate_group, fontsize=9)
                if col == 0:
                    ax.set_ylabel(material, fontsize=8)
                continue

            rate_color = colors[col]
            t_full, measured, t_ds_qlv, pred_qlv = qlv_time_data[material][rate_group]
            _, _, t_ds_bb, pred_bb = bb_time_data[material][rate_group]
            _, _, t_ds_fs, pred_fs = fs_time_data[material][rate_group]

            # Measured curve (faded grey)
            ax.plot(t_full, measured, color="grey", linewidth=1.8, alpha=0.35, zorder=1)

            # QLV-Marmot: solid line with circle markers
            ax.plot(t_ds_qlv, pred_qlv, color=rate_color, linewidth=1.1, linestyle="-",
                    marker="o", markersize=1.8, markevery=4, zorder=2)

            # Bergström-Boyce: dashed line with triangle markers
            ax.plot(t_ds_bb, pred_bb, color=rate_color, linewidth=1.1, linestyle="--",
                    marker="^", markersize=1.8, markevery=4, zorder=3)

            # Fractional Schapery: dash-dot line with square markers
            ax.plot(t_ds_fs, pred_fs, color=rate_color, linewidth=1.1, linestyle="-.",
                    marker="s", markersize=1.8, markevery=4, zorder=4)

            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(rate_group, fontsize=9)
            if col == 0:
                disp_mat = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}.get(material, material)
                ax.set_ylabel(f"{disp_mat}\nEngineering stress in MPa", fontsize=8)
            if row == n_real - 1:
                ax.set_xlabel("Time in s")

            rate = strain_rates.get(material, {}).get(rate_group)
            if rate is not None:
                ax.text(
                    0.96,
                    0.96,
                    rf"$\dot\varepsilon\approx${100 * rate:.3g}\%/s",
                    fontsize=6,
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
                )

    handles = [
        Line2D([0], [0], color="grey", alpha=0.45, linewidth=2.0, label="measured"),
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot"),
        Line2D([0], [0], color="black", linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce"),
        Line2D([0], [0], color="black", linestyle="-.", marker="s", markersize=3.5, label="Fractional Schapery"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    save_multidir(fig, "calibration_model_comparison_grid")
    plt.close(fig)


def plot_combined_ramp_grid(qlv_ramp_data, bb_ramp_data, fs_ramp_data, strain_rates):
    """Full 5-material x 3-rate comparison grid for stress-strain loading ramps."""
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    n_real = len(MATERIALS)
    height_ratios = [1.0] * n_real + [0.4] * len(PLACEHOLDER_MATERIALS)
    MAX_TOTAL_HEIGHT_IN = 7.5
    per_row_scale = min(0.8, MAX_TOTAL_HEIGHT_IN / (figsize_double[1] * sum(height_ratios)))

    fig, axes = plt.subplots(
        len(materials),
        3,
        figsize=(figsize_double[0], figsize_double[1] * per_row_scale * sum(height_ratios)),
        sharey="row",
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    for row, material in enumerate(materials):
        is_placeholder = material not in qlv_ramp_data
        for col, rate_group in enumerate(RATE_GROUPS):
            ax = axes[row, col]
            if is_placeholder:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=7,
                        color="grey", style="italic", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                if row == 0:
                    ax.set_title(rate_group, fontsize=9)
                if col == 0:
                    ax.set_ylabel(material, fontsize=8)
                continue

            rate_color = colors[col]
            strain_full, measured, strain_ds_qlv, pred_qlv = qlv_ramp_data[material][rate_group]
            _, _, strain_ds_bb, pred_bb = bb_ramp_data[material][rate_group]
            _, _, strain_ds_fs, pred_fs = fs_ramp_data[material][rate_group]

            # Measured curve (faded grey)
            ax.plot(strain_full, measured, color="grey", linewidth=1.8, alpha=0.35, zorder=1)

            # QLV-Marmot: solid line with circle markers
            ax.plot(strain_ds_qlv, pred_qlv, color=rate_color, linewidth=1.1, linestyle="-",
                    marker="o", markersize=1.8, markevery=2, zorder=2)

            # Bergström-Boyce: dashed line with triangle markers
            ax.plot(strain_ds_bb, pred_bb, color=rate_color, linewidth=1.1, linestyle="--",
                    marker="^", markersize=1.8, markevery=2, zorder=3)

            # Fractional Schapery: dash-dot line with square markers
            ax.plot(strain_ds_fs, pred_fs, color=rate_color, linewidth=1.1, linestyle="-.",
                    marker="s", markersize=1.8, markevery=2, zorder=4)

            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(rate_group, fontsize=9)
            if col == 0:
                disp_mat = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}.get(material, material)
                ax.set_ylabel(f"{disp_mat}\nEngineering stress in MPa", fontsize=8)
            if row == n_real - 1:
                ax.set_xlabel(r"Engineering strain in \%")

            rate = strain_rates.get(material, {}).get(rate_group)
            if rate is not None:
                ax.text(
                    0.04,
                    0.96,
                    rf"$\dot\varepsilon\approx${100 * rate:.3g}\%/s",
                    fontsize=6,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
                )

    handles = [
        Line2D([0], [0], color="grey", alpha=0.45, linewidth=2.0, label="measured"),
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot"),
        Line2D([0], [0], color="black", linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce"),
        Line2D([0], [0], color="black", linestyle="-.", marker="s", markersize=3.5, label="Fractional Schapery"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout()
    save_multidir(fig, "calibration_model_comparison_ramp_grid")
    plt.close(fig)


def main():
    print("=== Generating multi-model calibration comparison grids ===", flush=True)
    strain_rates = mean_strain_rates()
    qlv_time, qlv_ramp, bb_time, bb_ramp, fs_time, fs_ramp = simulate_all()

    # 1. Per-material 1x3 figsize_double overview and ramp plots
    for material in MATERIALS:
        plot_material_overview(material, qlv_time[material], bb_time[material], fs_time[material], strain_rates)
        plot_material_ramp(material, qlv_ramp[material], bb_ramp[material], fs_ramp[material], strain_rates)

    # 2. Combined 5-material x 3-rate relaxation grid
    plot_combined_grid(qlv_time, bb_time, fs_time, strain_rates)

    # 3. Combined 5-material x 3-rate ramp grid
    plot_combined_ramp_grid(qlv_ramp, bb_ramp, fs_ramp, strain_rates)

    print("=== Multi-model comparison grids generated successfully ===", flush=True)


if __name__ == "__main__":
    main()
