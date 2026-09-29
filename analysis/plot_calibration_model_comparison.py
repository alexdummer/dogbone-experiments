"""Direct comparison of calibrated QLV-Marmot (Mooney-Rivlin) and Bergström-Boyce
(Arruda-Boyce) models against experimental stress-relaxation test results in a
figsize_double layout.

Matches the presentation of the experimental results (e.g. A25V75_stress_overview.pdf:
1 row, 3 rate columns: very slow, slow, fast rate, sharey=True, figsize=figsize_double)
with both calibrated constitutive models overlaid.

Also generates the full 5-material x 3-rate comparison grid
(calibration_model_comparison_grid.pdf/.png) and ramp stress-strain comparison.
Follows plotstyle.py conventions.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plotstyle import colors, figsize_double
from fit_bb_model import BASE, MATERIALS, PLACEHOLDER_MATERIALS, validate_bb, validate_bb_ramp
from fit_qlv_marmot_model import validate, validate_ramp
from fit_qlv_model import load, mean_strain_rates

OUT_DIR = BASE
RATE_GROUPS = ["very slow", "slow", "fast"]
RATE_TITLES = ["Very slow rate", "Slow rate", "Fast rate"]


def load_parameters():
    params_qlv = pd.read_csv(BASE / "qlv_marmot_params.csv")
    params_bb = pd.read_csv(BASE / "bb_params.csv")
    return params_qlv, params_bb


def simulate_all():
    params_qlv, params_bb = load_parameters()
    qlv_time_data = {}
    qlv_ramp_data = {}
    bb_time_data = {}
    bb_ramp_data = {}

    for material, cfg in MATERIALS.items():
        print(f"Simulating {material}...", flush=True)
        r_qlv = params_qlv[
            (params_qlv["material"] == material) & (params_qlv["hyperelastic_base"] == "MooneyRivlin")
        ].iloc[0]
        _, qlv_time = validate(
            material,
            cfg,
            "MooneyRivlin",
            r_qlv["C1"],
            r_qlv["C2"],
            r_qlv["C3"],
            [r_qlv["gamma1"], r_qlv["gamma2"], r_qlv["gamma3"]],
            [r_qlv["tau1"], r_qlv["tau2"], r_qlv["tau3"]],
        )
        qlv_ramp = validate_ramp(
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

    return qlv_time_data, qlv_ramp_data, bb_time_data, bb_ramp_data


def plot_material_overview(material, qlv_time, bb_time, strain_rates):
    """Generate 1-row x 3-rate figsize_double comparison plot for a specific material."""
    cfg = MATERIALS[material]
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=True)

    for col, (ax, rate_group, title_prefix) in enumerate(zip(axes, RATE_GROUPS, RATE_TITLES)):
        # Replicate experimental test curves (faded)
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

        # QLV and BB predictions
        if rate_group in qlv_time and rate_group in bb_time:
            t_full, measured, t_ds_qlv, pred_qlv = qlv_time[rate_group]
            _, _, t_ds_bb, pred_bb = bb_time[rate_group]

            # Measured representative test
            ax.plot(
                t_full,
                measured,
                color="black",
                linewidth=1.2,
                alpha=0.45,
                zorder=2,
            )

            # QLV-Marmot (Mooney-Rivlin)
            ax.plot(
                t_ds_qlv,
                pred_qlv,
                color=colors[0],
                linestyle="-",
                linewidth=1.3,
                marker="o",
                markersize=2.2,
                markevery=4,
                zorder=3,
            )

            # Bergström-Boyce (Arruda-Boyce)
            ax.plot(
                t_ds_bb,
                pred_bb,
                color=colors[1],
                linestyle="--",
                linewidth=1.3,
                marker="^",
                markersize=2.2,
                markevery=4,
                zorder=4,
            )

        rate = strain_rates.get(material, {}).get(rate_group)
        rate_str = rf" ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)" if rate else ""
        ax.set_title(f"{title_prefix}{rate_str}", fontsize=8)
        ax.set_xlabel("Time in s")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("Engineering stress in MPa")

    handles = [
        Line2D([0], [0], color="0.75", linewidth=1.5, label="replicates"),
        Line2D([0], [0], color="black", linewidth=1.2, alpha=0.6, label="measured"),
        Line2D([0], [0], color=colors[0], linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot (Mooney-Rivlin)"),
        Line2D([0], [0], color=colors[1], linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce (Arruda-Boyce)"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7.5,
    )

    out_base = OUT_DIR / f"calibration_model_comparison_{material}"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_base}.pdf and .png")
    plt.close(fig)


def plot_material_ramp(material, qlv_ramp, bb_ramp, strain_rates):
    """Generate 1-row x 3-rate figsize_double ramp stress-strain comparison plot."""
    cfg = MATERIALS[material]
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=True)

    for col, (ax, rate_group, title_prefix) in enumerate(zip(axes, RATE_GROUPS, RATE_TITLES)):
        if rate_group in qlv_ramp and rate_group in bb_ramp:
            strain_full, measured, strain_ds_qlv, pred_qlv = qlv_ramp[rate_group]
            _, _, strain_ds_bb, pred_bb = bb_ramp[rate_group]

            # Measured ramp curve
            ax.plot(
                strain_full,
                measured,
                color="black",
                linewidth=1.2,
                alpha=0.45,
                zorder=2,
            )

            # QLV-Marmot
            ax.plot(
                strain_ds_qlv,
                pred_qlv,
                color=colors[0],
                linestyle="-",
                linewidth=1.3,
                marker="o",
                markersize=2.2,
                markevery=2,
                zorder=3,
            )

            # Bergström-Boyce
            ax.plot(
                strain_ds_bb,
                pred_bb,
                color=colors[1],
                linestyle="--",
                linewidth=1.3,
                marker="^",
                markersize=2.2,
                markevery=2,
                zorder=4,
            )

        rate = strain_rates.get(material, {}).get(rate_group)
        rate_str = rf" ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)" if rate else ""
        ax.set_title(f"{title_prefix}{rate_str}", fontsize=8)
        ax.set_xlabel(r"Engineering strain in \%")
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel("Engineering stress in MPa")

    handles = [
        Line2D([0], [0], color="black", linewidth=1.2, alpha=0.6, label="measured"),
        Line2D([0], [0], color=colors[0], linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot (Mooney-Rivlin)"),
        Line2D([0], [0], color=colors[1], linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce (Arruda-Boyce)"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7.5,
    )

    out_base = OUT_DIR / f"calibration_model_comparison_{material}_ramp"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_base}.pdf and .png")
    plt.close(fig)


def plot_combined_grid(qlv_time_data, bb_time_data, strain_rates):
    """Full 5-material x 3-rate comparison grid (width = figsize_double[0])."""
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
                ax.text(
                    0.5,
                    0.5,
                    "pending",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="grey",
                    style="italic",
                    transform=ax.transAxes,
                )
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

            # Measured curve (faded)
            ax.plot(t_full, measured, color="grey", linewidth=1.8, alpha=0.35, zorder=1)

            # QLV-Marmot: solid line with circle markers
            ax.plot(
                t_ds_qlv,
                pred_qlv,
                color=rate_color,
                linewidth=1.1,
                linestyle="-",
                marker="o",
                markersize=2.0,
                markevery=4,
                zorder=2,
            )

            # Bergström-Boyce: dashed line with triangle markers
            ax.plot(
                t_ds_bb,
                pred_bb,
                color=rate_color,
                linewidth=1.1,
                linestyle="--",
                marker="^",
                markersize=2.0,
                markevery=4,
                zorder=3,
            )

            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(rate_group, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{material}\nEngineering stress in MPa", fontsize=8)
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
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=3.5, label="QLV-Marmot (Mooney-Rivlin)"),
        Line2D([0], [0], color="black", linestyle="--", marker="^", markersize=3.5, label="Bergström-Boyce (Arruda-Boyce)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
    )
    fig.tight_layout()
    out_base = OUT_DIR / "calibration_model_comparison_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_base}.pdf and .png")
    plt.close(fig)


def main():
    strain_rates = mean_strain_rates()
    qlv_time, qlv_ramp, bb_time, bb_ramp = simulate_all()

    # 1. Per-material 1x3 figsize_double overview plots (matching <MAT>_stress_overview.pdf layout)
    for material in MATERIALS:
        plot_material_overview(material, qlv_time[material], bb_time[material], strain_rates)
        plot_material_ramp(material, qlv_ramp[material], bb_ramp[material], strain_rates)

    # 2. Combined 5-material x 3-rate grid
    plot_combined_grid(qlv_time, bb_time, strain_rates)


if __name__ == "__main__":
    main()
