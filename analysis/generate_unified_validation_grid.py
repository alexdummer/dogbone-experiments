#!/usr/bin/env python3
"""Generate publication-ready 5x3 validation grid comparing all 3 material models:
- Generalized-Maxwell / QLV-Marmot (Neo-Hookean)
- Bergström-Boyce (Arruda-Boyce)
- Fractional Schapery (Neo-Hookean)
against measured tensile stress-relaxation data with embedded loading-ramp insets
under an idealized relaxation protocol.
"""

from pathlib import Path
import sys

REPO_ROOT = Path('/home/alex/projects/3D-printing/dogbone-experiments')
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "bergstrom-boyce"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "qlv-marmot"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "fractional-schapery"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "1d-qlv"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "prf"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plotstyle import colors, figsize_double
from fit_bb_model import BASE, MATERIALS, PLACEHOLDER_MATERIALS, simulate_bb
import fit_qlv_marmot_model as qm
from fit_fractional_schapery_model import _simulate_once as sim_fs
from fit_qlv_model import load, find_ramp_bounds, mean_strain_rates
from fit_prf_model import downsample_for_fit
from poisson_ratios import POISSON_RATIOS

PARAMS_DIR = REPO_ROOT / "simulations" / "material-calibration" / "parameters"
RATE_GROUPS = ["very slow", "slow", "fast"]

LAYOUT_CFG = {
    "A0V100": {
        "inset": [0.18, 0.14, 0.38, 0.38],
        "rate": (0.96, 0.96, "right", "top"),
        "ylim": (0, 18.5),
    },
    "A25V75": {
        "inset": [0.55, 0.54, 0.39, 0.38],
        "rate": (0.04, 0.96, "left", "top"),
        "ylim": (0, 31.0),
    },
    "A50V50": {
        "inset": [0.55, 0.52, 0.39, 0.40],
        "rate": (0.04, 0.96, "left", "top"),
        "ylim": (0, 20.0),
    },
    "A75V25": {
        "inset": [0.55, 0.52, 0.39, 0.40],
        "rate": (0.04, 0.96, "left", "top"),
        "ylim": (0, 6.2),
    },
    "A100V0": {
        "inset": [0.55, 0.52, 0.39, 0.40],
        "rate": (0.04, 0.96, "left", "top"),
        "ylim": (0, 0.85),
    },
}


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
        print(f"Simulating {material} (idealized)...", flush=True)
        nu = POISSON_RATIOS.get(material, 0.49)

        # 1. QLV-Marmot (Neo-Hookean)
        r_qlv = params_qlv[
            (params_qlv["material"] == material) & (params_qlv["hyperelastic_base"] == "NeoHooke")
        ].iloc[0]
        C1_qlv = r_qlv["C1"]
        gammas_qlv = [r_qlv["gamma1"], r_qlv["gamma2"], r_qlv["gamma3"]]
        taus_qlv = [r_qlv["tau1"], r_qlv["tau2"], r_qlv["tau3"]]

        # 2. Bergström-Boyce (Arruda-Boyce)
        r_bb = params_bb[
            (params_bb["material"] == material) & (params_bb["hyperelastic_base"] == "ArrudaBoyce")
        ].iloc[0]
        A_bb = (r_bb["A1"], r_bb["A2"], r_bb["A3"])
        B_bb = (r_bb["B1"], r_bb["B2"], r_bb["B3"])
        c1_bb, c2_bb, c3_bb = r_bb["c1"], r_bb["c2"], r_bb["c3"]

        # 3. Fractional Schapery (Neo-Hookean base)
        r_fs = params_fs[
            (params_fs["material"] == material) & (params_fs["hyperelastic_base"] == "NeoHooke")
        ].iloc[0]
        C1_fs = r_fs["C1"]
        alpha_fs, tau0_fs, g_inf_fs = r_fs["alpha"], r_fs["tau0"], r_fs["g_inf"]
        a1_fs, a2_fs, b_S_fs = r_fs["a1"], r_fs["a2"], r_fs["b_S"]

        qlv_time = {}
        qlv_ramp = {}
        bb_time = {}
        bb_ramp = {}
        fs_time = {}
        fs_ramp = {}

        for rg, ids in cfg["rate_groups"].items():
            test_id = ids[0]
            df = load(cfg, material, test_id)
            t_full = df["time_s"].to_numpy()
            measured = df["stress"].to_numpy()

            t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=40, n_hold=60)
            onset, ramp_end = find_ramp_bounds(df["position_mm"])
            t_ramp_end = df["time_s"].iloc[ramp_end]
            is_ramp = t_ds <= t_ramp_end
            peak_strain = df["strain"].iloc[ramp_end]
            mean_strain_rate = peak_strain / t_ramp_end

            # Idealized strain: linear ramp with average strain rate, constant peak strain in hold phase
            strain_ideal = np.where(is_ramp, mean_strain_rate * t_ds, peak_strain)
            stretch_ideal = 1.0 + strain_ideal

            # Dense ramp simulation for loading-ramp inset
            t_ramp_dense = np.linspace(0, t_ramp_end, 40)
            strain_ramp_dense = np.linspace(0, peak_strain, 40)
            stretch_ramp_dense = 1.0 + strain_ramp_dense

            # Raw ramp data for inset background
            sl_ramp = slice(onset, ramp_end + 1)
            strain_raw_ramp = df["strain"].to_numpy()[sl_ramp]
            stress_raw_ramp = df["stress"].to_numpy()[sl_ramp]

            # QLV (Neo-Hookean)
            p_qlv = qm._simulate_once(t_ds, stretch_ideal, "NeoHooke", C1_qlv, 0.0, 0.0, gammas_qlv, taus_qlv, nu=nu)
            p_qlv_ramp = qm._simulate_once(t_ramp_dense, stretch_ramp_dense, "NeoHooke", C1_qlv, 0.0, 0.0, gammas_qlv, taus_qlv, nu=nu)
            qlv_time[rg] = (t_full, measured, t_ds, p_qlv)
            qlv_ramp[rg] = (100 * strain_raw_ramp, stress_raw_ramp, 100 * strain_ramp_dense, p_qlv_ramp)

            # BB (Arruda-Boyce)
            p_bb = simulate_bb(t_ds, stretch_ideal, "ArrudaBoyce", A_bb, B_bb, c1_bb, c2_bb, c3_bb, nu=nu)
            p_bb_ramp = simulate_bb(t_ramp_dense, stretch_ramp_dense, "ArrudaBoyce", A_bb, B_bb, c1_bb, c2_bb, c3_bb, nu=nu)
            bb_time[rg] = (t_full, measured, t_ds, p_bb)
            bb_ramp[rg] = (100 * strain_raw_ramp, stress_raw_ramp, 100 * strain_ramp_dense, p_bb_ramp)

            # FS (Neo-Hookean)
            p_fs = sim_fs(t_ds, stretch_ideal, "NeoHooke", C1_fs, 0.0, 0.0, alpha_fs, tau0_fs, g_inf_fs, a1_fs, a2_fs, b_S_fs, nu=nu)
            p_fs_ramp = sim_fs(t_ramp_dense, stretch_ramp_dense, "NeoHooke", C1_fs, 0.0, 0.0, alpha_fs, tau0_fs, g_inf_fs, a1_fs, a2_fs, b_S_fs, nu=nu)
            fs_time[rg] = (t_full, measured, t_ds, p_fs)
            fs_ramp[rg] = (100 * strain_raw_ramp, stress_raw_ramp, 100 * strain_ramp_dense, p_fs_ramp)

        qlv_time_data[material] = qlv_time
        qlv_ramp_data[material] = qlv_ramp
        bb_time_data[material] = bb_time
        bb_ramp_data[material] = bb_ramp
        fs_time_data[material] = fs_time
        fs_ramp_data[material] = fs_ramp

    return qlv_time_data, qlv_ramp_data, bb_time_data, bb_ramp_data, fs_time_data, fs_ramp_data


def plot_unified_grid(qlv_time_data, qlv_ramp_data, bb_time_data, bb_ramp_data, fs_time_data, fs_ramp_data):
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    strain_rates = mean_strain_rates()

    height_ratios = [1.0] * len(materials)
    MAX_TOTAL_HEIGHT_IN = 8.5
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
        cfg = LAYOUT_CFG[material]
        for col, rate_group in enumerate(RATE_GROUPS):
            ax = axes[row, col]
            color = colors[col]

            # 1. Main relaxation curve
            t_full, measured, t_ds_qlv, pred_qlv = qlv_time_data[material][rate_group]
            _, _, t_ds_bb, pred_bb = bb_time_data[material][rate_group]
            _, _, t_ds_fs, pred_fs = fs_time_data[material][rate_group]

            # Measured curve (faded grey)
            if measured is not None:
                ax.plot(t_full, measured, color="0.4", linewidth=2.0, alpha=0.35, zorder=1)

            # Generalized Maxwell (Mooney-Rivlin): solid line
            ax.plot(
                t_ds_qlv,
                pred_qlv,
                color=color,
                linewidth=1.0,
                linestyle="-",
                zorder=3,
            )

            # Bergström-Boyce (Arruda-Boyce): dashed line
            ax.plot(
                t_ds_bb,
                pred_bb,
                color=color,
                linewidth=1.0,
                linestyle="--",
                zorder=4,
            )

            # Fractional Schapery: dash-dot line
            ax.plot(
                t_ds_fs,
                pred_fs,
                color=color,
                linewidth=1.0,
                linestyle="-.",
                zorder=5,
            )

            ax.grid(True, alpha=0.3)

            if row == 0:
                rate_title = {"very slow": "slow", "slow": "medium", "fast": "fast"}.get(rate_group, rate_group)
                ax.set_title(rate_title, fontsize=9)
            disp_mat = {
                "A0V100": "A0",
                "A25V75": "A25",
                "A50V50": "A50",
                "A75V25": "A75",
                "A100V0": "A100",
            }.get(material, material)
            if col == 0:
                if disp_mat == "A50":
                    ax.set_ylabel(f"{disp_mat}\nEngineering stress in MPa", fontsize=8)
                else:
                    ax.set_ylabel(disp_mat, fontsize=8)
            if row == len(materials) - 1:
                ax.set_xlabel("Time in s", fontsize=8)

            rate = strain_rates.get(material, {}).get(rate_group)
            if rate is not None:
                lx, ly, ha, va = cfg["rate"]
                ax.text(
                    lx,
                    ly,
                    rf"$\dot\varepsilon\approx${100 * rate:.3g}\%/s",
                    fontsize=6,
                    ha=ha,
                    va=va,
                    transform=ax.transAxes,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
                )

            # Explicit ylim per row for breathing room
            ax.set_ylim(cfg["ylim"])

            # 2. INSET AXIS for loading ramp stress-strain
            rx_full, r_meas, rx_ds_qlv, r_pred_qlv = qlv_ramp_data[material][rate_group]
            _, _, rx_ds_bb, r_pred_bb = bb_ramp_data[material][rate_group]
            _, _, rx_ds_fs, r_pred_fs = fs_ramp_data[material][rate_group]

            ax_ins = ax.inset_axes(cfg["inset"])
            if r_meas is not None:
                ax_ins.plot(rx_full, r_meas, color="0.4", linewidth=1.5, alpha=0.35, zorder=1)

            ax_ins.plot(
                rx_ds_qlv,
                r_pred_qlv,
                color=color,
                linewidth=0.8,
                linestyle="-",
                zorder=3,
            )
            ax_ins.plot(
                rx_ds_bb,
                r_pred_bb,
                color=color,
                linewidth=0.8,
                linestyle="--",
                zorder=4,
            )
            ax_ins.plot(
                rx_ds_fs,
                r_pred_fs,
                color=color,
                linewidth=0.8,
                linestyle="-.",
                zorder=5,
            )

            ax_ins.set_facecolor("white")
            ax_ins.tick_params(labelsize=4.5, pad=1, length=1.5, width=0.5)
            ax_ins.grid(True, alpha=0.25, linewidth=0.4)
            for spine in ax_ins.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#555555")
            ax_ins.set_xlabel(r"$\varepsilon$ (\%)", fontsize=5.0, labelpad=0.5)
            ax_ins.set_ylabel(r"$\sigma$ (MPa)", fontsize=5.0, labelpad=0.5)

    legend_handles = [
        Line2D([0], [0], color="0.4", alpha=0.5, linewidth=2.0, label="measured"),
        Line2D([0], [0], color="black", linestyle="-", label="Generalized-Maxwell (Neo-Hookean)"),
        Line2D([0], [0], color="black", linestyle="--", label="Bergström-Boyce (Arruda-Boyce)"),
        Line2D([0], [0], color="black", linestyle="-.", label="Fractional Schapery (Neo-Hookean)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=7.0, bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout()

    scratch_png = Path("/home/alex/.gemini/antigravity/brain/7d08f8bf-6bf9-4623-973d-65f5d483607b/scratch/unified_validation_grid.png")
    paper_pdf = Path("/home/alex/projects/3D-printing/paper/figures/qlv_marmot_validation_grid.pdf")
    analysis_pdf = REPO_ROOT / "analysis" / "qlv_marmot_validation_grid.pdf"
    sim_pdf = REPO_ROOT / "simulations" / "material-calibration" / "qlv_marmot_validation_grid.pdf"

    fig.savefig(scratch_png, dpi=300, bbox_inches="tight")
    fig.savefig(paper_pdf, bbox_inches="tight")
    fig.savefig(analysis_pdf, bbox_inches="tight")
    fig.savefig(sim_pdf, bbox_inches="tight")

    print(f"Saved scratch png: {scratch_png}")
    print(f"Saved paper pdf: {paper_pdf}")
    plt.close(fig)


def main():
    print("=== Generating Unified 3-Model Validation Grid ===", flush=True)
    qlv_time, qlv_ramp, bb_time, bb_ramp, fs_time, fs_ramp = simulate_all()
    plot_unified_grid(qlv_time, qlv_ramp, bb_time, bb_ramp, fs_time, fs_ramp)
    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
