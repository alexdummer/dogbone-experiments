"""Progressive 3-step calibration build-up plots in figsize_double layout.

Step 1: Experimental mean curve only (clean, no individual replicate clutter).
Step 2: Experimental mean curve + GM (Generalized-Maxwell / QLV-Marmot, Mooney-Rivlin).
        (Also outputs the alternative Step 2 with BB).
Step 3: Experimental mean curve + GM + BB (Bergström-Boyce, Arruda-Boyce).

Uses locked axis limits across all build-up steps so curves transition seamlessly
without axes shifting. Follows plotstyle.py conventions.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import colors, figsize_double  # noqa: E402
from fit_bb_model import BASE, MATERIALS, validate_bb  # noqa: E402
from fit_qlv_marmot_model import validate  # noqa: E402
from fit_qlv_model import load, mean_strain_rates  # noqa: E402

OUT_DIR = BASE
RATE_GROUPS = ["very slow", "slow", "fast"]
RATE_TITLES = ["Very slow rate", "Slow rate", "Fast rate"]


def interp_or_nan(common_time, t, y):
    values = np.interp(common_time, t, y)
    values[common_time > t.max()] = np.nan
    return values


def mean_stress(dfs):
    common_time = np.linspace(0, max(df["time_s"].max() for df in dfs), 5000)
    stresses = np.array([interp_or_nan(common_time, df["time_s"], df["stress"]) for df in dfs])
    return common_time, np.nanmean(stresses, axis=0)


def generate_buildup_for_material(material, qlv_plots, bb_plots, strain_rates):
    cfg = MATERIALS[material]

    # Compute experimental mean curves per rate group
    exp_curves = {}
    for rg in RATE_GROUPS:
        dfs = [load(cfg, material, test_id) for test_id in cfg["rate_groups"][rg]]
        exp_curves[rg] = mean_stress(dfs)

    # Locked y-axis range across all build-up stages to prevent axis shifting
    all_max_stresses = [exp_curves[rg][1].max() for rg in RATE_GROUPS]
    for rg in RATE_GROUPS:
        if rg in qlv_plots:
            all_max_stresses.append(qlv_plots[rg][3].max())
        if rg in bb_plots:
            all_max_stresses.append(bb_plots[rg][3].max())
    ymax = max(all_max_stresses) * 1.08
    ylim = (-0.5, ymax)

    def render_step(include_gm, include_bb, out_stem):
        fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=True)

        for col, (ax, rg, title_prefix) in enumerate(zip(axes, RATE_GROUPS, RATE_TITLES)):
            # 1. Experimental mean curve
            t_exp, s_exp = exp_curves[rg]
            ax.plot(
                t_exp,
                s_exp,
                color="0.2",
                linewidth=1.4,
                linestyle="-",
                label="Experiment",
                zorder=1,
            )

            # 2. GM model prediction
            if include_gm and rg in qlv_plots:
                _, _, t_ds_qlv, pred_qlv = qlv_plots[rg]
                ax.plot(
                    t_ds_qlv,
                    pred_qlv,
                    color=colors[0],
                    linestyle="-",
                    linewidth=1.3,
                    marker="o",
                    markersize=2.2,
                    markevery=4,
                    label="GM",
                    zorder=2,
                )

            # 3. BB model prediction
            if include_bb and rg in bb_plots:
                _, _, t_ds_bb, pred_bb = bb_plots[rg]
                ax.plot(
                    t_ds_bb,
                    pred_bb,
                    color=colors[1],
                    linestyle="--",
                    linewidth=1.3,
                    marker="^",
                    markersize=2.2,
                    markevery=4,
                    label="BB",
                    zorder=3,
                )

            rate = strain_rates.get(material, {}).get(rg)
            rate_str = rf" ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)" if rate else ""
            ax.set_title(f"{title_prefix}{rate_str}", fontsize=8)
            ax.set_xlabel("Time in s")
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.4)

        axes[0].set_ylabel("Engineering stress in MPa")

        # Top legend handles matching exactly what is shown in this step
        handles = [Line2D([0], [0], color="0.2", linewidth=1.4, label="Experiment")]
        if include_gm:
            handles.append(
                Line2D([0], [0], color=colors[0], linestyle="-", marker="o", markersize=3.5, label="GM")
            )
        if include_bb:
            handles.append(
                Line2D([0], [0], color=colors[1], linestyle="--", marker="^", markersize=3.5, label="BB")
            )

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            fontsize=8,
        )

        out_base = OUT_DIR / f"{out_stem}_{material}"
        for ext in ("pdf", "png"):
            out_file = out_base.with_suffix(f".{ext}")
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"saved {out_base}.pdf and .png")
        plt.close(fig)

    # Step 1: Experiment only (mean curve)
    render_step(False, False, "calibration_buildup_1_exp")

    # Step 2: Experiment + GM
    render_step(True, False, "calibration_buildup_2_gm")

    # Step 2 alt: Experiment + BB
    render_step(False, True, "calibration_buildup_2_bb")

    # Step 3: Experiment + GM + BB
    render_step(True, True, "calibration_buildup_3_both")


def main():
    parser = argparse.ArgumentParser(description="Generate 3-step progressive calibration comparison plots")
    parser.add_argument("--material", default="A25V75", choices=list(MATERIALS.keys()) + ["all"],
                        help="Material to generate build-up for (default: A25V75)")
    args = parser.parse_args()

    strain_rates = mean_strain_rates()
    params_qlv = pd.read_csv(BASE / "qlv_marmot_params.csv")
    params_bb = pd.read_csv(BASE / "bb_params.csv")

    targets = list(MATERIALS.keys()) if args.material == "all" else [args.material]

    for material in targets:
        print(f"\nProcessing build-up for {material}...")
        cfg = MATERIALS[material]
        r_qlv = params_qlv[
            (params_qlv["material"] == material) & (params_qlv["hyperelastic_base"] == "MooneyRivlin")
        ].iloc[0]
        _, qlv_plots = validate(
            material,
            cfg,
            "MooneyRivlin",
            r_qlv["C1"],
            r_qlv["C2"],
            r_qlv["C3"],
            [r_qlv["gamma1"], r_qlv["gamma2"], r_qlv["gamma3"]],
            [r_qlv["tau1"], r_qlv["tau2"], r_qlv["tau3"]],
        )

        r_bb = params_bb[
            (params_bb["material"] == material) & (params_bb["hyperelastic_base"] == "ArrudaBoyce")
        ].iloc[0]
        _, bb_plots = validate_bb(
            material,
            cfg,
            "ArrudaBoyce",
            (r_bb["A1"], r_bb["A2"], r_bb["A3"]),
            (r_bb["B1"], r_bb["B2"], r_bb["B3"]),
            r_bb["c1"],
            r_bb["c2"],
            r_bb["c3"],
        )

        generate_buildup_for_material(material, qlv_plots, bb_plots, strain_rates)


if __name__ == "__main__":
    main()
