"""Cross-material comparison of the digital-composite relaxation series
(A0V100, A25V75, A50V50), all sharing the same 6 mm x 2 mm specimen
cross-section.

All three materials now have three loading-rate groups (~0.017, ~0.167,
~1.67 mm/s; via the retest dataset's 2 mm groups for A50V50). For each
material/rate group (3 replicates each, except A0V100's "slow" group
which is [1, 3, 13] after excluding outlier test 2, and A50V50's "slow"
group which stays [4, 5, 6] after excluding outlier test 16 -- see each
material's plot_stress_time.py) this computes:
  - secant modulus at 0.2% strain (low-strain, common to all materials)
  - rate-sensitivity exponent m: the slope of a log(modulus) vs.
    log(rate) fit across ALL available rate groups for that material
    (2 points for A50V50, 3 for A0V100/A25V75 -- letting the 3-point
    materials show whether that relationship is actually log-linear
    rather than just interpolating between two)
  - percent stress relaxation from peak to t=590 s
  - the normalized relaxation curve shape (for visual comparison)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotstyle import colors, figsize_double, figsize_single

BASE = Path(__file__).parent

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

SECANT_STRAIN = 0.002
RELAXATION_TIME_S = 590

MATERIALS = {
    "A0V100": {
        "a_fraction": 0,
        "dir": BASE / "A0V100-relax" / "cleaned",
        "rate_groups": {
            "very slow": [10, 11, 12],
            "slow": [1, 3, 13],  # test 2 excluded as an outlier, replaced by 13
            "fast": [4, 5, 6],
        },
    },
    "A25V75": {
        "a_fraction": 25,
        "dir": BASE / "A25V75-relax" / "cleaned",
        "rate_groups": {
            "very slow": [10, 11, 12],
            "slow": [1, 2, 3, 9],
            "fast": [4, 5, 6],
        },
    },
    "A50V50": {
        "a_fraction": 50,
        "dir": BASE / "A50V50-relax-retest" / "cleaned",
        "rate_groups": {
            "very slow": [13, 14, 15],
            "slow": [4, 5, 6],  # test 16 excluded as an outlier (much stiffer)
            "fast": [7, 11, 12],
        },
    },
}

RATE_GROUP_STYLE = {
    "very slow": ("o", ":"),
    "slow": ("o", "-"),
    "fast": ("s", "--"),
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


def analyze_test(material, test_id):
    df = pd.read_csv(MATERIALS[material]["dir"] / f"A50V50-{test_id}_cleaned.csv"
                      if material == "A50V50" else
                      MATERIALS[material]["dir"] / f"{material}-{test_id}_cleaned.csv")
    stress = -df["force_N"] / AREA_MM2
    onset, ramp_end = find_ramp_bounds(df["position_mm"])

    t = df["time_s"].iloc[onset:ramp_end].to_numpy()
    pos = df["position_mm"].iloc[onset:ramp_end].to_numpy()
    rate, _ = np.polyfit(t, pos, 1)

    secant_modulus = np.interp(SECANT_STRAIN, df["strain"], stress) / SECANT_STRAIN
    peak_stress = stress.max()
    peak_idx = stress.to_numpy().argmax()
    t_peak = df["time_s"].iloc[peak_idx]

    t_final_idx = (df["time_s"] - RELAXATION_TIME_S).abs().idxmin()
    stress_final = stress.iloc[t_final_idx]
    pct_relaxation = (peak_stress - stress_final) / peak_stress * 100

    time_since_peak = df["time_s"].to_numpy() - t_peak
    norm_stress = stress.to_numpy() / peak_stress

    return {
        "material": material,
        "test": test_id,
        "rate_mm_s": rate,
        "secant_modulus_mpa": secant_modulus,
        "peak_stress_mpa": peak_stress,
        "stress_final_mpa": stress_final,
        "pct_relaxation": pct_relaxation,
        "time_since_peak": time_since_peak,
        "norm_stress": norm_stress,
    }


def main():
    rows = []
    curves = {}
    for material, cfg in MATERIALS.items():
        for rate_group, ids in cfg["rate_groups"].items():
            for test_id in ids:
                result = analyze_test(material, test_id)
                result["rate_group"] = rate_group
                curves[(material, rate_group, test_id)] = (
                    result.pop("time_since_peak"), result.pop("norm_stress")
                )
                rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(BASE / "material_comparison_raw.csv", index=False)

    summary = df.groupby(["material", "rate_group"]).agg(
        rate_mm_s=("rate_mm_s", "mean"),
        secant_modulus_mean=("secant_modulus_mpa", "mean"),
        secant_modulus_std=("secant_modulus_mpa", "std"),
        pct_relaxation_mean=("pct_relaxation", "mean"),
        pct_relaxation_std=("pct_relaxation", "std"),
    ).reset_index()

    rate_sensitivity = []
    for material, cfg in MATERIALS.items():
        s = summary[summary["material"] == material]
        log_rate = np.log(s["rate_mm_s"].to_numpy())
        log_mod = np.log(s["secant_modulus_mean"].to_numpy())
        m, _ = np.polyfit(log_rate, log_mod, 1)
        resid = log_mod - np.polyval((m, _), log_rate)
        rate_sensitivity.append({
            "material": material,
            "a_fraction": cfg["a_fraction"],
            "rate_sensitivity_m": m,
            "n_rate_points": len(s),
            "max_abs_log_residual": np.max(np.abs(resid)),
        })
    rate_sensitivity = pd.DataFrame(rate_sensitivity)
    summary = summary.merge(rate_sensitivity[["material", "a_fraction"]], on="material")
    summary.to_csv(BASE / "material_comparison_summary.csv", index=False)
    rate_sensitivity.to_csv(BASE / "material_rate_sensitivity.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(rate_sensitivity.to_string(index=False))

    # --- comparison figure ---
    fig, ((ax_modulus, ax_rate_sens), (ax_relax, ax_curves)) = plt.subplots(
        2, 2, figsize=(figsize_double[0] * 1.3, figsize_double[1] * 2.6)
    )

    material_order = list(MATERIALS.keys())
    a_fractions = [MATERIALS[m]["a_fraction"] for m in material_order]

    for rate_group in ["very slow", "slow", "fast"]:
        marker, ls = RATE_GROUP_STYLE[rate_group]
        s = summary[summary["rate_group"] == rate_group].set_index("material")
        s = s.reindex([m for m in material_order if m in s.index])
        xs = [MATERIALS[m]["a_fraction"] for m in s.index]
        ax_modulus.errorbar(
            xs, s["secant_modulus_mean"], yerr=s["secant_modulus_std"],
            marker=marker, markersize=4, linestyle=ls, color=colors[0],
            label=f"{rate_group} rate",
        )
        ax_relax.errorbar(
            xs, s["pct_relaxation_mean"], yerr=s["pct_relaxation_std"],
            marker=marker, markersize=4, linestyle=ls, color=colors[1],
            label=f"{rate_group} rate",
        )

    ax_modulus.set_xlabel("Agilus fraction (\\%)")
    ax_modulus.set_ylabel(r"Secant modulus at 0.2\% strain (MPa)")
    ax_modulus.set_xticks(a_fractions)
    ax_modulus.legend(fontsize=8)
    ax_modulus.grid(True, alpha=0.4)

    ax_rate_sens.plot(rate_sensitivity["a_fraction"], rate_sensitivity["rate_sensitivity_m"],
                       marker="o", color=colors[2])
    ax_rate_sens.set_xlabel("Agilus fraction (\\%)")
    ax_rate_sens.set_ylabel(r"Rate sensitivity $m = \Delta\ln E / \Delta\ln\dot{x}$")
    ax_rate_sens.set_xticks(a_fractions)
    ax_rate_sens.grid(True, alpha=0.4)

    ax_relax.set_xlabel("Agilus fraction (\\%)")
    ax_relax.set_ylabel(f"Stress relaxed by t={RELAXATION_TIME_S}s (\\%)")
    ax_relax.set_xticks(a_fractions)
    ax_relax.legend(fontsize=8)
    ax_relax.grid(True, alpha=0.4)

    for material, cfg in MATERIALS.items():
        color = colors[material_order.index(material) + 3]
        slow_ids = cfg["rate_groups"]["slow"]
        for test_id in slow_ids:
            t, norm = curves[(material, "slow", test_id)]
            mask = t > 0
            ax_curves.plot(t[mask], norm[mask], color=color, linewidth=0.7,
                            label=material if test_id == slow_ids[0] else None)
    ax_curves.set_xscale("log")
    ax_curves.set_xlabel("Time since peak stress (s)")
    ax_curves.set_ylabel("Normalized stress (stress / peak stress)")
    ax_curves.legend(fontsize=8)
    ax_curves.set_title("Relaxation shape comparison (slow rate)", fontsize=9)
    ax_curves.grid(True, alpha=0.4)

    fig.tight_layout()
    out_path = BASE / "material_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nsaved {out_path} and {out_path.with_suffix('.png')}")

    # --- relaxation shape comparison, fast rate ---
    fig2, ax_curves_fast = plt.subplots(figsize=figsize_single)
    for material, cfg in MATERIALS.items():
        color = colors[material_order.index(material) + 3]
        fast_ids = cfg["rate_groups"]["fast"]
        for test_id in fast_ids:
            t, norm = curves[(material, "fast", test_id)]
            mask = t > 0
            ax_curves_fast.plot(t[mask], norm[mask], color=color, linewidth=0.7,
                                 label=material if test_id == fast_ids[0] else None)
    ax_curves_fast.set_xscale("log")
    ax_curves_fast.set_xlabel("Time since peak stress (s)")
    ax_curves_fast.set_ylabel("Normalized stress (stress / peak stress)")
    ax_curves_fast.legend(fontsize=8)
    ax_curves_fast.set_title("Relaxation shape comparison (fast rate)", fontsize=9)
    ax_curves_fast.grid(True, alpha=0.4)

    fig2.tight_layout()
    out_path2 = BASE / "material_relaxation_shape_fast.pdf"
    fig2.savefig(out_path2, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path2} and {out_path2.with_suffix('.png')}")

    # --- relaxation shape comparison: slow vs fast, per material ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(figsize_double[0] * 1.6, figsize_double[1]), sharey=True)
    for ax, (material, cfg) in zip(axes3, MATERIALS.items()):
        color = colors[material_order.index(material) + 3]
        for rate_group in ["slow", "fast"]:
            ids = cfg["rate_groups"][rate_group]
            _, ls = RATE_GROUP_STYLE[rate_group]
            for test_id in ids:
                t, norm = curves[(material, rate_group, test_id)]
                mask = t > 0
                ax.plot(t[mask], norm[mask], color=color, linestyle=ls, linewidth=0.7,
                        label=f"{rate_group} rate" if test_id == ids[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel("Time since peak stress (s)")
        ax.set_title(material, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    axes3[0].set_ylabel("Normalized stress (stress / peak stress)")

    fig3.tight_layout()
    out_path3 = BASE / "material_relaxation_shape_rate_comparison.pdf"
    fig3.savefig(out_path3, bbox_inches="tight")
    fig3.savefig(out_path3.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path3} and {out_path3.with_suffix('.png')}")

    # --- strain-rate dependency: modulus vs rate, per material, all points ---
    fig4, axes4 = plt.subplots(1, 3, figsize=(figsize_double[0] * 1.6, figsize_double[1]), sharey=False)
    for ax, (material, cfg) in zip(axes4, MATERIALS.items()):
        s = summary[summary["material"] == material].sort_values("rate_mm_s")
        ax.errorbar(s["rate_mm_s"], s["secant_modulus_mean"], yerr=s["secant_modulus_std"],
                    marker="o", markersize=4, linestyle="none", color=colors[material_order.index(material) + 3],
                    label="measured", zorder=5)

        m = rate_sensitivity.loc[rate_sensitivity["material"] == material, "rate_sensitivity_m"].iloc[0]
        log_rate = np.log(s["rate_mm_s"].to_numpy())
        log_mod = np.log(s["secant_modulus_mean"].to_numpy())
        intercept = np.polyfit(log_rate, log_mod, 1)[1]
        log10_rate = np.log10(s["rate_mm_s"].to_numpy())
        rate_grid = np.logspace(log10_rate.min() - 0.3, log10_rate.max() + 0.3, 50)
        ax.plot(rate_grid, np.exp(intercept) * rate_grid**m, color="grey", linewidth=1,
                linestyle="--", label=f"fit (m={m:.3f})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Loading rate (mm/s)")
        ax.set_title(f"{material} (n={len(s)} rate points)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    axes4[0].set_ylabel("Secant modulus at 0.2\\% strain (MPa)")

    fig4.tight_layout()
    out_path4 = BASE / "material_rate_dependency.pdf"
    fig4.savefig(out_path4, bbox_inches="tight")
    fig4.savefig(out_path4.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path4} and {out_path4.with_suffix('.png')}")


if __name__ == "__main__":
    main()
