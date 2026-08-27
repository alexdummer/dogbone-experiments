"""Cross-material comparison of the digital-composite relaxation series
(A0V100, A25V75, A50V50) at matched low/high loading rate (~10x), all
sharing the same 6 mm x 2 mm specimen cross-section.

For each material/rate group (3 replicates each) this computes:
  - secant modulus at 0.2% strain (low-strain, common to all materials)
  - rate-sensitivity exponent m = ln(E_fast/E_slow) / ln(rate_fast/rate_slow)
  - percent stress relaxation from peak to t=590 s
  - the normalized relaxation curve shape (for visual comparison)

A50V50 uses the retest dataset's 2 mm groups (tests 4-6 low rate,
7/11/12 high rate) since that is the only A50V50 subset with matched
target displacement and full 3x replication at both rates, consistent
with how A0V100 and A25V75 are structured.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotstyle import colors, figsize_double

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
        "slow": [1, 2, 3],
        "fast": [4, 5, 6],
    },
    "A25V75": {
        "a_fraction": 25,
        "dir": BASE / "A25V75-relax" / "cleaned",
        "slow": [1, 2, 3],
        "fast": [4, 5, 6],
    },
    "A50V50": {
        "a_fraction": 50,
        "dir": BASE / "A50V50-relax-retest" / "cleaned",
        "slow": [4, 5, 6],
        "fast": [7, 11, 12],
    },
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
        for rate_group, ids in [("slow", cfg["slow"]), ("fast", cfg["fast"])]:
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
        e_slow = s[s["rate_group"] == "slow"]["secant_modulus_mean"].iloc[0]
        e_fast = s[s["rate_group"] == "fast"]["secant_modulus_mean"].iloc[0]
        r_slow = s[s["rate_group"] == "slow"]["rate_mm_s"].iloc[0]
        r_fast = s[s["rate_group"] == "fast"]["rate_mm_s"].iloc[0]
        m = np.log(e_fast / e_slow) / np.log(r_fast / r_slow)
        rate_sensitivity.append({
            "material": material,
            "a_fraction": cfg["a_fraction"],
            "rate_sensitivity_m": m,
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

    for rate_group, marker, ls in [("slow", "o", "-"), ("fast", "s", "--")]:
        s = summary[summary["rate_group"] == rate_group].set_index("material").loc[material_order]
        ax_modulus.errorbar(
            a_fractions, s["secant_modulus_mean"], yerr=s["secant_modulus_std"],
            marker=marker, linestyle=ls, color=colors[0],
            label=f"{rate_group} rate",
        )
        ax_relax.errorbar(
            a_fractions, s["pct_relaxation_mean"], yerr=s["pct_relaxation_std"],
            marker=marker, linestyle=ls, color=colors[1],
            label=f"{rate_group} rate",
        )

    ax_modulus.set_xlabel("A fraction (\\%)")
    ax_modulus.set_ylabel(r"Secant modulus at 0.2\% strain (MPa)")
    ax_modulus.set_xticks(a_fractions)
    ax_modulus.legend(fontsize=8)

    ax_rate_sens.plot(rate_sensitivity["a_fraction"], rate_sensitivity["rate_sensitivity_m"],
                       marker="o", color=colors[2])
    ax_rate_sens.set_xlabel("A fraction (\\%)")
    ax_rate_sens.set_ylabel(r"Rate sensitivity $m = \Delta\ln E / \Delta\ln\dot{x}$")
    ax_rate_sens.set_xticks(a_fractions)

    ax_relax.set_xlabel("A fraction (\\%)")
    ax_relax.set_ylabel(f"Stress relaxed by t={RELAXATION_TIME_S}s (\\%)")
    ax_relax.set_xticks(a_fractions)
    ax_relax.legend(fontsize=8)

    for material, cfg in MATERIALS.items():
        color = colors[material_order.index(material) + 3]
        for test_id in cfg["slow"]:
            t, norm = curves[(material, "slow", test_id)]
            mask = t > 0
            ax_curves.plot(t[mask], norm[mask], color=color, linewidth=0.7,
                            label=material if test_id == cfg["slow"][0] else None)
    ax_curves.set_xscale("log")
    ax_curves.set_xlabel("Time since peak stress (s)")
    ax_curves.set_ylabel("Normalized stress (stress / peak stress)")
    ax_curves.legend(fontsize=8)
    ax_curves.set_title("Relaxation shape comparison (slow rate)", fontsize=9)

    fig.tight_layout()
    out_path = BASE / "material_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nsaved {out_path} and {out_path.with_suffix('.png')}")

    # --- relaxation shape comparison, fast rate ---
    fig2, ax_curves_fast = plt.subplots(figsize=figsize_double)
    for material, cfg in MATERIALS.items():
        color = colors[material_order.index(material) + 3]
        for test_id in cfg["fast"]:
            t, norm = curves[(material, "fast", test_id)]
            mask = t > 0
            ax_curves_fast.plot(t[mask], norm[mask], color=color, linewidth=0.7,
                                 label=material if test_id == cfg["fast"][0] else None)
    ax_curves_fast.set_xscale("log")
    ax_curves_fast.set_xlabel("Time since peak stress (s)")
    ax_curves_fast.set_ylabel("Normalized stress (stress / peak stress)")
    ax_curves_fast.legend(fontsize=8)
    ax_curves_fast.set_title("Relaxation shape comparison (fast rate)", fontsize=9)

    fig2.tight_layout()
    out_path2 = BASE / "material_relaxation_shape_fast.pdf"
    fig2.savefig(out_path2, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path2} and {out_path2.with_suffix('.png')}")

    # --- relaxation shape comparison: slow vs fast, per material ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(figsize_double[0] * 1.6, figsize_double[1]), sharey=True)
    for ax, (material, cfg) in zip(axes3, MATERIALS.items()):
        color = colors[material_order.index(material) + 3]
        for rate_group, ls in [("slow", "-"), ("fast", "--")]:
            for test_id in cfg[rate_group]:
                t, norm = curves[(material, rate_group, test_id)]
                mask = t > 0
                ax.plot(t[mask], norm[mask], color=color, linestyle=ls, linewidth=0.7,
                        label=f"{rate_group} rate" if test_id == cfg[rate_group][0] else None)
        ax.set_xscale("log")
        ax.set_xlabel("Time since peak stress (s)")
        ax.set_title(material, fontsize=9)
        ax.legend(fontsize=7)
    axes3[0].set_ylabel("Normalized stress (stress / peak stress)")

    fig3.tight_layout()
    out_path3 = BASE / "material_relaxation_shape_rate_comparison.pdf"
    fig3.savefig(out_path3, bbox_inches="tight")
    fig3.savefig(out_path3.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path3} and {out_path3.with_suffix('.png')}")


if __name__ == "__main__":
    main()
