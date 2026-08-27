"""Test classical two-phase mixture theories (Voigt, Reuss, and the
log/geometric-mean rule common for polymer blends) against the measured
secant modulus vs. A-fraction data.

Since we only have 3 compositions (0, 25, 50% A) and no pure-A (A100V0)
specimen, each theory is calibrated on the two compositions we DO have as
end-members (0% and 50% A) and then used to predict the held-out middle
point (25% A) -- a genuine extrapolate-then-validate test, not a fit to
all three points at once.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotstyle import colors, figsize_double

BASE = Path(__file__).parent

THEORIES = ["Voigt", "Reuss", "Log"]


def calibrate_and_predict(E_V, E_50, E_25_actual):
    results = {}

    # Voigt (arithmetic mean, isostrain/parallel): E(f) = (1-f) E_V + f E_A
    E_A = 2 * E_50 - E_V
    E_25 = 0.75 * E_V + 0.25 * E_A
    results["Voigt"] = (E_A, E_25)

    # Reuss (harmonic mean, isostress/series): 1/E(f) = (1-f)/E_V + f/E_A
    E_A = 1 / (2 / E_50 - 1 / E_V)
    E_25 = 1 / (0.75 / E_V + 0.25 / E_A)
    results["Reuss"] = (E_A, E_25)

    # log rule (geometric mean): ln E(f) = (1-f) ln E_V + f ln E_A
    E_A = E_50**2 / E_V
    E_25 = E_V**0.75 * E_A**0.25
    results["Log"] = (E_A, E_25)

    rows = []
    for name, (E_A, E_25_pred) in results.items():
        rows.append({
            "theory": name,
            "implied_E_A_mpa": E_A,
            "predicted_E_25_mpa": E_25_pred,
            "actual_E_25_mpa": E_25_actual,
            "pct_error": (E_25_pred - E_25_actual) / E_25_actual * 100,
        })
    return pd.DataFrame(rows)


def curve(theory, E_V, E_A, f):
    if theory == "Voigt":
        return (1 - f) * E_V + f * E_A
    if theory == "Reuss":
        return 1 / ((1 - f) / E_V + f / E_A)
    if theory == "Log":
        return E_V ** (1 - f) * E_A**f
    raise ValueError(theory)


def main():
    summary = pd.read_csv(BASE / "material_comparison_summary.csv")

    all_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(figsize_double[0] * 1.5, figsize_double[1]), sharey=True)

    for ax, rate_group in zip(axes, ["slow", "fast"]):
        s = summary[summary["rate_group"] == rate_group].set_index("a_fraction")
        E_V = s.loc[0, "secant_modulus_mean"]
        E_50 = s.loc[50, "secant_modulus_mean"]
        E_25_actual = s.loc[25, "secant_modulus_mean"]

        result = calibrate_and_predict(E_V, E_50, E_25_actual)
        result.insert(0, "rate_group", rate_group)
        all_rows.append(result)

        f_grid = np.linspace(0, 1, 200)
        for i, theory in enumerate(THEORIES):
            E_A = result.set_index("theory").loc[theory, "implied_E_A_mpa"]
            y = curve(theory, E_V, E_A, f_grid)
            valid = y > 0
            ax.plot(f_grid[valid] * 100, y[valid], color=colors[i], linestyle="-",
                    label=f"{theory} (implied $E_A$={E_A:.0f} MPa)")

        measured = s.reset_index()
        ax.errorbar(measured["a_fraction"], measured["secant_modulus_mean"],
                     yerr=measured["secant_modulus_std"], fmt="ko", capsize=3,
                     label="measured", zorder=5)

        ax.set_xlabel("A fraction (\\%)")
        ax.set_title(f"{rate_group} rate", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.set_xlim(-2, 102)

    axes[0].set_ylabel("Secant modulus at 0.2\\% strain (MPa)")

    fig.tight_layout()
    out_path = BASE / "mixture_theory_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(BASE / "mixture_theory_comparison.csv", index=False)
    print()
    print(full.to_string(index=False))


if __name__ == "__main__":
    main()
