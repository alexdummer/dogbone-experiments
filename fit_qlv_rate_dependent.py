"""Option 1 diagnostic: does letting the Prony series depend on loading
rate (instead of one curve averaged across all rate groups) close the
QLV model's validation gap -- especially for A50V50, where the separable
(rate-independent) model showed the worst error?

Method: fit a separate 3-term Prony curve to each rate group's own
relaxation-hold data (not one averaged curve), see how g_i/tau_i trend
with log(rate), then build a log-rate interpolant. Reuses the Ogden fit,
loading, and recursive integration machinery from fit_qlv_model.py
unchanged -- only the Prony kernel becomes rate-dependent.

Held-out check: fit only "very slow" and "fast", interpolate to predict
the "slow" (middle) rate group, and compare that genuinely-generalized
prediction against the single-averaged-Prony model's prediction for the
same test -- this is the fair test of whether rate-dependence actually
helps, since fitting-then-predicting the SAME rate group would trivially
look good regardless of whether the approach generalizes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize

from fit_qlv_model import (
    BASE, MATERIALS, AREA_MM2, TIME_GRID, load, find_ramp_bounds,
    prony, fit_ogden, ogden_nominal_stress, predict_stress, fit_prony,
)
from plotstyle import colors, figsize_double


def group_rate(cfg, material, ids):
    """Mean actuator rate (mm/s) for a rate group, from its own ramp segments."""
    rates = []
    for test_id in ids:
        df = load(cfg, material, test_id)
        onset, ramp_end = find_ramp_bounds(df["position_mm"])
        t = df["time_s"].to_numpy()[onset:ramp_end]
        pos = df["position_mm"].to_numpy()[onset:ramp_end]
        rate, _ = np.polyfit(t, pos, 1)
        rates.append(rate)
    return float(np.mean(rates))


def fit_prony_group(material, cfg, ids):
    """Fit a 3-term Prony curve to ONE rate group's own data (only 3-4
    curves, vs ~9-10 when averaging across all groups in fit_prony) --
    explicitly constrained so g1+g2+g3 <= 1 (g_inf >= 0), since with this
    little data curve_fit's independent [0,1] box bounds on each g_i can
    converge to a solution where the SUM exceeds 1, giving an unphysical
    negative g_inf."""
    curves = []
    for test_id in ids:
        df = load(cfg, material, test_id)
        onset, ramp_end = find_ramp_bounds(df["position_mm"])
        t_end = df["time_s"].iloc[ramp_end]
        s_end = df["stress"].iloc[ramp_end]
        t_rel = df["time_s"].to_numpy() - t_end
        mask = t_rel > 0
        curves.append(np.interp(TIME_GRID, t_rel[mask], df["stress"].to_numpy()[mask] / s_end))
    avg_curve = np.mean(curves, axis=0)

    def objective(x):
        g1, tau1, g2, tau2, g3, tau3 = x
        pred = prony(TIME_GRID, g1, tau1, g2, tau2, g3, tau3)
        return np.sum((pred - avg_curve) ** 2)

    x0 = [0.2, 1.0, 0.2, 10.0, 0.2, 100.0]
    bounds = [(0, 1), (1e-3, 1e4), (0, 1), (1e-3, 1e4), (0, 1), (1e-3, 1e4)]
    constraints = [{"type": "ineq", "fun": lambda x: 1 - (x[0] + x[2] + x[4])}]
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-12})
    g1, tau1, g2, tau2, g3, tau3 = result.x
    return {"g1": g1, "tau1": tau1, "g2": g2, "tau2": tau2, "g3": g3, "tau3": tau3,
            "g_inf": 1 - g1 - g2 - g3}


def build_interpolant(rate_prony_pairs):
    """Log-rate linear interpolant (with edge-value extrapolation) for
    each Prony parameter, from a list of (rate, params_dict)."""
    rates = np.array([r for r, _ in rate_prony_pairs])
    log_rates = np.log(rates)
    order = np.argsort(log_rates)
    log_rates = log_rates[order]
    keys = ["g1", "tau1", "g2", "tau2", "g3", "tau3"]
    series = {k: np.array([p[k] for _, p in rate_prony_pairs])[order] for k in keys}

    def interp(rate):
        lr = np.log(rate)
        out = {k: float(np.interp(lr, log_rates, series[k])) for k in keys}
        out["g_inf"] = 1 - out["g1"] - out["g2"] - out["g3"]
        return out

    return interp


def main():
    all_ogden = {}
    per_group_prony = {}
    group_rates = {}

    for material, cfg in MATERIALS.items():
        all_ogden[material] = fit_ogden(material, cfg)
        per_group_prony[material] = {}
        group_rates[material] = {}
        for rate_group, ids in cfg["rate_groups"].items():
            per_group_prony[material][rate_group] = fit_prony_group(material, cfg, ids)
            group_rates[material][rate_group] = group_rate(cfg, material, ids)

    print("Per-rate-group Prony parameters (trend check):")
    rows = []
    for material in MATERIALS:
        for rate_group in ["very slow", "slow", "fast"]:
            p = per_group_prony[material][rate_group]
            r = group_rates[material][rate_group]
            rows.append({"material": material, "rate_group": rate_group, "rate_mm_s": r, **p})
    trend_df = pd.DataFrame(rows)
    trend_df.to_csv(BASE / "qlv_rate_dependent_prony.csv", index=False)
    print(trend_df.to_string(index=False))

    # --- held-out check: fit only very-slow + fast, interpolate to predict "slow" ---
    print("\nHeld-out validation: predict 'slow' from very-slow+fast interpolation only")
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=False)
    summary_rows = []

    for ax, (material, cfg) in zip(axes, MATERIALS.items()):
        held_out_pairs = [
            (group_rates[material]["very slow"], per_group_prony[material]["very slow"]),
            (group_rates[material]["fast"], per_group_prony[material]["fast"]),
        ]
        interp = build_interpolant(held_out_pairs)

        test_id = cfg["rate_groups"]["slow"][0]
        df = load(cfg, material, test_id)
        onset, ramp_end = find_ramp_bounds(df["position_mm"])
        t = df["time_s"].to_numpy()[onset:ramp_end]
        pos = df["position_mm"].to_numpy()[onset:ramp_end]
        this_rate, _ = np.polyfit(t, pos, 1)

        rate_dependent_prony = interp(this_rate)
        pred_ratedep = predict_stress(df["time_s"].to_numpy(), df["strain"].to_numpy(),
                                       all_ogden[material], rate_dependent_prony)

        # baseline: single Prony curve averaged over ALL rate groups (original model)
        baseline_prony = fit_prony(material, cfg)
        pred_baseline = predict_stress(df["time_s"].to_numpy(), df["strain"].to_numpy(),
                                        all_ogden[material], baseline_prony)

        measured = df["stress"].to_numpy()
        peak = measured.max()
        rmse_ratedep = np.sqrt(np.mean((pred_ratedep - measured) ** 2))
        rmse_baseline = np.sqrt(np.mean((pred_baseline - measured) ** 2))
        summary_rows.append({
            "material": material, "held_out_rate_mm_s": this_rate,
            "rmse_baseline_pct": 100 * rmse_baseline / peak,
            "rmse_rate_dependent_pct": 100 * rmse_ratedep / peak,
        })

        n = len(df)
        step = max(1, n // 300)
        t_all = df["time_s"].to_numpy()
        ax.plot(t_all, measured, color=colors[3], linewidth=2.2, alpha=0.4, label="measured")
        ax.plot(t_all[::step], pred_baseline[::step], color=colors[4], linewidth=1.0,
                linestyle="--", marker="s", markersize=2.5, label="baseline (rate-independent)")
        ax.plot(t_all[::step], pred_ratedep[::step], color=colors[5], linewidth=1.0,
                linestyle="--", marker="o", markersize=2.5, label="rate-dependent (held-out)")
        ax.set_xlabel("Time (s)")
        ax.set_title(material, fontsize=9)
        ax.legend(fontsize=6)

    axes[0].set_ylabel("Stress (MPa)")
    fig.suptitle("Held-out 'slow' prediction: rate-dependent vs. rate-independent Prony", fontsize=10)
    fig.tight_layout()
    out_path = BASE / "qlv_rate_dependent_validation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(BASE / "qlv_rate_dependent_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
