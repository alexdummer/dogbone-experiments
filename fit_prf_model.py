"""Nonlinear Parallel Rheological Framework (PRF) model: a 2-network
alternative to the separable QLV (Ogden+Prony) model, aimed at capturing
the rate-dependent relaxation-shape crossover that QLV (and rate-dependent
Prony interpolation, which made things worse -- see fit_qlv_rate_dependent.py)
could not.

Architecture (matches Abaqus's built-in *HYPERELASTIC + *VISCOELASTIC,
PRF-style parallel network, using the standard Bergstrom-Boyce-type flow
law so parameters map onto that Abaqus material model):

  Network A (equilibrium): 1-term Ogden, stress sigma_A(lambda).
  Network B (flow): 1-term Ogden acting on an ELASTIC stretch lambda_e,
    in series with a nonlinear dashpot (viscous stretch lambda_v):
        lambda = lambda_e * lambda_v
        dlambda_v/dt = A_flow * lambda_v * sigma_B(lambda_e) ** m_flow
  Total predicted stress: sigma(t) = sigma_A(lambda(t)) + sigma_B(lambda_e(t))

Unlike QLV's linear Prony convolution, this flow law is genuinely
nonlinear and rate-dependent by construction: at fast rates lambda_v has
no time to creep (network B carries an "unrelaxed" load), at slow rates
lambda_v creeps substantially even during the ramp itself. All rate
groups are fit SIMULTANEOUSLY (not sequentially/independently per group),
which avoids the term-relabeling non-uniqueness that broke the
rate-dependent-Prony attempt.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from fit_qlv_model import BASE, MATERIALS, load, find_ramp_bounds, predict_stress, fit_ogden, fit_prony
from plotstyle import colors, figsize_double


def ogden1_stress(stretch, mu, alpha):
    return (2 * mu / alpha) * (stretch ** (alpha - 1) - stretch ** (-alpha / 2 - 1))


def integrate_prf(time_s, strain, muA, alphaA, muB, alphaB, A_flow, m_flow):
    """Integrate the PRF ODE for an imposed strain(t) history; returns
    predicted total stress at each input time point. lambda_v and sigma_B
    are clamped to a safe physical range throughout (not just inside the
    RHS) so an optimizer's exploratory (possibly unstable) parameter
    guesses can't return inf/nan and poison the finite-difference
    gradient -- a bad-but-finite residual is fine, a NaN is not."""
    stretch_interp = lambda t: 1 + np.interp(t, time_s, strain)  # noqa: E731
    LAM_V_MIN, LAM_V_MAX = 0.2, 5.0
    SIGMA_B_MAX = 500.0

    def rhs(t, y):
        lam_v = min(max(y[0], LAM_V_MIN), LAM_V_MAX)
        lam = stretch_interp(t)
        lam_e = np.clip(lam / lam_v, 0.5, 10.0)
        sigma_B = np.clip(ogden1_stress(lam_e, muB, alphaB), 0, SIGMA_B_MAX)
        rate = A_flow * lam_v * sigma_B ** m_flow
        return [rate]

    try:
        sol = solve_ivp(rhs, (time_s[0], time_s[-1]), [1.0], t_eval=time_s, method="LSODA",
                         rtol=1e-9, atol=1e-11, max_step=(time_s[-1] - time_s[0]) / 30)
        lam_v = np.clip(sol.y[0], LAM_V_MIN, LAM_V_MAX)
        if not sol.success or len(lam_v) != len(time_s):
            raise RuntimeError("integration failed")
    except Exception:
        lam_v = np.ones_like(time_s)

    lam = 1 + strain
    lam_e = np.clip(lam / lam_v, 0.5, 10.0)
    sigma_A = np.clip(ogden1_stress(lam, muA, alphaA), -SIGMA_B_MAX, SIGMA_B_MAX)
    sigma_B = np.clip(ogden1_stress(lam_e, muB, alphaB), -SIGMA_B_MAX, SIGMA_B_MAX)
    total = sigma_A + sigma_B
    total = np.nan_to_num(total, nan=SIGMA_B_MAX, posinf=SIGMA_B_MAX, neginf=-SIGMA_B_MAX)
    return total, lam_v


def downsample_for_fit(df, n_ramp=40, n_hold=60):
    """Log-spaced-in-hold, linear-in-ramp downsampling so the optimizer's
    repeated ODE solves stay fast without losing the ramp shape or the
    long-time relaxation tail."""
    onset, ramp_end = find_ramp_bounds(df["position_mm"])
    t = df["time_s"].to_numpy()
    ramp_idx = np.linspace(0, ramp_end, n_ramp, dtype=int)
    t_end = t[ramp_end]
    t_hold = np.logspace(np.log10(max(t[ramp_end + 1] - t_end, 1e-3)),
                          np.log10(t[-1] - t_end), n_hold) + t_end
    hold_idx = np.searchsorted(t, t_hold)
    hold_idx = np.clip(hold_idx, ramp_end + 1, len(t) - 1)
    idx = np.unique(np.concatenate([ramp_idx, hold_idx]))
    return df["time_s"].to_numpy()[idx], df["strain"].to_numpy()[idx], df["stress"].to_numpy()[idx]


def fit_prf_joint(material, cfg, verbose=True):
    """Fit network A, network B, and flow-law params SIMULTANEOUSLY against
    one representative test per rate group (very slow/slow/fast), residuals
    normalized by each test's own peak stress so no rate group dominates
    just because it reaches a higher absolute stress."""
    test_data = {}
    for rate_group, ids in cfg["rate_groups"].items():
        df = load(cfg, material, ids[0])
        t, strain, stress = downsample_for_fit(df)
        test_data[rate_group] = (t, strain, stress, stress.max())

    # initial guess informed by the QLV fit already on hand
    qlv_ogden = fit_ogden(material, cfg)
    qlv_prony = fit_prony(material, cfg)
    total_mu = qlv_ogden["mu1"] + qlv_ogden["mu2"] if qlv_ogden["alpha1"] > 0 else qlv_ogden["mu2"]
    muA0 = max(total_mu * qlv_prony["g_inf"], 1e-3)
    muB0 = max(total_mu * (1 - qlv_prony["g_inf"]), 1e-3)
    tau_mid = qlv_prony["tau2"]

    x0 = [muA0, 2.0, muB0, 2.0, 1.0 / tau_mid, 1.0]
    lb = [1e-4, 0.5, 1e-4, 0.5, 1e-5, 0.3]
    ub = [1e4, 8, 1e4, 8, 5.0, 3]

    def residuals(x):
        muA, alphaA, muB, alphaB, A_flow, m_flow = x
        out = []
        for rate_group, (t, strain, stress, peak) in test_data.items():
            pred, _ = integrate_prf(t, strain, muA, alphaA, muB, alphaB, A_flow, m_flow)
            out.append((pred - stress) / peak)
        return np.concatenate(out)

    result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", verbose=2 if verbose else 0,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, diff_step=1e-3, max_nfev=400)
    muA, alphaA, muB, alphaB, A_flow, m_flow = result.x

    params = {"muA": muA, "alphaA": alphaA, "muB": muB, "alphaB": alphaB,
              "A_flow": A_flow, "m_flow": m_flow, "cost": result.cost, "nfev": result.nfev}
    return params, test_data


def validate_prf(material, cfg, params):
    rows = []
    for rate_group, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, material, test_id)
        pred, _ = integrate_prf(df["time_s"].to_numpy(), df["strain"].to_numpy(),
                                 params["muA"], params["alphaA"], params["muB"], params["alphaB"],
                                 params["A_flow"], params["m_flow"])
        measured = df["stress"].to_numpy()
        peak = measured.max()
        rmse = np.sqrt(np.mean((pred - measured) ** 2))
        rows.append({"material": material, "rate_group": rate_group, "test_id": test_id,
                     "rmse_mpa": rmse, "rmse_pct_of_peak": 100 * rmse / peak})
    return pd.DataFrame(rows), df["time_s"].to_numpy(), pred


def export_abaqus_prf(material, params):
    """Document the fitted PRF parameters in an Abaqus-oriented material
    card. Unlike the simple *VISCOELASTIC, TIME=PRONY card used for QLV,
    Abaqus's actual nonlinear parallel-network (PRF / Bergstrom-Boyce flow)
    keyword syntax is version-specific (e.g. *VISCOELASTIC, NONLINEAR vs.
    a UMAT/UHYPER-based implementation in older versions) -- this file is a
    parameter reference, not a drop-in .inp, and should be checked against
    the target Abaqus version's documentation before use."""
    lines = [
        f"** PRF (2-network nonlinear viscoelastic) parameters for {material}",
        "** Network A (equilibrium), 1-term Ogden:",
        "*HYPERELASTIC, OGDEN, N=1",
        f"{params['muA']:.6g}, {params['alphaA']:.6g}, 0.0",
        "**",
        "** Network B (flow), 1-term Ogden acting on elastic stretch lambda_e:",
        "*HYPERELASTIC, OGDEN, N=1",
        f"{params['muB']:.6g}, {params['alphaB']:.6g}, 0.0",
        "**",
        "** Flow law: dlambda_v/dt = A_flow * lambda_v * sigma_B(lambda_e)**m_flow",
        f"** A_flow = {params['A_flow']:.6g}   m_flow = {params['m_flow']:.6g}",
        "** Map onto Abaqus's *VISCOELASTIC, NONLINEAR parallel-network flow",
        "** parameters (exact keyword/parameter names are version-dependent --",
        "** verify against Abaqus Analysis User's Guide for the target release).",
        "**",
        "** D_p = 0 (fully incompressible) assumed for both networks: uniaxial",
        "** tension data alone cannot identify compressibility.",
    ]
    out_path = BASE / f"prf_abaqus_{material}.inp"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def sanity_check():
    """Hand-picked-parameter check on a synthetic ramp+hold: a faster ramp
    should reach a higher peak (network B hasn't had time to creep), and
    both rates should converge to the same long-time equilibrium (network
    A alone) -- confirms the forward model behaves as the theory predicts
    before trusting it to fit real data."""
    t_ramp = np.linspace(0, 5, 200)
    strain_ramp = 0.05 * (t_ramp / 5)
    t_hold = np.linspace(5, 300, 200)[1:]
    strain_hold = np.full_like(t_hold, 0.05)
    t = np.concatenate([t_ramp, t_hold])
    strain = np.concatenate([strain_ramp, strain_hold])

    pred_fast, lam_v_fast = integrate_prf(t, strain, muA=5, alphaA=2, muB=15, alphaB=2,
                                            A_flow=0.05, m_flow=1.0)

    t_ramp_slow = np.linspace(0, 50, 200)
    strain_ramp_slow = 0.05 * (t_ramp_slow / 50)
    t_hold_slow = np.linspace(50, 300, 200)[1:]
    strain_hold_slow = np.full_like(t_hold_slow, 0.05)
    t_slow = np.concatenate([t_ramp_slow, t_hold_slow])
    strain_slow = np.concatenate([strain_ramp_slow, strain_hold_slow])
    pred_slow, lam_v_slow = integrate_prf(t_slow, strain_slow, muA=5, alphaA=2, muB=15, alphaB=2,
                                            A_flow=0.05, m_flow=1.0)

    fig, ax = plt.subplots(figsize=figsize_double)
    ax.plot(t, pred_fast, label="fast ramp (5s)")
    ax.plot(t_slow, pred_slow, label="slow ramp (50s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stress (sanity-check units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / "prf_sanity_check.png", dpi=200, bbox_inches="tight")
    print("saved prf_sanity_check.png")
    print(f"fast: peak={pred_fast.max():.3f}, final={pred_fast[-1]:.3f}")
    print(f"slow: peak={pred_slow.max():.3f}, final={pred_slow[-1]:.3f}")
    print("(expect: fast has a higher, sharper peak; slow shows more creep during ramp -> lower peak, "
          "and both should converge toward similar long-time equilibrium stress = network A alone)")
    print(f"equilibrium (network A only) at stretch=1.05: {ogden1_stress(1.05, 5, 2):.3f}")


def main():
    all_prf = {}
    all_qlv_ogden = {}
    all_qlv_prony = {}
    validation_rows = []
    qlv_validation_rows = []
    plot_data = {}

    for material, cfg in MATERIALS.items():
        print(f"\n=== fitting PRF: {material} ===")
        params, _ = fit_prf_joint(material, cfg, verbose=False)
        all_prf[material] = params
        all_qlv_ogden[material] = fit_ogden(material, cfg)
        all_qlv_prony[material] = fit_prony(material, cfg)
        print({k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()})

        prf_val, _, _ = validate_prf(material, cfg, params)
        validation_rows.append(prf_val)

        for rate_group, ids in cfg["rate_groups"].items():
            test_id = ids[0]
            df = load(cfg, material, test_id)
            time_s = df["time_s"].to_numpy()
            strain = df["strain"].to_numpy()
            measured = df["stress"].to_numpy()

            pred_prf, _ = integrate_prf(time_s, strain, params["muA"], params["alphaA"],
                                         params["muB"], params["alphaB"],
                                         params["A_flow"], params["m_flow"])
            pred_qlv = predict_stress(time_s, strain, all_qlv_ogden[material], all_qlv_prony[material])

            peak = measured.max()
            rmse_qlv = np.sqrt(np.mean((pred_qlv - measured) ** 2))
            qlv_validation_rows.append({"material": material, "rate_group": rate_group,
                                         "test_id": test_id, "rmse_mpa": rmse_qlv,
                                         "rmse_pct_of_peak": 100 * rmse_qlv / peak})
            plot_data[(material, rate_group)] = (time_s, measured, pred_qlv, pred_prf)

        export_path = export_abaqus_prf(material, params)
        print(f"saved {export_path}")

    prf_validation = pd.concat(validation_rows, ignore_index=True)
    qlv_validation = pd.DataFrame(qlv_validation_rows)

    params_df = pd.DataFrame(all_prf).T
    params_df.index.name = "material"
    params_df.to_csv(BASE / "prf_params.csv")
    print(f"\nsaved {BASE / 'prf_params.csv'}")

    # --- combined validation figure: measured vs QLV vs PRF, all materials/rates ---
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=False)
    for ax, material in zip(axes, MATERIALS):
        for i, rate_group in enumerate(["very slow", "slow", "fast"]):
            t, measured, pred_qlv, pred_prf = plot_data[(material, rate_group)]
            n = len(t)
            step = max(1, n // 250)
            color = colors[i]
            ax.plot(t, measured, color=color, linewidth=2.2, alpha=0.35)
            ax.plot(t[::step], pred_qlv[::step], color=color, linewidth=1.0,
                    linestyle="--", marker="s", markersize=2.2)
            ax.plot(t[::step], pred_prf[::step], color=color, linewidth=1.0,
                    linestyle="-", marker="o", markersize=2.2, label=f"{rate_group}")
        ax.set_xlabel("Time (s)")
        ax.set_title(material, fontsize=9)
        ax.legend(fontsize=6, title="rate (solid=PRF, dashed=QLV, faded=measured)", title_fontsize=5)

    axes[0].set_ylabel("Stress (MPa)")
    fig.suptitle("PRF vs. QLV vs. measured stress relaxation", fontsize=10)
    fig.tight_layout()
    fig.savefig(BASE / "prf_model_validation.png", dpi=300, bbox_inches="tight")
    print(f"saved {BASE / 'prf_model_validation.png'}")

    # --- final comparison table: QLV vs PRF RMSE, all materials/rate groups ---
    merged = qlv_validation.merge(prf_validation, on=["material", "rate_group", "test_id"],
                                   suffixes=("_qlv", "_prf"))
    merged["improvement_pct_points"] = merged["rmse_pct_of_peak_qlv"] - merged["rmse_pct_of_peak_prf"]
    merged.to_csv(BASE / "prf_vs_qlv_comparison.csv", index=False)
    print(f"\nsaved {BASE / 'prf_vs_qlv_comparison.csv'}")
    print("\n=== Final comparison: QLV baseline vs. PRF (RMSE, % of peak stress) ===")
    print(merged[["material", "rate_group", "rmse_pct_of_peak_qlv", "rmse_pct_of_peak_prf",
                  "improvement_pct_points"]].to_string(index=False))


if __name__ == "__main__":
    sanity_check()
    main()
