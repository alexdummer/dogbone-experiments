"""Fit a universal large-deformation hyper-viscoelastic model (2-term Ogden
hyperelasticity + 3-term Prony series) per composition, for later use as an
Abaqus *HYPERELASTIC / *VISCOELASTIC material card in FE structural models.

Framework: finite-strain quasi-linear viscoelasticity (QLV), the standard
separable ansatz used by Abaqus/ANSYS:

    sigma(t) = g_inf * sigma_ogden(lambda(t))
               + sum_i  q_i(t)                    (internal variables)
    dq_i/dt + q_i/tau_i = g_i * d(sigma_ogden)/dt

The Ogden potential represents the INSTANTANEOUS (unrelaxed, t=0) response;
the Prony series g(t) = g_inf + sum_i g_i exp(-t/tau_i) describes how much
of that instantaneous stress survives at long time (g_inf) and how fast the
rest decays (g_i, tau_i). g(0) = 1 is enforced (g_inf = 1 - sum_i g_i) so
the Ogden fit and the Prony fit share a consistent t=0 reference.

Fitting data:
  - Ogden params: the FAST-rate ramp segment (nominal stress vs. stretch),
    since fast loading best approximates the instantaneous/unrelaxed curve.
  - Prony params: the normalized relaxation-hold curve, averaged across
    ALL rate groups (very slow/slow/fast) per composition -- the QLV
    ansatz assumes this shape is rate-independent, so using all groups
    gives the best average compromise rather than favoring one rate.

Important caveats (see conversation): the separable QLV ansatz assumes a
strain/rate-independent relaxation shape, but this dataset shows a real
rate-dependent crossover in relaxation behavior; the validation step
below quantifies how much error that assumption introduces. The Ogden fit
identifies only the deviatoric (shear) response since these are uniaxial
tension tests with no transverse strain measurement -- compressibility
(D_p) must be assumed (near-incompressible) for FE use, not fit from
this data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from plotstyle import colors, figsize_double

BASE = Path(__file__).parent

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

TIME_GRID = np.logspace(np.log10(0.05), np.log10(590), 300)

MATERIALS = {
    "A0V100": {
        "dir": BASE / "A0V100-relax" / "cleaned",
        "rate_groups": {
            "very slow": [10, 11, 12],
            "slow": [1, 3, 13],
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
            "slow": [4, 5, 6],
            "fast": [7, 11, 12],
        },
    },
}


def load(cfg, material, test_id):
    prefix = "A50V50" if material == "A50V50" else material
    df = pd.read_csv(cfg["dir"] / f"{prefix}-{test_id}_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


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


# ---------------------------------------------------------------- Prony ---
def prony(t, g1, tau1, g2, tau2, g3, tau3):
    g_inf = 1 - g1 - g2 - g3
    return g_inf + g1 * np.exp(-t / tau1) + g2 * np.exp(-t / tau2) + g3 * np.exp(-t / tau3)


def fit_prony(material, cfg):
    curves = []
    for ids in cfg["rate_groups"].values():
        for test_id in ids:
            df = load(cfg, material, test_id)
            onset, ramp_end = find_ramp_bounds(df["position_mm"])
            t_end = df["time_s"].iloc[ramp_end]
            s_end = df["stress"].iloc[ramp_end]
            t_rel = df["time_s"].to_numpy() - t_end
            mask = t_rel > 0
            curves.append(np.interp(TIME_GRID, t_rel[mask], df["stress"].to_numpy()[mask] / s_end))
    avg_curve = np.mean(curves, axis=0)

    p0 = [0.2, 1.0, 0.2, 10.0, 0.2, 100.0]
    bounds = (0, [1, 1e4, 1, 1e4, 1, 1e4])
    popt, _ = curve_fit(prony, TIME_GRID, avg_curve, p0=p0, bounds=bounds, maxfev=50000)
    resid = avg_curve - prony(TIME_GRID, *popt)

    g1, tau1, g2, tau2, g3, tau3 = popt
    g_inf = 1 - g1 - g2 - g3
    return {
        "g1": g1, "tau1": tau1, "g2": g2, "tau2": tau2, "g3": g3, "tau3": tau3,
        "g_inf": g_inf, "n_curves": len(curves), "rms_residual": float(np.sqrt(np.mean(resid**2))),
    }


# ---------------------------------------------------------------- Ogden ---
def ogden_nominal_stress(stretch, mu1, alpha1, mu2, alpha2):
    def term(mu, alpha):
        return (2 * mu / alpha) * (stretch**(alpha - 1) - stretch**(-alpha / 2 - 1))
    return term(mu1, alpha1) + term(mu2, alpha2)


def fit_ogden(material, cfg):
    # fast-rate group = closest available approximation to the
    # "instantaneous" (unrelaxed) reference curve the Ogden term represents
    test_id = cfg["rate_groups"]["fast"][0]
    df = load(cfg, material, test_id)
    onset, ramp_end = find_ramp_bounds(df["position_mm"])

    strain = df["strain"].to_numpy()[onset:ramp_end]
    stress = df["stress"].to_numpy()[onset:ramp_end]
    stretch = 1 + strain

    # mixed-sign alpha (one term > 0, one < 0) is a standard Ogden
    # parameterization that avoids the same-sign fit collapsing onto a
    # single degenerate term; |alpha| <= 16 also keeps the model numerically
    # well-behaved if it is later used in FE
    p0 = [stress.max() / 4, 3.0, stress.max() / 4, -3.0]
    bounds = ([1e-4, 0.1, 1e-4, -16], [1e5, 16, 1e5, -0.1])
    popt, _ = curve_fit(ogden_nominal_stress, stretch, stress, p0=p0, bounds=bounds, maxfev=100000)
    pred = ogden_nominal_stress(stretch, *popt)
    resid = stress - pred
    r2 = 1 - np.var(resid) / np.var(stress)

    mu1, alpha1, mu2, alpha2 = popt
    return {
        "mu1": mu1, "alpha1": alpha1, "mu2": mu2, "alpha2": alpha2,
        "fit_test_id": test_id, "r2": float(r2),
        "fit_strain_max": float(strain.max()),
        "small_strain_modulus_3mu": 3 * (mu1 + mu2),
    }


# ------------------------------------------------------------ QLV model ---
def predict_stress(time_s, strain, ogden_params, prony_params):
    """Recursive (algorithmic) Prony update: the same incremental scheme
    FE codes use internally. Given a measured strain(t) history, predicts
    the QLV stress(t) from the fitted Ogden (instantaneous) hyperelastic
    law and Prony relaxation kernel."""
    stretch = 1 + strain
    sigma_ogden = ogden_nominal_stress(
        stretch, ogden_params["mu1"], ogden_params["alpha1"],
        ogden_params["mu2"], ogden_params["alpha2"],
    )

    taus = [prony_params["tau1"], prony_params["tau2"], prony_params["tau3"]]
    gs = [prony_params["g1"], prony_params["g2"], prony_params["g3"]]
    g_inf = prony_params["g_inf"]

    n = len(time_s)
    q = np.zeros((3, n))
    dt = np.diff(time_s)
    d_sigma = np.diff(sigma_ogden)

    for i in range(3):
        tau = taus[i]
        decay = np.exp(-dt / tau)
        for k in range(n - 1):
            q[i, k + 1] = q[i, k] * decay[k] + gs[i] * d_sigma[k] * decay[k]

    return g_inf * sigma_ogden + q.sum(axis=0)


def main():
    all_prony = {}
    all_ogden = {}

    for material, cfg in MATERIALS.items():
        prony_params = fit_prony(material, cfg)
        ogden_params = fit_ogden(material, cfg)
        all_prony[material] = prony_params
        all_ogden[material] = ogden_params
        print(f"\n{material}")
        print("  Prony:", {k: round(v, 4) if isinstance(v, float) else v for k, v in prony_params.items()})
        print("  Ogden:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ogden_params.items()})

    pd.DataFrame(all_prony).T.to_csv(BASE / "qlv_prony_params.csv")
    pd.DataFrame(all_ogden).T.to_csv(BASE / "qlv_ogden_params.csv")
    print(f"\nsaved qlv_prony_params.csv and qlv_ogden_params.csv")

    # --- validation: predicted vs measured stress-time, all rate groups ---
    fig, axes = plt.subplots(1, 3, figsize=figsize_double, sharey=False)
    rate_colors = {"very slow": colors[0], "slow": colors[1], "fast": colors[2]}

    validation_rows = []
    for ax, (material, cfg) in zip(axes, MATERIALS.items()):
        for rate_group, ids in cfg["rate_groups"].items():
            test_id = ids[0]
            df = load(cfg, material, test_id)
            pred = predict_stress(df["time_s"].to_numpy(), df["strain"].to_numpy(),
                                   all_ogden[material], all_prony[material])
            measured = df["stress"].to_numpy()
            peak = measured.max()
            rmse = np.sqrt(np.mean((pred - measured) ** 2))
            validation_rows.append({
                "material": material, "rate_group": rate_group, "test_id": test_id,
                "rmse_mpa": rmse, "rmse_pct_of_peak": 100 * rmse / peak,
            })
            ax.plot(df["time_s"], df["stress"], color=rate_colors[rate_group], linewidth=2.2,
                    alpha=0.4, label=f"{rate_group} (measured)")
            # downsample the QLV curve for plotting -- with 1e5-1e6 densely
            # sampled points, a dashed/marker linestyle renders sub-pixel and
            # becomes visually indistinguishable from a solid line otherwise
            n = len(df)
            step = max(1, n // 300)
            ax.plot(df["time_s"].to_numpy()[::step], pred[::step], color=rate_colors[rate_group],
                    linewidth=1.0, linestyle="--", marker="o", markersize=2.5,
                    label=f"{rate_group} (QLV)")
        ax.set_xlabel("Time (s)")
        ax.set_title(material, fontsize=9)
        ax.legend(fontsize=6)

    axes[0].set_ylabel("Stress (MPa)")
    fig.suptitle("QLV model (Ogden + Prony) vs. measured -- one specimen per rate group", fontsize=10)
    fig.tight_layout()
    out_path = BASE / "qlv_model_validation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(BASE / "qlv_model_validation.csv", index=False)
    print("\nvalidation (RMSE vs. measured, one specimen per rate group):")
    print(validation.to_string(index=False))
    print(
        "\nNote: the QLV ansatz assumes a rate-independent relaxation shape "
        "(the Prony fit is one compromise curve averaged across all rate "
        "groups). The 'very slow' group -- especially for A50V50, which "
        "shows a real overshoot/hump the model does not reproduce -- is "
        "the expected failure mode flagged before fitting: a single g(t) "
        "cannot capture the rate-dependent relaxation-shape crossover this "
        "dataset shows. Treat 'very slow' fit quality as a lower bound on "
        "accuracy, not representative of the whole model."
    )

    export_abaqus(all_ogden, all_prony)


def export_abaqus(all_ogden, all_prony):
    """Write one parameter file per composition in a layout that maps
    directly onto an Abaqus *HYPERELASTIC, OGDEN, N=2 card (mu_i, alpha_i,
    D_i) and a *VISCOELASTIC, TIME=PRONY card (g_i^P, k_i^P, tau_i).

    D_i (compressibility) is set to 0 (fully incompressible) as a
    placeholder: these are uniaxial tension tests with no transverse
    strain measurement, so compressibility cannot be identified from this
    data and must come from elsewhere (assume near-incompressible, or
    measure independently, before using in a real FE model). k_i^P (bulk
    Prony weights) are set equal to g_i^P, the standard assumption when no
    separate volumetric relaxation data exists.
    """
    for material in all_ogden:
        og = all_ogden[material]
        pr = all_prony[material]
        lines = [
            f"# {material} -- Ogden (N=2) + Prony (N=3) QLV parameters",
            "# Units: stress in MPa, time in seconds",
            "",
            "*HYPERELASTIC, OGDEN, N=2",
            "** mu1, alpha1, mu2, alpha2, D1, D2",
            f"{og['mu1']:.6g}, {og['alpha1']:.6g}, {og['mu2']:.6g}, {og['alpha2']:.6g}, 0, 0",
            "",
            "*VISCOELASTIC, TIME=PRONY",
            "** g_i^P, k_i^P, tau_i",
            f"{pr['g1']:.6g}, {pr['g1']:.6g}, {pr['tau1']:.6g}",
            f"{pr['g2']:.6g}, {pr['g2']:.6g}, {pr['tau2']:.6g}",
            f"{pr['g3']:.6g}, {pr['g3']:.6g}, {pr['tau3']:.6g}",
            "",
            f"** implied g_inf (long-term/instantaneous stiffness ratio) = {pr['g_inf']:.4g}",
            f"** Ogden fit R2 = {og['r2']:.4g} (fit strain range: 0 to {og['fit_strain_max']:.4g})",
            f"** Prony fit rms residual = {pr['rms_residual']:.4g} (n={pr['n_curves']} curves, all rate groups)",
            "** CAUTION: D1=D2=0 assumes full incompressibility -- these are",
            "** uniaxial tests with no transverse strain measurement, so",
            "** compressibility is NOT identified from this data.",
        ]
        out_path = BASE / f"qlv_abaqus_{material}.inp"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
