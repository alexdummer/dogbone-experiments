"""Calibrate Marmot's CompressibleFiniteStrainFractionalSchapery material --
a finite-strain fractional-order Schapery viscoelastic model with selectable
hyperelastic base potentials (NeoHooke, MooneyRivlin, Yeoh) -- against multi-rate
stress-relaxation experiments across photopolymer blends (A0V100 through A100V0).

Combines:
  - Selectable hyperelastic base potentials composing Psi = Psi_iso + kappa/8*(ln J)^2,
    with kappa = 100*C1 fixed near-incompressible.
  - A fractional-order relaxation kernel with power-law exponent alpha in (0, 1)
    and characteristic time scale tau0.
  - Schapery non-linear strain-dependent functions g1(Ibar1), g2(Ibar1), and shift factor a_S(Ibar1).
"""

import os
import sys
from pathlib import Path

# Insert calibration and analysis module paths
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "1d-qlv"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "prf"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration" / "qlv-marmot"))
sys.path.insert(0, str(REPO_ROOT / "simulations" / "material-calibration"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import marmot

from fit_prf_model import downsample_for_fit
from fit_qlv_model import MATERIALS, load, find_ramp_bounds, mean_strain_rates, estimate_youngs_modulus
from plotstyle import colors, figsize_double
from poisson_ratios import POISSON_RATIOS, bulk_modulus_from_poisson, shear_modulus_from_base

CALIB_DIR = REPO_ROOT / "simulations" / "material-calibration"
PARAMS_DIR = CALIB_DIR / "parameters"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

BASES = {"NeoHooke": 0, "Yeoh": 1, "MooneyRivlin": 2}
N_ELASTIC = {"NeoHooke": 2, "Yeoh": 4, "MooneyRivlin": 3}
BASE_LINESTYLE = {"NeoHooke": "-", "Yeoh": "--", "MooneyRivlin": ":"}
BASE_MARKER = {"NeoHooke": "o", "Yeoh": "s", "MooneyRivlin": "^"}
ONLY_SHEAR_CREEP = 1.0

IS_GRADU_CONTROLLED = np.array([[True, True, True], [True, False, True], [True, True, False]])
IS_STRESS_CONTROLLED = np.logical_not(IS_GRADU_CONTROLLED)


def _solve_quiet(solver):
    """Run solver.solve(), discarding C++ stdout."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        solver.solve()
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def build_props(base_name, elastic, alpha, tau0, g_inf, a1, a2, b_S):
    base_id = BASES[base_name]
    assert len(elastic) == N_ELASTIC[base_name]
    props = [base_id, ONLY_SHEAR_CREEP, *elastic, alpha, tau0, g_inf, a1, a2, b_S]
    return np.array(props, dtype=np.float64)


def elastic_from_C1(base_name, C1, C2, C3, nu=0.49):
    mu_0 = shear_modulus_from_base(base_name, C1, C2, C3)
    kappa = bulk_modulus_from_poisson(mu_0, nu)
    if base_name == "NeoHooke":
        return [C1, kappa]
    if base_name == "MooneyRivlin":
        return [C1, C2, kappa]
    if base_name == "Yeoh":
        return [C1, C2, C3, kappa]
    raise ValueError(base_name)


def _simulate_once(t, stretch, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S, nu=0.49):
    """Replay measured (t, stretch) through CompressibleFiniteStrainFractionalSchapery."""
    elastic = elastic_from_C1(base_name, C1, C2, C3, nu=nu)
    props = build_props(base_name, elastic, alpha, tau0, g_inf, a1, a2, b_S)
    options = marmot.solvers.FiniteStrainSolver.SolverOptions()
    solver = marmot.solvers.FiniteStrainSolver("COMPRESSIBLEFINITESTRAINFRACTIONALSCHAPERY", props, options)

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt <= 0:
            continue
        step = marmot.solvers.FiniteStrainSolver.Step()
        step.timeStart = float(t[i - 1])
        step.timeEnd = float(t[i])
        step.dTStart = dt
        step.dTMax = dt
        step.dTMin = dt / 50.0
        step.maxIncrements = 30
        step.isGradUComponentControlled = IS_GRADU_CONTROLLED
        step.isStressComponentControlled = IS_STRESS_CONTROLLED
        gradu_target = np.zeros((3, 3))
        gradu_target[0, 0] = stretch[i] - stretch[i - 1]
        step.gradUIncrementTarget = gradu_target
        step.stressIncrementTarget = np.zeros((3, 3))
        solver.addStep(step)

    _solve_quiet(solver)
    history = solver.getHistory()
    h_t = np.array([h.time for h in history])
    h_F11 = np.array([h.F[0, 0] for h in history])
    h_tau11 = np.array([h.stress[0, 0] for h in history])
    h_P11 = h_tau11 / h_F11
    result = np.interp(t, h_t, h_P11)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite prediction")
    return result


def unpack(x, base_name, material=None):
    if material is not None and material != "A100V0":
        a1, a2, b_S = 0.0, 0.0, 0.0
        if base_name == "NeoHooke":
            C1, alpha, tau0, g_inf = x
            return C1, 0.0, 0.0, alpha, tau0, g_inf, a1, a2, b_S
        if base_name == "MooneyRivlin":
            C1, C2, alpha, tau0, g_inf = x
            return C1, C2, 0.0, alpha, tau0, g_inf, a1, a2, b_S
        if base_name == "Yeoh":
            C1, C2, C3, alpha, tau0, g_inf = x
            return C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S
    else:
        if base_name == "NeoHooke":
            C1, alpha, tau0, g_inf, a1, a2, b_S = x
            return C1, 0.0, 0.0, alpha, tau0, g_inf, a1, a2, b_S
        if base_name == "MooneyRivlin":
            C1, C2, alpha, tau0, g_inf, a1, a2, b_S = x
            return C1, C2, 0.0, alpha, tau0, g_inf, a1, a2, b_S
        if base_name == "Yeoh":
            C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S = x
            return C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S
    raise ValueError(base_name)


def seed_and_bounds(base_name, G0, tau0_seed=10.0, tau_lb=1e-3, material=None):
    alpha_x0 = 0.35
    alpha_lb, alpha_ub = 0.05, 0.95
    tau_lb, tau_ub = tau_lb, 1e4
    g_inf_x0 = 0.25
    g_inf_lb, g_inf_ub = 0.02, 0.95

    fit_nonlinear = (material == "A100V0" or material is None)
    if fit_nonlinear:
        a1_x0, a2_x0, b_S_x0 = 0.0, 0.0, 0.0
        a_lb, a_ub = -1.0, 5.0
        b_lb, b_ub = 0.0, 5.0

    if base_name == "NeoHooke":
        if fit_nonlinear:
            x0 = [G0, alpha_x0, tau0_seed, g_inf_x0, a1_x0, a2_x0, b_S_x0]
            lb = [1e-3, alpha_lb, tau_lb, g_inf_lb, a_lb, a_lb, b_lb]
            ub = [1e4, alpha_ub, tau_ub, g_inf_ub, a_ub, a_ub, b_ub]
        else:
            x0 = [G0, alpha_x0, tau0_seed, g_inf_x0]
            lb = [1e-3, alpha_lb, tau_lb, g_inf_lb]
            ub = [1e4, alpha_ub, tau_ub, g_inf_ub]
    elif base_name == "MooneyRivlin":
        C1_0 = G0 / 2.0
        C2_0 = 0.1 * C1_0
        if fit_nonlinear:
            x0 = [C1_0, C2_0, alpha_x0, tau0_seed, g_inf_x0, a1_x0, a2_x0, b_S_x0]
            lb = [1e-3, -1e3, alpha_lb, tau_lb, g_inf_lb, a_lb, a_lb, b_lb]
            ub = [1e4, 1e4, alpha_ub, tau_ub, g_inf_ub, a_ub, a_ub, b_ub]
        else:
            x0 = [C1_0, C2_0, alpha_x0, tau0_seed, g_inf_x0]
            lb = [1e-3, -1e3, alpha_lb, tau_lb, g_inf_lb]
            ub = [1e4, 1e4, alpha_ub, tau_ub, g_inf_ub]
    elif base_name == "Yeoh":
        C1_0 = G0 / 2.0
        if fit_nonlinear:
            x0 = [C1_0, 0.01 * C1_0, 0.0, alpha_x0, tau0_seed, g_inf_x0, a1_x0, a2_x0, b_S_x0]
            lb = [1e-3, -1e3, -1e3, alpha_lb, tau_lb, g_inf_lb, a_lb, a_lb, b_lb]
            ub = [1e4, 1e4, 1e4, alpha_ub, tau_ub, g_inf_ub, a_ub, a_ub, b_ub]
        else:
            x0 = [C1_0, 0.01 * C1_0, 0.0, alpha_x0, tau0_seed, g_inf_x0]
            lb = [1e-3, -1e3, -1e3, alpha_lb, tau_lb, g_inf_lb]
            ub = [1e4, 1e4, 1e4, alpha_ub, tau_ub, g_inf_ub]
    else:
        raise ValueError(base_name)
    return list(np.clip(x0, lb, ub)), lb, ub


def fit_joint(material, cfg, base_name, G0, tau0_seed=10.0, tau_lb=1e-3, max_nfev=60, verbose=False):
    test_data = {}
    for rate_group, ids in cfg["rate_groups"].items():
        df = load(cfg, material, ids[0])
        t, strain, stress = downsample_for_fit(df, n_ramp=30, n_hold=45)
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t <= t_ramp_end
        n_ramp_pts, n_hold_pts = is_ramp.sum(), (~is_ramp).sum()
        weight = np.where(is_ramp, 1.0 / np.sqrt(max(n_ramp_pts, 1)), 1.0 / np.sqrt(max(n_hold_pts, 1)))
        test_data[rate_group] = (t, 1.0 + strain, stress, stress.max(), weight)

    nu = POISSON_RATIOS.get(material, 0.49)
    x0, lb, ub = seed_and_bounds(base_name, G0, tau0_seed=tau0_seed, tau_lb=tau_lb, material=material)

    def residuals(x):
        C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S = unpack(x, base_name, material=material)
        out = []
        for t, stretch, stress, peak, weight in test_data.values():
            try:
                pred = _simulate_once(t, stretch, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S, nu=nu)
            except Exception:
                pred = np.zeros_like(stress)
            out.append((pred - stress) / peak * weight)
        return np.concatenate(out)

    result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", verbose=2 if verbose else 0,
                           xtol=1e-8, ftol=1e-8, gtol=1e-8, diff_step=1e-3, max_nfev=max_nfev)
    C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S = unpack(result.x, base_name, material=material)
    mu_0 = shear_modulus_from_base(base_name, C1, C2, C3)
    K = bulk_modulus_from_poisson(mu_0, nu)
    return {
        "hyperelastic_base": base_name, "nu": nu,
        "C1": C1, "C2": C2, "C3": C3,
        "K": K,
        "alpha": alpha, "tau0": tau0, "g_inf": g_inf,
        "a1": a1, "a2": a2, "b_S": b_S,
        "cost": result.cost, "nfev": result.nfev,
    }


def validate(material, cfg, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S):
    nu = POISSON_RATIOS.get(material, 0.49)
    rows = []
    plot_rows = {}
    for rate_group, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, material, test_id)
        t = df["time_s"].to_numpy()
        measured = df["stress"].to_numpy()
        t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=50, n_hold=80)
        try:
            pred = _simulate_once(t_ds, 1.0 + strain_ds, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S, nu=nu)
        except Exception as exc:
            print(f"  [validate] {material}/{base_name}/{rate_group}: solver failed ({exc}); skipping")
            continue
        peak = stress_ds.max()
        rmse = np.sqrt(np.mean((pred - stress_ds) ** 2))
        rows.append({"material": material, "hyperelastic_base": base_name, "rate_group": rate_group,
                     "test_id": test_id, "rmse_mpa": rmse, "rmse_pct_of_peak": 100.0 * rmse / peak})
        plot_rows[rate_group] = (t, measured, t_ds, pred)
    return pd.DataFrame(rows), plot_rows


def validate_ramp(material, cfg, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S):
    nu = POISSON_RATIOS.get(material, 0.49)
    ramp_rows = {}
    for rate_group, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, material, test_id)
        onset, ramp_end = find_ramp_bounds(df["position_mm"])
        sl = slice(onset, ramp_end + 1)
        t_full = df["time_s"].to_numpy()[sl]
        strain_full = df["strain"].to_numpy()[sl]
        measured_stress = df["stress"].to_numpy()[sl]

        n_pts = 35
        idx = np.linspace(0, len(t_full) - 1, n_pts, dtype=int)
        t_ds = t_full[idx]
        strain_ds = strain_full[idx]
        try:
            pred = _simulate_once(t_ds, 1.0 + strain_ds, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S, nu=nu)
        except Exception as exc:
            print(f"  [validate_ramp] {material}/{base_name}/{rate_group}: solver failed ({exc}); skipping")
            continue
        ramp_rows[rate_group] = (100.0 * strain_full, measured_stress, 100.0 * strain_ds, pred)
    return ramp_rows


def make_grid_plot(data_by_material_base, rate_groups, materials, bases, xlabel, suptitle, out_path,
                   strain_rates=None, sharex=True, rate_label_loc="upper right"):
    n_real = sum(1 for m in materials if m in data_by_material_base)
    height_ratios = [1.0] * n_real
    MAX_TOTAL_HEIGHT_IN = 7.5
    per_row_scale = min(0.8, MAX_TOTAL_HEIGHT_IN / (figsize_double[1] * sum(height_ratios)))
    fig, axes = plt.subplots(len(materials), 3,
                             figsize=(figsize_double[0], figsize_double[1] * per_row_scale * sum(height_ratios)),
                             sharey="row", sharex=sharex, gridspec_kw={"height_ratios": height_ratios})
    for row, material in enumerate(materials):
        for col, rate_group in enumerate(rate_groups):
            ax = axes[row, col]
            color = colors[col]
            x_full, measured = None, None
            for base_name in bases:
                if base_name not in data_by_material_base[material]:
                    continue
                if rate_group not in data_by_material_base[material][base_name]:
                    continue
                x_full, measured, x_ds, pred = data_by_material_base[material][base_name][rate_group]
                ax.plot(x_ds, pred, color=color, linewidth=1.1, linestyle=BASE_LINESTYLE[base_name],
                        marker=BASE_MARKER[base_name], markersize=2.2, markevery=4)
            if measured is not None:
                ax.plot(x_full, measured, color=color, linewidth=2.2, alpha=0.35, zorder=0)
            ax.grid(True, alpha=0.4)
            if row == 0:
                rate_title = {"very slow": "slow", "slow": "medium", "fast": "fast"}.get(rate_group, rate_group)
                ax.set_title(rate_title, fontsize=9)
            if col == 0:
                disp_mat = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}.get(material, material)
                ax.set_ylabel(f"{disp_mat}\nEngineering stress in MPa", fontsize=8)
            if row == n_real - 1:
                ax.set_xlabel(xlabel)
            if strain_rates is not None:
                rate = strain_rates.get(material, {}).get(rate_group)
                if rate is not None:
                    label_x, ha = (0.04, "left") if rate_label_loc == "upper left" else (0.96, "right")
                    ax.text(label_x, 0.96, rf"$\dot\varepsilon\approx${100 * rate:.3g}\%/s", fontsize=6,
                            ha=ha, va="top", transform=ax.transAxes, color="black",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))

    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color="grey", alpha=0.35, linewidth=2.2, label="measured")]
    legend_handles += [
        Line2D([0], [0], color="grey", linestyle=BASE_LINESTYLE[b], marker=BASE_MARKER[b],
               markersize=4, label=b) for b in bases
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(bases) + 1, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(suptitle, fontsize=11, y=1.07)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path.with_suffix('.pdf')} and .png")


def main():
    print("=== Starting Fractional Schapery calibration pipeline ===", flush=True)
    all_rows = []
    validation_rows = []
    plot_data = {mat: {} for mat in MATERIALS}
    ramp_data = {mat: {} for mat in MATERIALS}

    for material, cfg in MATERIALS.items():
        E_mean = np.mean(list(estimate_youngs_modulus(cfg, material).values()))
        G0 = E_mean / 3.0
        tau_lb = 0.1 if material == "A100V0" else 1e-3

        plot_data[material] = {}
        ramp_data[material] = {}

        for base_name in BASES:
            print(f"\n--- Fitting Fractional Schapery [{base_name}] for {material} ---", flush=True)
            params = fit_joint(material, cfg, base_name, G0, tau0_seed=10.0, tau_lb=tau_lb, max_nfev=60, verbose=False)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}, flush=True)
            row = {"material": material}
            row.update(params)
            all_rows.append(row)

            C1, C2, C3 = params["C1"], params["C2"], params["C3"]
            alpha, tau0, g_inf = params["alpha"], params["tau0"], params["g_inf"]
            a1, a2, b_S = params["a1"], params["a2"], params["b_S"]

            val, plot_rows = validate(material, cfg, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S)
            validation_rows.append(val)
            plot_data[material][base_name] = plot_rows
            ramp_data[material][base_name] = validate_ramp(material, cfg, base_name, C1, C2, C3, alpha, tau0, g_inf, a1, a2, b_S)

    # Save calibrated parameters
    params_df = pd.DataFrame(all_rows)
    params_df.to_csv(PARAMS_DIR / "fractional_schapery_params.csv", index=False)
    params_df.to_csv(CALIB_DIR / "fractional_schapery_params.csv", index=False)
    print(f"\nSaved calibrated parameters to {PARAMS_DIR / 'fractional_schapery_params.csv'}")

    validation = pd.concat(validation_rows, ignore_index=True)
    validation.to_csv(PARAMS_DIR / "fractional_schapery_validation.csv", index=False)
    validation.to_csv(CALIB_DIR / "fractional_schapery_validation.csv", index=False)
    print(f"Saved validation metrics to {PARAMS_DIR / 'fractional_schapery_validation.csv'}")

    # Summary table
    print("\n=== Fractional Schapery Validation RMSE (% of peak stress) ===")
    print(validation[["material", "hyperelastic_base", "rate_group", "rmse_pct_of_peak"]].to_string(index=False))

    # Generate publication grid plots
    rate_groups = ["very slow", "slow", "fast"]
    materials = list(MATERIALS)
    bases = list(BASES)
    strain_rates = mean_strain_rates()

    make_grid_plot(plot_data, rate_groups, materials, bases, "Time in s",
                   "CompressibleFiniteStrainFractionalSchapery (Marmot) vs. measured stress relaxation",
                   CALIB_DIR / "fractional_schapery_validation_grid", strain_rates=strain_rates)
    make_grid_plot(ramp_data, rate_groups, materials, bases, "Engineering strain in %",
                   "CompressibleFiniteStrainFractionalSchapery (Marmot) vs. measured stress-strain (ramp only)",
                   CALIB_DIR / "fractional_schapery_stress_strain_grid", strain_rates=strain_rates, sharex=False,
                   rate_label_loc="upper left")

    print("\n=== Fractional Schapery pipeline completed successfully ===", flush=True)


if __name__ == "__main__":
    main()
