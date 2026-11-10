#!/usr/bin/env python3
"""
predict_qlv_from_dma.py

Cross-modality constitutive modeling and blind forward predictions:
1. Extract linear viscoelastic relaxation spectra from small-strain DMA master curves at 20 C:
   - Discrete 6-term generalized Maxwell spectrum (gammas, taus) for QLV.
   - Continuous fractional power-law spectrum (alpha, tau0, g_inf) for Fractional Schapery.
2. Calibrate hyperelastic base parameters on a SINGLE tensile loading ramp:
   - Mooney-Rivlin parameters (C1, C2) for QLV.
   - Neo-Hookean parameter (C1) for Fractional Schapery (with bulk modulus kappa from physical nu).
   Default calibration rate: "slow" (0.167 mm/s), directly matching the lattice experiment rate.
3. Perform blind forward predictions across the remaining rates ("very slow" 0.017 mm/s, "fast" 1.67 mm/s)
   and full 600-second stress-relaxation holds.
4. Generate comparative 5x4 validation grid and LaTeX summary table.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import seaborn as sns
from matplotlib.lines import Line2D

# Add local paths for helper imports
REPO_ROOT = Path(__file__).resolve().parents[3]
SYS_PATHS = [
    REPO_ROOT / "simulations" / "material-calibration" / "qlv-marmot",
    REPO_ROOT / "simulations" / "material-calibration" / "fractional-schapery",
    REPO_ROOT / "simulations" / "material-calibration" / "1d-qlv",
    REPO_ROOT / "simulations" / "material-calibration" / "prf",
    REPO_ROOT / "simulations" / "material-calibration",
    REPO_ROOT / "analysis",
]
for p in SYS_PATHS:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import marmot
from fit_prf_model import downsample_for_fit
from fit_qlv_model import MATERIALS, load, find_ramp_bounds, mean_strain_rates
import fit_qlv_marmot_model as qm
from fit_fractional_schapery_model import _simulate_once as sim_fs, elastic_from_C1
from poisson_ratios import POISSON_RATIOS
from plotstyle import colors, figsize_double

DMA_RAW_DIR = REPO_ROOT / "data" / "dma" / "raw"
OUTPUT_DIR = REPO_ROOT / "analysis"
PARAMS_DIR = REPO_ROOT / "simulations" / "material-calibration" / "parameters"

QUALIFYING_DMA_SAMPLES = {
    "A0V100": ["A0V100-2", "A0V100-4", "A0V100-5"],
    "A25V75": ["A25V75-2", "A25V75-3", "A25V75-4"],
    "A50V50": ["A50V50-7", "A50V50-8", "A50V50-9"],
    "A75V25": ["A75V25-2", "A75V25-3", "A75V25-4"],
    "A100V0": ["A100V0-1", "A100V0-2", "A100V0-3"],
}

FACTORS_FS = np.array([0.01, 0.1, 1.0, 10.0, 100.0])


def build_props(base_name, elastic, gammas, taus):
    """Assemble material properties for arbitrary number of Maxwell elements."""
    base_id = qm.BASES[base_name]
    n_maxwell = len(gammas)
    props = [base_id, qm.ONLY_SHEAR_CREEP, *elastic, n_maxwell]
    for g, t in zip(gammas, taus):
        props += [g, t]
    return np.array(props, dtype=np.float64)


def simulate_once(t, stretch, base_name, C1, C2, C3, gammas, taus):
    """Run Marmot solver for arbitrary number of Maxwell elements."""
    elastic = qm.elastic_from_C1(base_name, C1, C2, C3)
    props = build_props(base_name, elastic, gammas, taus)
    options = marmot.solvers.FiniteStrainSolver.SolverOptions()
    solver = marmot.solvers.FiniteStrainSolver("COMPRESSIBLEFINITESTRAINLINEARVISCOELASTICITY", props, options)

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
        step.isGradUComponentControlled = qm.IS_GRADU_CONTROLLED
        step.isStressComponentControlled = qm.IS_STRESS_CONTROLLED
        gradu_target = np.zeros((3, 3))
        gradu_target[0, 0] = stretch[i] - stretch[i - 1]
        step.gradUIncrementTarget = gradu_target
        step.stressIncrementTarget = np.zeros((3, 3))
        solver.addStep(step)

    qm._solve_quiet(solver)
    history = solver.getHistory()
    h_t = np.array([h.time for h in history])
    h_F11 = np.array([h.F[0, 0] for h in history])
    h_tau11 = np.array([h.stress[0, 0] for h in history])
    h_P11 = h_tau11 / h_F11
    result = np.interp(t, h_t, h_P11)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite prediction")
    return result


def load_dma_data(mat):
    samples = QUALIFYING_DMA_SAMPLES[mat]
    if isinstance(samples, str):
        samples = [samples]

    dfs = []
    for sample in samples:
        csv_path = DMA_RAW_DIR / mat / "processed" / f"{sample}__direct_wlf_master_curve.csv"
        if csv_path.exists():
            d = pd.read_csv(csv_path)
            d = d[d["storage_modulus_kPa"] > 0].copy()
            d["E_prime_MPa"] = d["storage_modulus_kPa"] / 1000.0
            dfs.append(d)

    if not dfs:
        candidates = list((DMA_RAW_DIR / mat / "processed").glob("*__direct_wlf_master_curve.csv"))
        if not candidates:
            raise FileNotFoundError(f"No DMA master curve found for {mat}")
        d = pd.read_csv(candidates[0])
        d = d[d["storage_modulus_kPa"] > 0].copy()
        d["E_prime_MPa"] = d["storage_modulus_kPa"] / 1000.0
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    f_min = max(df["reduced_freq_hz"].min(), 1e-5)
    f_max = min(max(df["reduced_freq_hz"].min() * 1e7, 1e2), 1e4)
    sub = df[(df["reduced_freq_hz"] >= f_min) & (df["reduced_freq_hz"] <= f_max)].copy()
    if len(sub) < 18:
        sub = df.iloc[:min(len(df), 50)].copy()

    sub = sub.sort_values("reduced_freq_hz")
    freq = sub["reduced_freq_hz"].to_numpy()
    omega = 2 * np.pi * freq
    E_prime = sub["E_prime_MPa"].to_numpy()
    return freq, omega, E_prime


def fit_dma_prony(mat, n_terms=6):
    """
    Fit an N-term discrete Prony series to the DMA storage modulus master curve:
      E'(omega) = E_inf + sum_{i=1}^N E_i * (omega * tau_i)^2 / (1 + (omega * tau_i)^2)
    """
    freq, omega, E_prime = load_dma_data(mat)

    t_min = 1.0 / omega.max()
    t_max = 1.0 / omega.min()
    taus_0 = np.logspace(np.log10(t_min), np.log10(t_max), n_terms)

    def prony_eval(p, w):
        E_inf = p[0]
        E = p[1 : 1 + n_terms]
        tau = p[1 + n_terms : 1 + 2 * n_terms]
        pred = E_inf * np.ones_like(w)
        for ei, ti in zip(E, tau):
            w2t2 = (w * ti) ** 2
            pred += ei * (w2t2 / (1.0 + w2t2))
        return pred

    def resid(p):
        pred = prony_eval(p, omega)
        return np.log10(pred) - np.log10(E_prime)

    e_min, e_max = E_prime.min(), E_prime.max()
    dE = max(e_max - e_min, 1.0)
    x0 = [e_min] + [dE / n_terms] * n_terms + list(taus_0)
    lb = [1e-4] + [0.0] * n_terms + [t_min * 0.1] * n_terms
    ub = [1e4] + [1e4] * n_terms + [t_max * 10.0] * n_terms

    res = least_squares(resid, x0, bounds=(lb, ub), method="trf", ftol=1e-8, xtol=1e-8)
    p = res.x
    E_inf = p[0]
    E = p[1 : 1 + n_terms]
    taus = p[1 + n_terms : 1 + 2 * n_terms]
    E0 = E_inf + sum(E)
    gammas = E / E0

    idx = np.argsort(taus)
    gammas = gammas[idx]
    taus = taus[idx]
    E = E[idx]

    pred = prony_eval(p, omega)
    rmse_pct = 100 * np.sqrt(np.mean(((pred - E_prime) / E_prime) ** 2))

    return {
        "freq": freq,
        "omega": omega,
        "E_prime_data": E_prime,
        "E_prime_fit": pred,
        "E0": E0,
        "E_inf": E_inf,
        "gammas": gammas,
        "taus": taus,
        "g_inf": 1.0 - sum(gammas),
        "E_terms": E,
        "n_terms": n_terms,
        "rmse_pct": rmse_pct,
    }


def fit_dma_fractional(mat):
    """
    Fit Marmot's 5-mode continuous fractional power-law spectrum to the DMA storage modulus:
      E'(omega) = E0 * [ g_inf + sum_{m=0}^4 gamma_m * (omega * tau_m)^2 / (1 + (omega * tau_m)^2) ]
    where tau_m = tau0 * [0.01, 0.1, 1.0, 10.0, 100.0],
          weights_m = factors^(-alpha),
          gamma_m = (1 - g_inf) * weights_m / sum(weights_m).
    """
    freq, omega, E_prime = load_dma_data(mat)

    def model(p, w):
        E0, alpha, tau0, g_inf = p
        weights = FACTORS_FS ** (-alpha)
        gamma = (1.0 - g_inf) * (weights / np.sum(weights))
        taus = tau0 * FACTORS_FS
        pred = g_inf * np.ones_like(w)
        for g, t in zip(gamma, taus):
            w2t2 = (w * t) ** 2
            pred += g * (w2t2 / (1.0 + w2t2))
        return E0 * pred

    def resid(p):
        return np.log10(model(p, omega)) - np.log10(E_prime)

    e_max = E_prime.max()
    e_min = E_prime.min()
    x0 = [e_max, 0.3, 1.0, max(e_min / e_max, 0.01)]
    lb = [1e-2, 0.01, 1e-5, 0.001]
    ub = [1e5, 0.99, 1e4, 0.99]

    res = least_squares(resid, x0, bounds=(lb, ub), ftol=1e-9, xtol=1e-9)
    E0, alpha, tau0, g_inf = res.x
    pred = model(res.x, omega)
    rmse_pct = 100 * np.sqrt(np.mean(((pred - E_prime) / E_prime) ** 2))

    return {
        "freq": freq,
        "omega": omega,
        "E_prime_data": E_prime,
        "E_prime_fit": pred,
        "E0": E0,
        "E_inf": E0 * g_inf,
        "alpha": alpha,
        "tau0": tau0,
        "g_inf": g_inf,
        "rmse_pct": rmse_pct,
        "model_func": model,
        "params": res.x,
    }


def fit_mooney_rivlin_on_single_ramp(mat, gammas, taus, cal_rate_group="slow"):
    """
    Fit Mooney-Rivlin parameters (C1, C2) on the ramp phase of a single rate group,
    keeping DMA-derived (gammas, taus) fixed.
    """
    cfg = MATERIALS[mat]
    test_id = cfg["rate_groups"][cal_rate_group][0]
    df = load(cfg, mat, test_id)
    t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=40, n_hold=10)
    _, ramp_end = find_ramp_bounds(df["position_mm"])
    t_ramp_end = df["time_s"].iloc[ramp_end]
    ramp_mask = t_ds <= t_ramp_end

    t_ramp = t_ds[ramp_mask]
    peak_strain = df["strain"].iloc[ramp_end]
    mean_strain_rate = peak_strain / t_ramp_end
    strain_sim_ramp = mean_strain_rate * t_ramp
    stretch_ramp = 1.0 + strain_sim_ramp
    stress_ramp = stress_ds[ramp_mask]

    peak_stress = stress_ramp.max()

    def ramp_residuals(x):
        C1, C2 = x
        try:
            pred = simulate_once(t_ramp, stretch_ramp, "MooneyRivlin", C1, C2, 0.0, gammas, taus)
            return (pred - stress_ramp) / peak_stress
        except Exception:
            return np.ones_like(stress_ramp) * 10.0

    G_est = peak_stress / max(stretch_ramp.max() - 1.0, 1e-4) / 3.0
    x0 = [max(G_est / 2.0, 1.0), max(G_est / 4.0, 0.1)]
    lb = [1e-3, 0.0]
    ub = [1e4, 1e4]

    res = least_squares(ramp_residuals, x0, bounds=(lb, ub), method="trf", xtol=1e-6, ftol=1e-6)
    C1, C2 = res.x
    return C1, C2, test_id, t_ramp, stretch_ramp, stress_ramp


def fit_fs_neo_hooke_on_single_ramp(mat, alpha, tau0, g_inf, cal_rate_group="slow"):
    """
    Fit Neo-Hookean base parameter C1 for Fractional Schapery on the ramp phase of a single rate group,
    keeping DMA-derived (alpha, tau0, g_inf) fixed and physical bulk modulus from Poisson's ratio.
    """
    cfg = MATERIALS[mat]
    nu = POISSON_RATIOS.get(mat, 0.49)
    test_id = cfg["rate_groups"][cal_rate_group][0]
    df = load(cfg, mat, test_id)
    t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=40, n_hold=10)
    _, ramp_end = find_ramp_bounds(df["position_mm"])
    t_ramp_end = df["time_s"].iloc[ramp_end]
    ramp_mask = t_ds <= t_ramp_end

    t_ramp = t_ds[ramp_mask]
    peak_strain = df["strain"].iloc[ramp_end]
    mean_strain_rate = peak_strain / t_ramp_end
    strain_sim_ramp = mean_strain_rate * t_ramp
    stretch_ramp = 1.0 + strain_sim_ramp
    stress_ramp = stress_ds[ramp_mask]

    peak_stress = stress_ramp.max()

    def resid(x):
        C1 = x[0]
        try:
            pred = sim_fs(t_ramp, stretch_ramp, "NeoHooke", C1, 0.0, 0.0, alpha, tau0, g_inf, 0.0, 0.0, 0.0, nu=nu)
            return (pred - stress_ramp) / peak_stress
        except Exception:
            return np.ones_like(stress_ramp) * 10.0

    G_est = peak_stress / max(stretch_ramp.max() - 1.0, 1e-4) / 3.0
    res = least_squares(resid, [max(G_est, 1.0)], bounds=([1e-3], [1e4]), method="trf", xtol=1e-6, ftol=1e-6)
    return res.x[0], test_id, t_ramp, stretch_ramp, stress_ramp


def simulate_all_rates(mat, C1, C2, gammas, taus):
    """
    Simulate all rate groups across both ramp and hold using calibrated QLV parameters.
    """
    cfg = MATERIALS[mat]
    results = {}
    for rg, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, mat, test_id)
        t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=40, n_hold=60)
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t_ds <= t_ramp_end

        peak_strain = df["strain"].iloc[ramp_end]
        mean_strain_rate = peak_strain / t_ramp_end

        strain_sim = np.where(is_ramp, mean_strain_rate * t_ds, peak_strain)
        stretch_sim = 1.0 + strain_sim

        pred = simulate_once(t_ds, stretch_sim, "MooneyRivlin", C1, C2, 0.0, gammas, taus)

        rmse_total = np.sqrt(np.mean((pred - stress_ds) ** 2))
        rmse_pct = 100.0 * rmse_total / stress_ds.max()

        rmse_ramp = np.sqrt(np.mean((pred[is_ramp] - stress_ds[is_ramp]) ** 2))
        rmse_ramp_pct = 100.0 * rmse_ramp / stress_ds[is_ramp].max()

        rmse_hold = np.sqrt(np.mean((pred[~is_ramp] - stress_ds[~is_ramp]) ** 2))
        rmse_hold_pct = 100.0 * rmse_hold / stress_ds.max()

        results[rg] = {
            "test_id": test_id,
            "t": t_ds,
            "strain": strain_ds,
            "stretch": stretch_sim,
            "strain_sim": strain_sim,
            "stress_exp": stress_ds,
            "stress_pred": pred,
            "is_ramp": is_ramp,
            "peak_exp": stress_ds.max(),
            "peak_pred": pred.max(),
            "rmse_total": rmse_total,
            "rmse_pct": rmse_pct,
            "rmse_ramp": rmse_ramp,
            "rmse_ramp_pct": rmse_ramp_pct,
            "rmse_hold": rmse_hold,
            "rmse_hold_pct": rmse_hold_pct,
        }
    return results


def simulate_fs_all_rates(mat, C1, alpha, tau0, g_inf):
    """
    Simulate all rate groups across both ramp and hold using calibrated Fractional Schapery parameters.
    """
    cfg = MATERIALS[mat]
    nu = POISSON_RATIOS.get(mat, 0.49)
    results = {}
    for rg, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, mat, test_id)
        t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=40, n_hold=60)
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t_ds <= t_ramp_end
        peak_strain = df["strain"].iloc[ramp_end]
        mean_strain_rate = peak_strain / t_ramp_end
        strain_sim = np.where(is_ramp, mean_strain_rate * t_ds, peak_strain)
        stretch_sim = 1.0 + strain_sim

        pred = sim_fs(t_ds, stretch_sim, "NeoHooke", C1, 0.0, 0.0, alpha, tau0, g_inf, 0.0, 0.0, 0.0, nu=nu)
        rmse_total = np.sqrt(np.mean((pred - stress_ds) ** 2))
        rmse_pct = 100.0 * rmse_total / stress_ds.max()

        rmse_ramp = np.sqrt(np.mean((pred[is_ramp] - stress_ds[is_ramp]) ** 2))
        rmse_ramp_pct = 100.0 * rmse_ramp / stress_ds[is_ramp].max()

        rmse_hold = np.sqrt(np.mean((pred[~is_ramp] - stress_ds[~is_ramp]) ** 2))
        rmse_hold_pct = 100.0 * rmse_hold / stress_ds.max()

        results[rg] = {
            "test_id": test_id,
            "t": t_ds,
            "strain": strain_ds,
            "stretch": stretch_sim,
            "strain_sim": strain_sim,
            "stress_exp": stress_ds,
            "stress_pred": pred,
            "is_ramp": is_ramp,
            "peak_exp": stress_ds.max(),
            "peak_pred": pred.max(),
            "rmse_total": rmse_total,
            "rmse_pct": rmse_pct,
            "rmse_ramp": rmse_ramp,
            "rmse_ramp_pct": rmse_ramp_pct,
            "rmse_hold": rmse_hold,
            "rmse_hold_pct": rmse_hold_pct,
        }
    return results


def run_material_pipeline(mat, cal_rate_group="slow", n_terms=6):
    print(f"\n==================================================")
    print(f" Running DMA Cross-Modality Calibration: {mat}")
    print(f"==================================================")

    # 1. DMA Fits
    dma_fit = fit_dma_prony(mat, n_terms=n_terms)
    dma_fs = fit_dma_fractional(mat)
    gammas = dma_fit["gammas"]
    taus = dma_fit["taus"]

    print(f"\n[1] DMA Viscoelastic Spectrum Calibration (T_ref = 20 C):")
    print(f"    - Generalized Maxwell ({n_terms} Prony modes): E0 = {dma_fit['E0']:.1f} MPa, E_inf = {dma_fit['E_inf']:.1f} MPa (RMSE = {dma_fit['rmse_pct']:.2f}%)")
    print(f"    - Fractional Schapery (continuous): alpha = {dma_fs['alpha']:.3f}, tau0 = {dma_fs['tau0']:.2e} s, g_inf = {dma_fs['g_inf']:.3f} (RMSE = {dma_fs['rmse_pct']:.2f}%)")

    # 2. Single-Ramp Calibration
    print(f"\n[2] Single-Ramp Hyperelastic Calibration (Rate: '{cal_rate_group}'):")
    C1_qlv, C2_qlv, cal_test_id, t_ramp, str_ramp, s_ramp = fit_mooney_rivlin_on_single_ramp(
        mat, gammas, taus, cal_rate_group=cal_rate_group
    )
    C1_fs, _, _, _, _ = fit_fs_neo_hooke_on_single_ramp(
        mat, dma_fs["alpha"], dma_fs["tau0"], dma_fs["g_inf"], cal_rate_group=cal_rate_group
    )
    print(f"    - QLV (Mooney-Rivlin)      : C1 = {C1_qlv:.2f} MPa, C2 = {C2_qlv:.2f} MPa (E_inst = {6.0*(C1_qlv+C2_qlv):.1f} MPa)")
    print(f"    - Fract. Schapery (Neo-Hook): C1 = {C1_fs:.2f} MPa")

    # 3. Simulate All Rates
    print(f"\n[3] Blind Predictions & Evaluations Across Rates:")
    sim_results = simulate_all_rates(mat, C1_qlv, C2_qlv, gammas, taus)
    fs_sim_results = simulate_fs_all_rates(mat, C1_fs, dma_fs["alpha"], dma_fs["tau0"], dma_fs["g_inf"])

    for rg in ["very slow", "slow", "fast"]:
        is_cal = " [CALIBRATION RAMP]" if rg == cal_rate_group else " [BLIND PREDICTION]"
        q_e = sim_results[rg]["rmse_pct"]
        f_e = fs_sim_results[rg]["rmse_pct"]
        print(f"    Rate '{rg:9s}'{is_cal}: QLV RMSE = {q_e:5.1f}% | Fract. Schapery RMSE = {f_e:5.1f}%")

    joint_params = None
    baseline_csv = PARAMS_DIR / "qlv_marmot_params.csv"
    if baseline_csv.exists():
        bdf = pd.read_csv(baseline_csv)
        bdf_mat = bdf[(bdf["material"] == mat) & (bdf["hyperelastic_base"] == "MooneyRivlin")]
        if not bdf_mat.empty:
            row = bdf_mat.iloc[0]
            joint_params = {
                "C1": row["C1"],
                "C2": row["C2"],
                "gammas": [row["gamma1"], row["gamma2"], row["gamma3"]],
                "taus": [row["tau1"], row["tau2"], row["tau3"]],
            }

    joint_sim = None
    if joint_params is not None:
        joint_sim = simulate_all_rates(mat, joint_params["C1"], joint_params["C2"], joint_params["gammas"], joint_params["taus"])

    return {
        "mat": mat,
        "dma_fit": dma_fit,
        "dma_fs": dma_fs,
        "C1": C1_qlv,
        "C2": C2_qlv,
        "C1_fs": C1_fs,
        "sim_results": sim_results,
        "fs_sim_results": fs_sim_results,
        "joint_sim": joint_sim,
    }


def generate_paper_dma_qlv_grid(all_results, cal_rate_group="slow", n_terms=6):
    # Setup publication theme
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=0.88,
        rc={
            "lines.linewidth": 1.1,
            "text.usetex": True,
            "font.family": "cm",
            "legend.frameon": False,
            "legend.fontsize": "small",
        },
    )
    palette = sns.color_palette("colorblind")
    mat_colors = {
        "A0V100": palette[0],
        "A25V75": palette[1],
        "A50V50": palette[2],
        "A75V25": palette[3],
        "A100V0": palette[4],
    }
    mat_order = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]
    rate_groups = ["very slow", "slow", "fast"]

    fig, axes = plt.subplots(5, 4, figsize=(19 / 2.54, 19.5 / 2.54))

    for r, mat in enumerate(mat_order):
        res = all_results[mat]
        dma_p = res["dma_fit"]
        dma_fs = res["dma_fs"]
        color = mat_colors[mat]

        # Col 0: DMA Master Curve
        ax0 = axes[r, 0]
        w_dense = np.logspace(np.log10(dma_p["omega"].min()) - 0.5, np.log10(dma_p["omega"].max()) + 0.5, 200)
        freq_dense = w_dense / (2 * np.pi)

        # Generalized Maxwell Prony curve
        E_prony = dma_p["E_inf"] * np.ones_like(w_dense)
        for g, t in zip(dma_p["gammas"], dma_p["taus"]):
            w2t2 = (w_dense * t) ** 2
            E_prony += (g * dma_p["E0"]) * (w2t2 / (1.0 + w2t2))

        # Fractional Schapery curve
        E_fs = dma_fs["model_func"](dma_fs["params"], w_dense)

        ax0.loglog(dma_p["freq"], dma_p["E_prime_data"], "o", markersize=2.0, color="gray", alpha=0.55, zorder=1)
        ax0.loglog(freq_dense, E_prony, "-", color=color, linewidth=1.4, zorder=3)
        ax0.loglog(freq_dense, E_fs, "-.", color="black", linewidth=1.2, zorder=2)

        disp_mat = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}.get(mat, mat)
        ax0.set_ylabel(f"\\textbf{{{disp_mat}}}\nStorage $E'$ (MPa)", fontsize=8.0)
        if r == 4:
            ax0.set_xlabel("Reduced freq. $f_r$ (Hz)", fontsize=8.0)
        ax0.grid(True, alpha=0.3, linewidth=0.5)
        if r == 0:
            ax0.set_title(r"(a) DMA Master ($20^\circ$C)", fontsize=8.5, fontweight="bold")
        badge_text = f"$E_0={dma_p['E0']:.0f}$ MPa\n$\\alpha={dma_fs['alpha']:.3f}$"
        ax0.text(0.06, 0.08, badge_text, transform=ax0.transAxes, fontsize=6.3,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))

        # Max peak stress across rates for consistent y-limits
        max_peak = max(res["sim_results"][rg]["peak_exp"] for rg in rate_groups)
        y_max = 1.16 * max_peak

        # Cols 1, 2, 3: Rates
        for c_idx, rg in enumerate(rate_groups, start=1):
            ax = axes[r, c_idx]
            q_res = res["sim_results"][rg]
            fs_res = res["fs_sim_results"][rg]
            jsim = res["joint_sim"][rg] if res["joint_sim"] is not None else None

            # Measured test curve
            ax.plot(q_res["t"], q_res["stress_exp"], color="gray", linewidth=2.0, alpha=0.5, zorder=1)
            # DMA-calibrated QLV (Generalized Maxwell)
            ax.plot(q_res["t"], q_res["stress_pred"], "-", color=color, linewidth=1.3, zorder=4)
            # DMA-calibrated Fractional Schapery (Neo-Hookean)
            ax.plot(fs_res["t"], fs_res["stress_pred"], "-.", color="black", linewidth=1.1, zorder=3)
            # Joint-fit baseline
            if jsim is not None:
                ax.plot(jsim["t"], jsim["stress_pred"], ":", color="0.4", linewidth=0.9, zorder=2)

            ax.set_xscale("log")
            ax.set_xlim(0.2, 650)
            ax.set_ylim(0, y_max)
            ax.grid(True, alpha=0.3, linewidth=0.5)

            if c_idx == 1:
                ax.set_ylabel(r"Stress $\sigma$ (MPa)", fontsize=8.0)
            else:
                ax.tick_params(labelleft=False)

            if r == 4:
                ax.set_xlabel(r"Time $t$ (s)", fontsize=8.0)

            if r == 0:
                col_titles = [
                    r"(b) Very Slow ($0.017\,\mathrm{mm/s}$)",
                    r"(c) Slow ($0.167\,\mathrm{mm/s}$)",
                    r"(d) Fast ($1.67\,\mathrm{mm/s}$)"
                ]
                ax.set_title(col_titles[c_idx - 1], fontsize=8.5, fontweight="bold")

            # Annotation badge
            is_cal = (rg == cal_rate_group)
            tag_label = "Cal. Ramp" if is_cal else "Blind Pred."
            err_text = f"{tag_label}\nGM: {q_res['rmse_pct']:.1f}\\%\nFS: {fs_res['rmse_pct']:.1f}\\%"
            ax.text(0.95, 0.94, err_text, transform=ax.transAxes, fontsize=6.2,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))

    # Global legend (2 neat rows)
    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=2.0, alpha=0.5, label="Measured tensile test"),
        Line2D([0], [0], color="black", linewidth=1.3, label="DMA-calibrated Generalized Maxwell (Mooney--Rivlin)"),
        Line2D([0], [0], color="0.4", linestyle=":", linewidth=1.0, label="Joint 3-rate fitted baseline"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=1.1, label="DMA-calibrated Fractional Schapery (Neo-Hookean)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.998), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    # Save to paper figures and analysis
    paper_dir = Path(__file__).resolve().parents[4] / "paper"
    if not paper_dir.is_dir():
        paper_dir = Path(__file__).resolve().parents[3] / "paper"
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_pdf = fig_dir / "dma_qlv_validation_grid.pdf"
    fig_png = fig_dir / "dma_qlv_validation_grid.png"
    fig.savefig(fig_pdf, dpi=300)
    fig.savefig(fig_png, dpi=300)
    fig.savefig(OUTPUT_DIR / "dma_qlv_validation_grid.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "dma_qlv_validation_grid.png", dpi=300)
    plt.close(fig)
    print(f"\n[Saved Grid Figure]: {fig_pdf}")

    # Build LaTeX Table
    table_path = paper_dir / "dma_qlv_summary_table.tex"
    table_tex = [
        "% Auto-generated summary table for DMA-calibrated constitutive models",
        "\\begin{table}[htpb!]",
        "    \\centering",
        "    \\small",
        "    \\caption{Cross-modality parameter calibration: constitutive parameters derived from small-strain DMA master curves ($T_{\\mathrm{ref}} = \\SI{20}{\\celsius}$) and calibrated on a single slow loading ramp ($0.167\\,\\si{\\milli\\meter\\per\\second}$). Prediction errors (RMSE, \\% of peak stress) are reported for the calibration ramp and blind forward predictions across rates for both Generalized Maxwell (Mooney--Rivlin base, 6 Prony modes) and Fractional Schapery (Neo-Hookean base, 5-decade fractional kernel) models.}",
        "    \\label{tab:dma-qlv-summary}",
        "    \\begin{tabular}{llcccc}",
        "        \\toprule",
        "        Blend & Constitutive Model & DMA Spectrum \\& Calibrated Ramp Modulus & Very Slow & Slow (Cal Ramp) & Fast \\\\",
        "              &                    &                                         & (Blind, \\%) & (Ramp, \\%) & (Blind, \\%) \\\\",
        "        \\midrule",
    ]

    for i, mat in enumerate(mat_order):
        res = all_results[mat]
        dp = res["dma_fit"]
        df = res["dma_fs"]
        disp_mat = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}.get(mat, mat)

        # GM row
        gm_vs = res["sim_results"]["very slow"]["rmse_pct"]
        gm_sl = res["sim_results"]["slow"]["rmse_pct"]
        gm_fs = res["sim_results"]["fast"]["rmse_pct"]
        gm_param = f"$E_0={dp['E0']:.1f},\\, C_1={res['C1']:.2f},\\, C_2={res['C2']:.2f}\\,\\text{{MPa}}$"

        # FS row
        fs_vs = res["fs_sim_results"]["very slow"]["rmse_pct"]
        fs_sl = res["fs_sim_results"]["slow"]["rmse_pct"]
        fs_fs = res["fs_sim_results"]["fast"]["rmse_pct"]
        if df["tau0"] < 0.1:
            m, exp = f"{df['tau0']:.2e}".split("e")
            tau_str = f"{m}\times 10^{{{int(exp)}}}"
        else:
            tau_str = f"{df['tau0']:.1f}"
        fs_param = f"$\\alpha={df['alpha']:.3f},\\, \\tau_0={tau_str}\\,\\text{{s}},\\, C_1={res['C1_fs']:.2f}\\,\\text{{MPa}}$"

        table_tex.append(
            f"        \\multirow{{2}}{{*}}{{{disp_mat}}} & Generalized Maxwell & {gm_param} & {gm_vs:.1f}\\% & {gm_sl:.1f}\\% & {gm_fs:.1f}\\% \\\\"
        )
        table_tex.append(
            f"                             & Fractional Schapery & {fs_param} & {fs_vs:.1f}\\% & {fs_sl:.1f}\\% & {fs_fs:.1f}\\% \\\\"
        )
        if i < len(mat_order) - 1:
            table_tex.append("        \\midrule")

    table_tex.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ])

    with open(table_path, "w") as f:
        f.write("\n".join(table_tex))
    print(f"[Saved Summary Table]: {table_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict tensile stress relaxation from DMA master curves")
    parser.add_argument("--material", default="all", choices=list(MATERIALS.keys()) + ["all"],
                        help="Material to run (default: all)")
    parser.add_argument("--ramp-rate", default="slow", choices=["very slow", "slow", "fast"],
                        help="Rate group used to calibrate hyperelastic base (default: slow)")
    parser.add_argument("--n-terms", type=int, default=6,
                        help="Number of Prony series terms (default: 6)")
    args = parser.parse_args()

    materials = list(MATERIALS.keys()) if args.material == "all" else [args.material]

    all_summary = []
    all_results = {}
    for mat in materials:
        res = run_material_pipeline(mat, cal_rate_group=args.ramp_rate, n_terms=args.n_terms)
        all_results[mat] = res
        for rg, sres in res["sim_results"].items():
            all_summary.append({
                "material": mat,
                "model": "GeneralizedMaxwell",
                "n_terms": args.n_terms,
                "cal_ramp_rate": args.ramp_rate,
                "evaluated_rate": rg,
                "is_cal_ramp": (rg == args.ramp_rate),
                "peak_exp_mpa": sres["peak_exp"],
                "peak_pred_mpa": sres["peak_pred"],
                "rmse_total_mpa": sres["rmse_total"],
                "rmse_total_pct": sres["rmse_pct"],
                "rmse_hold_pct": sres["rmse_hold_pct"],
            })
        for rg, fres in res["fs_sim_results"].items():
            all_summary.append({
                "material": mat,
                "model": "FractionalSchapery",
                "n_terms": 5,
                "cal_ramp_rate": args.ramp_rate,
                "evaluated_rate": rg,
                "is_cal_ramp": (rg == args.ramp_rate),
                "peak_exp_mpa": fres["peak_exp"],
                "peak_pred_mpa": fres["peak_pred"],
                "rmse_total_mpa": fres["rmse_total"],
                "rmse_total_pct": fres["rmse_pct"],
                "rmse_hold_pct": fres["rmse_hold_pct"],
            })

    summary_df = pd.DataFrame(all_summary)
    summary_csv = OUTPUT_DIR / "qlv_dma_prediction_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table to: {summary_csv}")

    if args.material == "all":
        generate_paper_dma_qlv_grid(all_results, cal_rate_group=args.ramp_rate, n_terms=args.n_terms)


if __name__ == "__main__":
    main()
