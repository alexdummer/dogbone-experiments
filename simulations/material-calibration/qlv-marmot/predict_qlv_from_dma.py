#!/usr/bin/env python3
"""
predict_qlv_from_dma.py

Investigate whether multi-rate tensile stress-relaxation tests can be predicted
purely by:
1. Extracting the linear viscoelastic relaxation spectrum (Prony series) from DMA master curves at 20 C.
2. Fitting only the instantaneous Mooney-Rivlin parameters (C1, C2) from a single loading ramp (e.g. fast rate).
3. Performing blind forward predictions for all other strain rates and the full 10-minute relaxation hold.

Uses Marmot's CompressibleFiniteStrainLinearViscoelasticity material model with N (default 6) Maxwell elements.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Add local paths for helper imports
REPO_ROOT = Path(__file__).resolve().parents[3]
SYS_PATHS = [
    REPO_ROOT / "simulations" / "material-calibration" / "qlv-marmot",
    REPO_ROOT / "simulations" / "material-calibration" / "1d-qlv",
    REPO_ROOT / "simulations" / "material-calibration" / "prf",
    REPO_ROOT / "analysis",
]
for p in SYS_PATHS:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import marmot
from fit_prf_model import downsample_for_fit
from fit_qlv_model import MATERIALS, load, find_ramp_bounds, mean_strain_rates
import fit_qlv_marmot_model as qm
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
        step.dTMin = dt / 50
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


def fit_dma_prony(mat, n_terms=6):
    """
    Fit an N-term discrete Prony series to the DMA storage modulus master curve:
      E'(omega) = E_inf + sum_{i=1}^N E_i * (omega * tau_i)^2 / (1 + (omega * tau_i)^2)
    using adaptive logarithmic windowing over the observation domain across qualifying replicates.
    """
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

    # Window covering relevant room-temperature frequency range
    f_min = max(df["reduced_freq_hz"].min(), 1e-5)
    f_max = min(max(df["reduced_freq_hz"].min() * 1e7, 1e2), 1e4)
    sub = df[(df["reduced_freq_hz"] >= f_min) & (df["reduced_freq_hz"] <= f_max)].copy()
    if len(sub) < 3 * n_terms:
        sub = df.iloc[:min(len(df), 50)].copy()

    sub = sub.sort_values("reduced_freq_hz")
    freq = sub["reduced_freq_hz"].to_numpy()
    omega = 2 * np.pi * freq
    E_prime = sub["E_prime_MPa"].to_numpy()

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

    fit_info = {
        "freq": freq,
        "omega": omega,
        "E_prime_data": E_prime,
        "E_prime_fit": prony_eval(p, omega),
        "E0": E0,
        "E_inf": E_inf,
        "gammas": gammas,
        "taus": taus,
        "g_inf": 1.0 - sum(gammas),
        "E_terms": E,
        "n_terms": n_terms,
    }
    return fit_info


def fit_mooney_rivlin_on_single_ramp(mat, gammas, taus, cal_rate_group="fast"):
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
    stretch_ramp = 1.0 + strain_ds[ramp_mask]
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
    lb = [1e-3, -1e3]
    ub = [1e4, 1e4]

    res = least_squares(ramp_residuals, x0, bounds=(lb, ub), method="trf", xtol=1e-6, ftol=1e-6)
    C1, C2 = res.x
    return C1, C2, test_id, t_ramp, stretch_ramp, stress_ramp


def simulate_all_rates(mat, C1, C2, gammas, taus):
    """
    Simulate all rate groups across both ramp and hold using the calibrated parameters.
    """
    cfg = MATERIALS[mat]
    results = {}
    for rg, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, mat, test_id)
        t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=40, n_hold=60)
        stretch_ds = 1.0 + strain_ds
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t_ds <= t_ramp_end

        pred = simulate_once(t_ds, stretch_ds, "MooneyRivlin", C1, C2, 0.0, gammas, taus)

        rmse_total = np.sqrt(np.mean((pred - stress_ds) ** 2))
        rmse_pct = 100.0 * rmse_total / stress_ds.max()

        rmse_ramp = np.sqrt(np.mean((pred[is_ramp] - stress_ds[is_ramp]) ** 2))
        rmse_ramp_pct = 100.0 * rmse_ramp / stress_ds[is_ramp].max()

        rmse_hold = np.sqrt(np.mean((pred[~is_ramp] - stress_ds[~is_ramp]) ** 2))
        rmse_hold_pct = 100.0 * rmse_hold / stress_ds.max()

        # Generate smooth ramp simulation on clean linear strain for clean plotting
        t_ramp_clean = np.linspace(0, t_ramp_end, 50)
        # Use average strain rate over ramp
        ramp_strain_peak = strain_ds[is_ramp].max()
        strain_clean = np.linspace(0, ramp_strain_peak, 50)
        stretch_clean = 1.0 + strain_clean
        pred_clean = simulate_once(t_ramp_clean, stretch_clean, "MooneyRivlin", C1, C2, 0.0, gammas, taus)

        results[rg] = {
            "test_id": test_id,
            "t": t_ds,
            "strain": strain_ds,
            "stretch": stretch_ds,
            "stress_exp": stress_ds,
            "stress_pred": pred,
            "is_ramp": is_ramp,
            "strain_clean": strain_clean,
            "pred_clean": pred_clean,
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


def run_material_pipeline(mat, cal_rate_group="fast", n_terms=6):
    print(f"\n==================================================")
    print(f" Running DMA -> QLV Prediction Pipeline for {mat} ({n_terms} Prony terms)")
    print(f"==================================================")

    dma_fit = fit_dma_prony(mat, n_terms=n_terms)
    gammas = dma_fit["gammas"]
    taus = dma_fit["taus"]
    print(f"\n[1] DMA Prony Calibration (T_ref = 20 C, {n_terms} terms):")
    print(f"    E0 = {dma_fit['E0']:.1f} MPa, E_inf = {dma_fit['E_inf']:.1f} MPa (g_inf = {dma_fit['g_inf']:.3f})")
    for i, (g, t, e) in enumerate(zip(gammas, taus, dma_fit["E_terms"]), 1):
        print(f"    Mode {i}: tau = {t:10.4e} s, gamma = {g:6.4f} (E = {e:7.1f} MPa)")

    print(f"\n[2] Single-Ramp Mooney-Rivlin Calibration (Rate: '{cal_rate_group}'):")
    C1, C2, cal_test_id, t_ramp, str_ramp, s_ramp = fit_mooney_rivlin_on_single_ramp(
        mat, gammas, taus, cal_rate_group=cal_rate_group
    )
    print(f"    Calibrated test: {mat}-{cal_test_id} ramp (duration = {t_ramp[-1]:.3f} s, peak = {s_ramp.max():.2f} MPa)")
    print(f"    Fitted parameters: C1 = {C1:.3f} MPa, C2 = {C2:.3f} MPa (K = {100.0*C1:.1f} MPa)")
    print(f"    Instantaneous Young's modulus E_inst = 6*(C1+C2) = {6.0*(C1+C2):.1f} MPa")

    print(f"\n[3] Blind Forward Prediction Across All Rate Groups:")
    sim_results = simulate_all_rates(mat, C1, C2, gammas, taus)
    for rg, res in sim_results.items():
        is_cal = " [CALIBRATION RAMP]" if rg == cal_rate_group else " [BLIND PREDICTION]"
        print(f"    Rate '{rg:9s}' (test {res['test_id']:2d}){is_cal}:")
        print(f"      Peak stress: exp = {res['peak_exp']:6.2f} MPa, pred = {res['peak_pred']:6.2f} MPa")
        print(f"      RMSE total : {res['rmse_total']:6.3f} MPa ({res['rmse_pct']:5.2f}% of peak)")
        print(f"      RMSE hold  : {res['rmse_hold']:6.3f} MPa ({res['rmse_hold_pct']:5.2f}% of peak)")

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

    plot_results(mat, dma_fit, sim_results, joint_sim, cal_rate_group, C1, C2)

    return {
        "mat": mat,
        "C1": C1,
        "C2": C2,
        "gammas": gammas,
        "taus": taus,
        "sim_results": sim_results,
    }


def plot_results(mat, dma_fit, sim_results, joint_sim, cal_rate_group, C1, C2):
    fig, axes = plt.subplots(1, 3, figsize=(22 / 2.54, 7.5 / 2.54))

    # --- Panel (a): DMA Master Curve & Prony Fit ---
    ax_dma = axes[0]
    w_dense = np.logspace(np.log10(dma_fit["omega"].min()) - 0.5, np.log10(dma_fit["omega"].max()) + 0.5, 200)
    freq_dense = w_dense / (2 * np.pi)
    E_dense = dma_fit["E_inf"] * np.ones_like(w_dense)
    for g, t in zip(dma_fit["gammas"], dma_fit["taus"]):
        w2t2 = (w_dense * t) ** 2
        E_dense += (g * dma_fit["E0"]) * (w2t2 / (1.0 + w2t2))

    ax_dma.loglog(dma_fit["freq"], dma_fit["E_prime_data"], "o", markersize=3.5, color="gray", alpha=0.7, label="DMA data ($20^\\circ$C)")
    ax_dma.loglog(freq_dense, E_dense, "-", color="black", linewidth=1.5, label=f"{dma_fit['n_terms']}-term Prony fit")
    ax_dma.set_xlabel("Reduced frequency $f_r$ (Hz)")
    ax_dma.set_ylabel("Storage modulus $E'$ (MPa)")
    ax_dma.set_title(f"(a) DMA Master Curve: {mat}", fontsize=9)
    ax_dma.legend(loc="lower right", fontsize=7.5)

    # --- Panel (b): Stress-Strain (Ramp Phase) ---
    ax_ss = axes[1]
    rate_colors = {"very slow": colors[0], "slow": colors[1], "fast": colors[2]}
    for rg in ["very slow", "slow", "fast"]:
        res = sim_results[rg]
        ramp_mask = res["is_ramp"]
        c = rate_colors[rg]
        # Measured experimental points
        ax_ss.plot(res["strain"][ramp_mask] * 100, res["stress_exp"][ramp_mask], "o", markersize=2.5, color=c, alpha=0.45)
        # DMA QLV prediction along smooth linear strain trajectory
        label = f"{rg} (pred)" if rg != cal_rate_group else f"{rg} (cal ramp)"
        lw = 1.8 if rg == cal_rate_group else 1.3
        ax_ss.plot(res["strain_clean"] * 100, res["pred_clean"], linestyle="-", color=c, linewidth=lw, label=label)
        # Baseline joint fit if available
        if joint_sim is not None:
            ax_ss.plot(joint_sim[rg]["strain_clean"] * 100, joint_sim[rg]["pred_clean"], ":", color=c, linewidth=1.0)

    ax_ss.set_xlabel("Engineering strain $\\varepsilon$ (\\%)")
    ax_ss.set_ylabel("Engineering stress $\\sigma$ (MPa)")
    ax_ss.set_title(f"(b) Tensile Ramps: {mat}", fontsize=9)
    ax_ss.legend(loc="upper left", fontsize=7.0)

    # --- Panel (c): Stress-Time (Ramp + Hold Relaxation) ---
    ax_st = axes[2]
    for rg in ["very slow", "slow", "fast"]:
        res = sim_results[rg]
        c = rate_colors[rg]
        ax_st.plot(res["t"], res["stress_exp"], "o", markersize=2.0, color=c, alpha=0.4)
        ax_st.plot(res["t"], res["stress_pred"], "-", color=c, linewidth=1.4)
        if joint_sim is not None:
            ax_st.plot(joint_sim[rg]["t"], joint_sim[rg]["stress_pred"], ":", color=c, linewidth=1.0)

    custom_lines = [
        plt.Line2D([0], [0], color="black", marker="o", markersize=3, linestyle="none", alpha=0.6, label="Measured test"),
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="DMA QLV prediction"),
    ]
    if joint_sim is not None:
        custom_lines.append(plt.Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Joint-fit baseline"))

    ax_st.set_xscale("log")
    ax_st.set_xlabel("Time $t$ (s)")
    ax_st.set_ylabel("Engineering stress $\\sigma$ (MPa)")
    ax_st.set_title(f"(c) Stress Relaxation (600 s): {mat}", fontsize=9)
    ax_st.legend(handles=custom_lines, loc="upper right", fontsize=7.0)

    fig.tight_layout()
    out_pdf = OUTPUT_DIR / f"qlv_dma_prediction_{mat}.pdf"
    out_png = OUTPUT_DIR / f"qlv_dma_prediction_{mat}.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"\n[Saved figure]: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Predict tensile stress relaxation from DMA master curves")
    parser.add_argument("--material", default="A25V75", choices=list(MATERIALS.keys()) + ["all"],
                        help="Material to run (default: A25V75)")
    parser.add_argument("--ramp-rate", default="fast", choices=["very slow", "slow", "fast"],
                        help="Rate group used to calibrate Mooney-Rivlin (default: fast)")
    parser.add_argument("--n-terms", type=int, default=6,
                        help="Number of Prony series terms (default: 6)")
    args = parser.parse_args()

    materials = list(MATERIALS.keys()) if args.material == "all" else [args.material]

    all_summary = []
    for mat in materials:
        res = run_material_pipeline(mat, cal_rate_group=args.ramp_rate, n_terms=args.n_terms)
        for rg, sres in res["sim_results"].items():
            all_summary.append({
                "material": mat,
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

    summary_df = pd.DataFrame(all_summary)
    summary_csv = OUTPUT_DIR / "qlv_dma_prediction_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table to: {summary_csv}")


if __name__ == "__main__":
    main()
