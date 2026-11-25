#!/usr/bin/env python3
"""
calibrate_lattice_dma_ramp.py

Calibrate the hyperelastic base parameter (shear modulus mu, with bulk modulus kappa
determined from physical Poisson's ratio nu) of the DMA-derived Generalized Maxwell model
such that the loading ramp in the architected lattice dogbone specimens is accurately captured.

Methodology:
- Fixed relaxation spectra: 6 Prony modes (gamma_i, tau_i) identified from small-strain DMA master curves at 20 C.
- Bounded gradient-free surrogate optimization:
  Because the displacement field u(X, t) is strictly invariant to mu in compressible neo-Hookean
  linear viscoelasticity under displacement boundary conditions with fixed nu, the reaction force
  scales linearly: F_sim(u; mu) = (mu / mu_0) * F_sim(u; mu_0).
  Thus, a single 3D FE evaluation at mu_solid yields the exact optimal mu_lattice minimizing RMSE
  against the experimental lattice loading ramp.
- High-fidelity single-ramp finite element simulations using EdelweissFE with all-hex mesh.
- Results saved to CSV, summary table generated, and publication-ready validation figure plotted.
"""

import os
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parents[2]
FE_TENSION_DIR = Path(__file__).resolve().parent
MESH_PATH = FE_TENSION_DIR / "mesh" / "lattice_mesh.inc"
EXP_DIR = REPO_ROOT / "data" / "tensile-lattice" / "cleaned"
OUTPUT_DIR = FE_TENSION_DIR / "lattice_dma_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS = REPO_ROOT.parent / "paper" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)
PAPER_DIR = REPO_ROOT.parent / "paper"

EDELWEISS_BIN = "/home/alex/miniforge3/envs/edelweissfe/bin/edelweissfe"

MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]
DISPLAY_NAMES = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}

POISSON_RATIOS = {
    "A0V100": 0.350,
    "A25V75": 0.385,
    "A50V50": 0.420,
    "A75V25": 0.455,
    "A100V0": 0.490,
}

# Solid-dogbone calibrated shear modulus mu_solid (from DMA cross-modality in Section 4.3)
MU_SOLID = {
    "A0V100": 1582.34,
    "A25V75": 798.78,
    "A50V50": 333.04,
    "A75V25": 199.07,
    "A100V0": 147.09,
}

# 6-term DMA Prony series (gammas, taus)
DMA_PRONY = {
    "A0V100": {
        "gammas": [0.3972, 0.0431, 0.0590, 0.0888, 0.0932, 0.0673],
        "taus":   [0.0007, 0.1188, 3.3795, 68.4215, 1072.5914, 6601.9834]
    },
    "A25V75": {
        "gammas": [0.1120, 0.1500, 0.1775, 0.1846, 0.1399, 0.1284],
        "taus":   [0.0077, 0.1524, 3.2387, 56.5182, 672.1217, 6809.3039]
    },
    "A50V50": {
        "gammas": [0.2729, 0.2250, 0.1919, 0.1415, 0.0847, 0.0482],
        "taus":   [0.0111, 0.1943, 2.4467, 30.8928, 333.5006, 3315.0603]
    },
    "A75V25": {
        "gammas": [0.5322, 0.2546, 0.1159, 0.0505, 0.0233, 0.0099],
        "taus":   [0.0089, 0.1185, 1.4076, 17.0647, 202.2458, 2302.4647]
    },
    "A100V0": {
        "gammas": [0.4448, 0.3766, 0.1499, 0.0211, 0.0041, 0.0014],
        "taus":   [0.00003, 0.0001, 0.0006, 0.0040, 0.0391, 0.5279]
    }
}

# Specimen protocol (ramp time in s, octant grip displacement in mm)
RAMP_PROTOCOLS = {
    "A0V100": {"t_ramp": 2.958,  "u_octant": 0.246503},
    "A25V75": {"t_ramp": 5.934,  "u_octant": 0.494208},
    "A50V50": {"t_ramp": 11.871, "u_octant": 0.989416},
    "A75V25": {"t_ramp": 29.722, "u_octant": 2.474240},
    "A100V0": {"t_ramp": 11.908, "u_octant": 9.898960},
}


def bulk_modulus(mu, nu):
    return 2.0 * mu * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))


def load_experimental_ramp(mat):
    """Load experimental replicates and compute mean loading ramp curve."""
    reps = []
    for r in [1, 2, 3]:
        p = EXP_DIR / f"{mat}-{r}_cleaned.csv"
        if p.exists():
            df = pd.read_csv(p)
            u = df["position_mm"].values
            F = -df["force_N"].values
            idx_peak = np.argmax(F)
            reps.append((u[:idx_peak + 1], F[:idx_peak + 1]))

    proto = RAMP_PROTOCOLS[mat]
    u_max = 2.0 * proto["u_octant"]
    u_grid = np.linspace(0, u_max, 50)

    f_interp = []
    for u, F in reps:
        interp = interp1d(u, F, bounds_error=False, fill_value=(0.0, F[-1]))
        f_interp.append(interp(u_grid))

    mean_F = np.mean(f_interp, axis=0)
    std_F = np.std(f_interp, axis=0)
    return u_grid, mean_F, std_F, reps


def run_fe_ramp(mat, mu, job_dir=None, n_inc=10, n_threads=4):
    """Run single-step ramp FE simulation in EdelweissFE and return (u, F)."""
    nu = POISSON_RATIOS[mat]
    kappa = bulk_modulus(mu, nu)
    proto = RAMP_PROTOCOLS[mat]
    t_ramp = proto["t_ramp"]
    u_oct = proto["u_octant"]
    prony = DMA_PRONY[mat]

    if job_dir is None:
        job_dir = OUTPUT_DIR / f"run_{mat}_{mu:.2f}"
    job_dir.mkdir(parents=True, exist_ok=True)

    f_csv = job_dir / "gripForce.csv"
    d_csv = job_dir / "gripDisp.csv"

    # Reuse existing result if already computed
    if f_csv.exists() and d_csv.exists() and os.path.getsize(f_csv) > 200:
        try:
            f_data = np.loadtxt(f_csv)
            d_data = np.loadtxt(d_csv)
            t_uniq, idx = np.unique(d_data[:, 0], return_index=True)
            u_full = 2.0 * d_data[idx, 1]
            F_full = 4.0 * f_data[idx, 1]
            if len(u_full) >= n_inc:
                print(f"[{mat}] Reusing existing results in {job_dir.name}")
                return u_full, F_full
        except Exception:
            pass

    inp_lines = [
        f"*include, input={MESH_PATH}",
        "",
        f"*material, name=CompressibleFiniteStrainLinearViscoelasticity, id=mat_{mat}",
        "0, 1,",
        f"{mu:.6f}, {kappa:.6f},",
        f"{len(prony['gammas'])},"
    ]
    for g, t in zip(prony["gammas"], prony["taus"]):
        inp_lines.append(f"{g:.6f}, {t:.6e},")

    inc_frac = 1.0 / n_inc
    inp_lines.extend([
        "",
        f"*section, name=sec1, material=mat_{mat}, type=solid",
        "all",
        "",
        f"*job, name={mat}_ramp, domain=3d",
        "*solver, solver=NISTParallel, name=theSolver",
        "",
        "*fieldOutput",
        ">>perNode, name=gripDisp, nSet=movingGrip, field=displacement, result=U, f(x)='np.mean(x[:,0])', saveHistory=True, export=gripDisp",
        ">>perNode, name=gripForce, nSet=movingGrip, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=gripForce",
        "",
        "*step, solver=theSolver",
        f"startInc={inc_frac:.4f}, maxInc={inc_frac:.4f}, minInc=1e-4, maxNumInc=1000, maxIter=25, stepLength={t_ramp:.4f}",
        ">>options, name=theSolver, extrapolation=linear, linsolver=pardiso",
        ">>dirichlet, name=xsym, nSet=xSymmetry, field=displacement, 1=0",
        ">>dirichlet, name=ysym, nSet=ySymmetry, field=displacement, 2=0",
        ">>dirichlet, name=zsym, nSet=zSymmetry, field=displacement, 3=0",
        ">>dirichlet, name=fixmove, nSet=movingGrip, field=displacement, 2=0, 3=0",
        f">>dirichlet, name=ramp, nSet=movingGrip, field=displacement, 1={u_oct:.6f}",
    ])

    inp_path = job_dir / f"{mat}_ramp.inp"
    with open(inp_path, "w") as f:
        f.write("\n".join(inp_lines) + "\n")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)

    res = subprocess.run([EDELWEISS_BIN, "--quiet", "--noplot", str(inp_path)],
                         cwd=str(job_dir), env=env, capture_output=True, text=True)

    if not (f_csv.exists() and d_csv.exists()):
        raise RuntimeError(f"EdelweissFE failed for {mat} with mu={mu}: {res.stderr}")

    f_data = np.loadtxt(f_csv)
    d_data = np.loadtxt(d_csv)
    t_uniq, idx = np.unique(d_data[:, 0], return_index=True)
    u_full = 2.0 * d_data[idx, 1]
    F_full = 4.0 * f_data[idx, 1]
    return u_full, F_full


def calibrate_blend(mat, run_confirmation=True, n_threads=4):
    print(f"\n==========================================")
    print(f" Calibrating Lattice Ramp for: {mat}")
    print(f"==========================================")

    u_exp, F_exp, std_exp, reps = load_experimental_ramp(mat)
    F_peak_exp = np.max(F_exp)
    mu_solid = MU_SOLID[mat]

    print(f"[{mat}] Experimental Peak Force: {F_peak_exp:.2f} N")
    print(f"[{mat}] Solid-calibrated Modulus mu_solid: {mu_solid:.2f} MPa")

    # Step 1: Trial simulation at mu_solid
    dir_solid = OUTPUT_DIR / f"{mat}_solid_eval"
    print(f"[{mat}] Running FE simulation at mu_solid = {mu_solid:.2f} MPa...")
    u_sim_solid, F_sim_solid = run_fe_ramp(mat, mu_solid, job_dir=dir_solid, n_inc=10, n_threads=n_threads)

    interp_solid = interp1d(u_sim_solid, F_sim_solid, bounds_error=False, fill_value=(0.0, F_sim_solid[-1]))
    F_solid_on_exp = interp_solid(u_exp)
    rmse_solid = np.sqrt(np.mean((F_solid_on_exp - F_exp)**2))
    rmse_solid_pct = 100.0 * rmse_solid / F_peak_exp
    peak_solid = np.max(F_sim_solid)
    peak_err_solid_pct = 100.0 * (peak_solid - F_peak_exp) / F_peak_exp

    print(f"[{mat}] Solid Baseline: Peak Force = {peak_solid:.2f} N (Error: {peak_err_solid_pct:+.1f}%), Ramp RMSE = {rmse_solid:.2f} N ({rmse_solid_pct:.2f}%)")

    # Gradient-free optimal mu: F_sim(u; mu) = (mu / mu_solid) * F_sim(u; mu_solid)
    mu_opt = mu_solid * np.sum(F_solid_on_exp * F_exp) / np.sum(F_solid_on_exp**2)
    print(f"[{mat}] Optimal Lattice Modulus: mu_lattice = {mu_opt:.2f} MPa (Ratio mu_lattice/mu_solid = {mu_opt/mu_solid:.3f})")

    if run_confirmation:
        dir_opt = OUTPUT_DIR / f"{mat}_opt_eval"
        print(f"[{mat}] Running confirmation FE simulation at mu_lattice = {mu_opt:.2f} MPa...")
        u_sim_opt, F_sim_opt = run_fe_ramp(mat, mu_opt, job_dir=dir_opt, n_inc=10, n_threads=n_threads)
    else:
        u_sim_opt = u_sim_solid
        F_sim_opt = F_sim_solid * (mu_opt / mu_solid)

    interp_opt = interp1d(u_sim_opt, F_sim_opt, bounds_error=False, fill_value=(0.0, F_sim_opt[-1]))
    F_opt_on_exp = interp_opt(u_exp)
    rmse_opt = np.sqrt(np.mean((F_opt_on_exp - F_exp)**2))
    rmse_opt_pct = 100.0 * rmse_opt / F_peak_exp
    peak_opt = np.max(F_sim_opt)
    peak_err_opt_pct = 100.0 * (peak_opt - F_peak_exp) / F_peak_exp

    print(f"[{mat}] Calibrated Lattice: Peak Force = {peak_opt:.2f} N (Error: {peak_err_opt_pct:+.1f}%), Ramp RMSE = {rmse_opt:.2f} N ({rmse_opt_pct:.2f}%)")

    return {
        "material": mat,
        "nu": POISSON_RATIOS[mat],
        "F_peak_exp": F_peak_exp,
        "mu_solid": mu_solid,
        "peak_solid": peak_solid,
        "peak_err_solid_pct": peak_err_solid_pct,
        "rmse_solid": rmse_solid,
        "rmse_solid_pct": rmse_solid_pct,
        "mu_lattice": mu_opt,
        "ratio_lattice_solid": mu_opt / mu_solid,
        "peak_lattice": peak_opt,
        "peak_err_lattice_pct": peak_err_opt_pct,
        "rmse_lattice": rmse_opt,
        "rmse_lattice_pct": rmse_opt_pct,
        "u_exp": u_exp,
        "F_exp": F_exp,
        "std_exp": std_exp,
        "reps": reps,
        "u_sim_solid": u_sim_solid,
        "F_sim_solid": F_sim_solid,
        "u_sim_opt": u_sim_opt,
        "F_sim_opt": F_sim_opt,
    }


def plot_calibration_results(results):
    mats = [m for m in MATERIALS if m in results]
    n_mats = len(mats)
    fig, axes = plt.subplots(1, n_mats, figsize=(3.2 * n_mats, 3.4), sharex=False)
    if n_mats == 1:
        axes = [axes]

    colors_mat = {
        "A0V100": "#2b5c8f",
        "A25V75": "#3b8b88",
        "A50V50": "#d95f02",
        "A75V25": "#7570b3",
        "A100V0": "#e7298a",
    }

    for ax, mat in zip(axes, mats):
        res = results[mat]
        disp_name = DISPLAY_NAMES[mat]
        c = colors_mat[mat]

        # Plot individual replicates
        for u_r, f_r in res["reps"]:
            ax.plot(u_r, f_r, color="grey", alpha=0.35, lw=1.0)

        # Plot mean experiment
        ax.plot(res["u_exp"], res["F_exp"], color="black", lw=1.8, label=r"Exp. mean ($n=3$)")
        ax.fill_between(res["u_exp"], res["F_exp"] - res["std_exp"], res["F_exp"] + res["std_exp"],
                        color="black", alpha=0.12)

        # Plot solid-calibrated FE
        ax.plot(res["u_sim_solid"], res["F_sim_solid"], color=c, ls="--", lw=1.6,
                label=rf"Solid: $\mu={res['mu_solid']:.1f}$ MPa")

        # Plot lattice-calibrated FE
        ax.plot(res["u_sim_opt"], res["F_sim_opt"], color=c, ls="-", lw=2.2,
                label=rf"Lattice: $\mu={res['mu_lattice']:.1f}$ MPa")

        ax.set_title(disp_name, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlabel(r"Displacement $u$ [mm]", fontsize=9.5)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=7.8, loc="upper left", frameon=True, framealpha=0.9)

    axes[0].set_ylabel(r"Axial Reaction Force $F$ [N]", fontsize=10)
    sns.despine(fig=fig)
    plt.tight_layout()

    out_pdf = PAPER_FIGS / "lattice_dma_calibrated_ramp.pdf"
    out_png = PAPER_FIGS / "lattice_dma_calibrated_ramp.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "lattice_dma_calibrated_ramp.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "lattice_dma_calibrated_ramp.png", dpi=300, bbox_inches="tight")
    print(f"\n[Figure Saved]: {out_pdf}")


def generate_latex_table(summary_df):
    tex_lines = [
        r"% Auto-generated summary table for DMA-calibrated lattice loading ramp",
        r"\begin{table}[htpb!]",
        r"    \centering",
        r"    \small",
        r"    \caption{Lattice loading ramp calibration of the DMA-derived Generalized Maxwell model ($N=6$ discrete Prony modes). With fixed linear viscoelastic relaxation spectra identified from small-strain DMA master curves at $T_{\mathrm{ref}} = \SI{20}{\celsius}$, the compressible neo-Hookean ground shear modulus ($\mu$, with bulk modulus $\kappa$ determined from physical Poisson's ratio $\nu$) was calibrated directly against the measured loading ramp of the 3D-printed architected lattice specimens ($n=3$).}",
        r"    \label{tab:dma-lattice-calibration}",
        r"    \begin{tabular}{lcccccccc}",
        r"        \toprule",
        r"        Blend & Poisson's & \multicolumn{3}{c}{Solid-Calibrated Baseline} & \multicolumn{4}{c}{Lattice-Calibrated Model} \\",
        r"        \cmidrule(lr){3-5} \cmidrule(lr){6-9}",
        r"              & Ratio $\nu$ & $\mu_{\mathrm{solid}}$ (MPa) & Peak Err. & Ramp RMSE & $\mu_{\mathrm{lattice}}$ (MPa) & $\mu_{\mathrm{lattice}}/\mu_{\mathrm{solid}}$ & Peak Err. & Ramp RMSE \\",
        r"        \midrule",
    ]

    for _, row in summary_df.iterrows():
        disp = row["display_name"]
        nu = f"{row['nu']:.3f}"
        mu_s = f"{row['mu_solid_MPa']:.1f}"
        pk_err_s = f"{row['peak_err_solid_pct']:+.1f}\\%"
        rmse_s = f"{row['rmse_solid_pct']:.1f}\\%"
        mu_l = f"{row['mu_lattice_MPa']:.1f}"
        ratio = f"{row['ratio_lattice_solid']:.3f}"
        pk_err_l = f"{row['peak_err_lattice_pct']:+.1f}\\%"
        rmse_l = f"{row['rmse_lattice_pct']:.1f}\\%"

        tex_lines.append(
            f"        {disp:5s} & {nu:5s} & {mu_s:7s} & {pk_err_s:7s} & {rmse_s:6s} & {mu_l:7s} & {ratio:6s} & {pk_err_l:7s} & {rmse_l:6s} \\\\"
        )

    tex_lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])

    table_path = PAPER_DIR / "lattice_dma_calibrated_table.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"[LaTeX Table Saved]: {table_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="all", choices=MATERIALS + ["all"])
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers across materials")
    parser.add_argument("--threads", type=int, default=4, help="OMP/MKL threads per job")
    parser.add_argument("--no-confirmation", action="store_true", help="Skip confirmation FE run (use exact scaled solution)")
    args = parser.parse_args()

    mats = MATERIALS if args.material == "all" else [args.material]
    all_res = {}
    summary_rows = []

    run_conf = not args.no_confirmation

    if len(mats) > 1 and args.workers > 1:
        print(f"Running calibration across {len(mats)} materials using {args.workers} parallel workers ({args.threads} threads each)...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_mat = {
                executor.submit(calibrate_blend, m, run_confirmation=run_conf, n_threads=args.threads): m
                for m in mats
            }
            for future in as_completed(future_to_mat):
                m = future_to_mat[future]
                res = future.result()
                all_res[m] = res
    else:
        for m in mats:
            res = calibrate_blend(m, run_confirmation=run_conf, n_threads=args.threads)
            all_res[m] = res

    # Sort results by MATERIALS order
    sorted_res = {m: all_res[m] for m in MATERIALS if m in all_res}

    for mat in sorted_res:
        res = sorted_res[mat]
        summary_rows.append({
            "material": mat,
            "display_name": DISPLAY_NAMES[mat],
            "nu": res["nu"],
            "F_peak_exp_N": res["F_peak_exp"],
            "mu_solid_MPa": res["mu_solid"],
            "peak_solid_N": res["peak_solid"],
            "peak_err_solid_pct": res["peak_err_solid_pct"],
            "rmse_solid_N": res["rmse_solid"],
            "rmse_solid_pct": res["rmse_solid_pct"],
            "mu_lattice_MPa": res["mu_lattice"],
            "ratio_lattice_solid": res["ratio_lattice_solid"],
            "peak_lattice_N": res["peak_lattice"],
            "peak_err_lattice_pct": res["peak_err_lattice_pct"],
            "rmse_lattice_N": res["rmse_lattice"],
            "rmse_lattice_pct": res["rmse_lattice_pct"],
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = FE_TENSION_DIR / "lattice_dma_calibrated_params.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Summary CSV Saved]: {csv_path}")

    # Generate LaTeX table
    generate_latex_table(summary_df)

    # Plot
    plot_calibration_results(sorted_res)

    print("\nCalibration Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
