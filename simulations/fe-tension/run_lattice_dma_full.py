#!/usr/bin/env python3
"""
run_lattice_dma_full.py

Simulate and plot the full structural response (loading ramp + 600-second relaxation hold)
of architected lattice dogbone specimens for the DMA-calibrated Generalized Maxwell model.

Features:
- Full 2-step protocol: constant-velocity loading ramp followed by 600 s constant-displacement hold.
- Evaluates solid baseline (mu_solid) and lattice-calibrated base (mu_lattice = ratio * mu_solid)
  via the exact linear scaling property F(t; mu) = (mu / mu_solid) * F(t; mu_solid).
- 2-row publication figure:
  * Row (a): Axial force F [N] vs. grip displacement u [mm] during the loading ramp.
  * Row (b): Axial force F [N] vs. time t [s] across the entire 600-second test.
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parents[2]
FE_TENSION_DIR = Path(__file__).resolve().parent
MESH_PATH = FE_TENSION_DIR / "mesh" / "lattice_mesh.inc"
EXP_DIR = REPO_ROOT / "data" / "tensile-lattice" / "cleaned"
OUTPUT_DIR = FE_TENSION_DIR / "lattice_dma_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS = REPO_ROOT.parent / "paper" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

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

MU_SOLID = {
    "A0V100": 1582.34,
    "A25V75": 798.78,
    "A50V50": 333.04,
    "A75V25": 199.07,
    "A100V0": 147.09,
}

MU_LATTICE = {
    "A0V100": 1269.21,
    "A25V75": 751.55,
    "A50V50": 613.48,
    "A75V25": 254.59,
    "A100V0": 101.72,
}

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

FULL_PROTOCOLS = {
    "A0V100": {"t_ramp": 2.958,  "u_octant": 0.246503, "t_hold": 600.657},
    "A25V75": {"t_ramp": 5.934,  "u_octant": 0.494208, "t_hold": 600.866},
    "A50V50": {"t_ramp": 11.871, "u_octant": 0.989416, "t_hold": 600.752},
    "A75V25": {"t_ramp": 29.722, "u_octant": 2.474240, "t_hold": 600.978},
    "A100V0": {"t_ramp": 11.908, "u_octant": 9.898960, "t_hold": 600.792},
}


def bulk_modulus(mu, nu):
    return 2.0 * mu * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))


def load_experimental_full(mat):
    """Load full experimental replicates (t, u, F)."""
    reps = []
    for r in [1, 2, 3]:
        p = EXP_DIR / f"{mat}-{r}_cleaned.csv"
        if p.exists():
            df = pd.read_csv(p)
            t = df["time_s"].values
            u = df["position_mm"].values
            F = -df["force_N"].values
            reps.append((t, u, F))

    proto = FULL_PROTOCOLS[mat]
    t_tot = proto["t_ramp"] + proto["t_hold"]
    t_grid = np.linspace(0, t_tot, 500)

    f_interp = []
    for t, u, F in reps:
        interp = interp1d(t, F, bounds_error=False, fill_value=(0.0, F[-1]))
        f_interp.append(interp(t_grid))

    mean_F = np.mean(f_interp, axis=0)
    std_F = np.std(f_interp, axis=0)

    # Also compute ramp mean
    u_max = 2.0 * proto["u_octant"]
    u_grid = np.linspace(0, u_max, 100)
    f_ramp_interp = []
    ramp_reps = []
    for t, u, F in reps:
        idx_peak = np.argmax(F)
        u_r = u[:idx_peak + 1]
        F_r = F[:idx_peak + 1]
        ramp_reps.append((u_r, F_r))
        interp_u = interp1d(u_r, F_r, bounds_error=False, fill_value=(0.0, F_r[-1]))
        f_ramp_interp.append(interp_u(u_grid))

    mean_F_ramp = np.mean(f_ramp_interp, axis=0)
    std_F_ramp = np.std(f_ramp_interp, axis=0)

    return {
        "t_grid": t_grid, "mean_F": mean_F, "std_F": std_F, "reps": reps,
        "u_grid": u_grid, "mean_F_ramp": mean_F_ramp, "std_F_ramp": std_F_ramp, "ramp_reps": ramp_reps
    }


def run_fe_full(mat, n_threads=8):
    """Run or load full 2-step FE simulation at mu_solid."""
    job_dir = OUTPUT_DIR / f"{mat}_full_eval"
    job_dir.mkdir(parents=True, exist_ok=True)
    f_csv = job_dir / "gripForce.csv"
    d_csv = job_dir / "gripDisp.csv"

    if f_csv.exists() and d_csv.exists() and os.path.getsize(f_csv) > 400:
        f_data = np.loadtxt(f_csv)
        d_data = np.loadtxt(d_csv)
        t_uniq, idx = np.unique(d_data[:, 0], return_index=True)
        t_full = d_data[idx, 0]
        u_full = 2.0 * d_data[idx, 1]
        F_full = 4.0 * f_data[idx, 1]
        proto = FULL_PROTOCOLS[mat]
        if t_full[-1] >= proto["t_ramp"] + 0.9 * proto["t_hold"]:
            print(f"[{mat}] Reusing completed full simulation ({len(t_full)} incs, t_max={t_full[-1]:.1f}s)")
            return t_full, u_full, F_full

    mu = MU_SOLID[mat]
    nu = POISSON_RATIOS[mat]
    kappa = bulk_modulus(mu, nu)
    proto = FULL_PROTOCOLS[mat]
    t_ramp = proto["t_ramp"]
    t_hold = proto["t_hold"]
    u_oct = proto["u_octant"]
    prony = DMA_PRONY[mat]

    inc_ramp = 0.05 if mat == "A100V0" else 0.10
    inc_hold = 0.05

    inp_lines = [
        f"*include, input={MESH_PATH}",
        "",
        f"*material, name=CompressibleFiniteStrainLinearViscoelasticity, id=mat_{mat}",
        "0, 1,",
        f"{mu:.6f}, {kappa:.6f},",
        f"{len(prony['gammas'])},"
    ]
    for g, tau in zip(prony["gammas"], prony["taus"]):
        inp_lines.append(f"{g:.6f}, {tau:.6e},")
    inp_lines.extend([
        "",
        f"*section, name=sec1, material=mat_{mat}, type=solid",
        "all",
        "",
        f"*job, name={mat}_full, domain=3d",
        "*solver, solver=NISTParallel, name=theSolver",
        "",
        "*fieldOutput",
        ">>perNode, name=gripDisp, nSet=movingGrip, field=displacement, result=U, f(x)='np.mean(x[:,0])', saveHistory=True, export=gripDisp",
        ">>perNode, name=gripForce, nSet=movingGrip, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=gripForce",
        "",
        "*step, solver=theSolver",
        f"startInc={inc_ramp:.4f}, maxInc={inc_ramp:.4f}, minInc=1e-4, maxNumInc=1000, maxIter=25, stepLength={t_ramp:.4f}",
        ">>options, name=theSolver, extrapolation=linear, linsolver=pardiso",
        ">>dirichlet, name=xsym, nSet=xSymmetry, field=displacement, 1=0",
        ">>dirichlet, name=ysym, nSet=ySymmetry, field=displacement, 2=0",
        ">>dirichlet, name=zsym, nSet=zSymmetry, field=displacement, 3=0",
        ">>dirichlet, name=fixmove, nSet=movingGrip, field=displacement, 2=0, 3=0",
        f">>dirichlet, name=ramp, nSet=movingGrip, field=displacement, 1={u_oct:.6f}",
        "",
        "*step, solver=theSolver",
        f"startInc={inc_hold:.4f}, maxInc={inc_hold:.4f}, minInc=1e-4, maxNumInc=1000, maxIter=25, stepLength={t_hold:.4f}",
        ">>options, name=theSolver, extrapolation=linear, linsolver=pardiso",
    ])

    inp_path = job_dir / f"{mat}_full.inp"
    with open(inp_path, "w") as f:
        f.write("\n".join(inp_lines) + "\n")

    print(f"[{mat}] Running full 2-step FE simulation with {n_threads} threads...")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)

    res = subprocess.run([EDELWEISS_BIN, "--quiet", "--noplot", str(inp_path)],
                         cwd=str(job_dir), env=env, capture_output=True, text=True)

    if not (f_csv.exists() and d_csv.exists()):
        raise RuntimeError(f"EdelweissFE failed for {mat}: {res.stderr}")

    f_data = np.loadtxt(f_csv)
    d_data = np.loadtxt(d_csv)
    t_uniq, idx = np.unique(d_data[:, 0], return_index=True)
    t_full = d_data[idx, 0]
    u_full = 2.0 * d_data[idx, 1]
    F_full = 4.0 * f_data[idx, 1]
    return t_full, u_full, F_full


def plot_combined_results(all_exp, all_fe):
    """Plot 2-row x 5-column figure: (a) Ramp F vs u, (b) Full F vs t."""
    mats = MATERIALS
    n_mats = len(mats)
    fig, axes = plt.subplots(2, n_mats, figsize=(16.0, 6.2), sharex=False)

    colors_mat = {
        "A0V100": "#2b5c8f",
        "A25V75": "#3b8b88",
        "A50V50": "#d95f02",
        "A75V25": "#7570b3",
        "A100V0": "#e7298a",
    }

    for col, mat in enumerate(mats):
        disp_name = DISPLAY_NAMES[mat]
        c = colors_mat[mat]
        exp = all_exp[mat]
        fe = all_fe[mat]

        t_sim = fe["t"]
        u_sim = fe["u"]
        F_solid = fe["F_solid"]
        F_opt = fe["F_opt"]

        proto = FULL_PROTOCOLS[mat]
        t_ramp = proto["t_ramp"]
        mask_ramp = t_sim <= t_ramp + 1e-4

        # ---------- Row 0: Loading Ramp (F vs u) ----------
        ax0 = axes[0, col]
        for u_r, f_r in exp["ramp_reps"]:
            ax0.plot(u_r, f_r, color="grey", alpha=0.35, lw=1.0)
        ax0.plot(exp["u_grid"], exp["mean_F_ramp"], color="black", lw=1.8, label=r"Exp. mean ($n=3$)")
        ax0.fill_between(exp["u_grid"], exp["mean_F_ramp"] - exp["std_F_ramp"], exp["mean_F_ramp"] + exp["std_F_ramp"],
                         color="black", alpha=0.12)
        ax0.plot(u_sim[mask_ramp], F_solid[mask_ramp], color=c, ls="--", lw=1.6,
                 label=rf"Solid: $\mu={MU_SOLID[mat]:.1f}$ MPa")
        ax0.plot(u_sim[mask_ramp], F_opt[mask_ramp], color=c, ls="-", lw=2.2,
                 label=rf"Lattice: $\mu={MU_LATTICE[mat]:.1f}$ MPa")
        ax0.set_title(disp_name, fontsize=11, fontweight="bold", pad=5)
        ax0.set_xlabel(r"Displacement $u$ [mm]", fontsize=9.5)
        ax0.grid(True, ls=":", alpha=0.6)
        ax0.legend(fontsize=7.5, loc="upper left", frameon=True, framealpha=0.9)

        # ---------- Row 1: Full Test (F vs t) ----------
        ax1 = axes[1, col]
        for t_r, _, f_r in exp["reps"]:
            ax1.plot(t_r, f_r, color="grey", alpha=0.35, lw=1.0)
        ax1.plot(exp["t_grid"], exp["mean_F"], color="black", lw=1.8, label=r"Exp. mean ($n=3$)")
        ax1.fill_between(exp["t_grid"], exp["mean_F"] - exp["std_F"], exp["mean_F"] + exp["std_F"],
                         color="black", alpha=0.12)
        ax1.plot(t_sim, F_solid, color=c, ls="--", lw=1.6,
                 label=rf"Solid baseline")
        ax1.plot(t_sim, F_opt, color=c, ls="-", lw=2.2,
                 label=rf"Lattice calibrated")
        ax1.set_xlabel(r"Time $t$ [s]", fontsize=9.5)
        ax1.grid(True, ls=":", alpha=0.6)
        ax1.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.9)

    axes[0, 0].set_ylabel(r"(a) Loading Ramp Force $F$ [N]", fontsize=10)
    axes[1, 0].set_ylabel(r"(b) Full Test Force $F$ [N]", fontsize=10)

    sns.despine(fig=fig)
    plt.tight_layout()

    out_pdf = PAPER_FIGS / "lattice_dma_calibrated_ramp.pdf"
    out_png = PAPER_FIGS / "lattice_dma_calibrated_ramp.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "lattice_dma_calibrated_ramp.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "lattice_dma_calibrated_ramp.png", dpi=300, bbox_inches="tight")
    print(f"\n[Updated 2-Row Figure Saved]: {out_pdf}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="all", choices=MATERIALS + ["all"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    mats = MATERIALS if args.material == "all" else [args.material]

    print("Loading experimental full datasets...")
    all_exp = {m: load_experimental_full(m) for m in MATERIALS}

    all_fe = {}
    for m in MATERIALS:
        if m in mats or (OUTPUT_DIR / f"{m}_full_eval" / "gripForce.csv").exists():
            t_sim, u_sim, F_solid = run_fe_full(m, n_threads=args.threads)
            ratio = MU_LATTICE[m] / MU_SOLID[m]
            F_opt = F_solid * ratio
            all_fe[m] = {
                "t": t_sim, "u": u_sim, "F_solid": F_solid, "F_opt": F_opt
            }

    if len(all_fe) == len(MATERIALS):
        print("\nPlotting combined 2-row publication figure...")
        plot_combined_results(all_exp, all_fe)
        print("Done!")
    else:
        print(f"Completed {len(all_fe)} of {len(MATERIALS)} materials. Run remaining materials to generate final plot.")


if __name__ == "__main__":
    main()
