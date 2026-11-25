#!/usr/bin/env python3
"""Generate publication-ready lattice validation figure with:
Row 1: Axial force normalized by mean experimental peak force (F / \bar{F}_peak),
       with loading ramp insets (F / \bar{F}_peak vs. u).
Row 2: Normalized relaxation curves (F / F_peak) during the hold phase (no loading ramp),
       where each curve (experiment and individual constitutive models) is normalized by
       its own peak load, isolating and directly comparing intrinsic relaxation decay kinetics.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors

FE_BASE = REPO_ROOT / "simulations" / "fe-tension" / "jobs"
EXP_BASE = REPO_ROOT / "data" / "tensile-lattice" / "cleaned"
PAPER_FIGS = Path(__file__).resolve().parents[2] / "paper" / "figures"
ANALYSIS_DIR = Path(__file__).resolve().parent

MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]
DISPLAY_NAMES = {"A0V100": "A0", "A25V75": "A25", "A50V50": "A50", "A75V25": "A75", "A100V0": "A100"}
MODELS = [("", "GM", "-"), ("_bb", "BB", "--"), ("_fractional_schapery", "FS", "-.")]
RAMP_TIMES = {"A0V100": 2.96, "A25V75": 5.93, "A50V50": 11.87, "A75V25": 29.72, "A100V0": 11.91}

X_SYMMETRY_FACTOR = 2.0
CROSS_SECTION_SYMMETRY_FACTOR = 4.0


def load_fe(mat, sfx):
    dp = FE_BASE / f"{mat}{sfx}" / "gripDisp.csv"
    fp = FE_BASE / f"{mat}{sfx}" / "gripForce.csv"
    if not (dp.exists() and fp.exists()):
        return None
    ddata = np.loadtxt(dp)
    fdata = np.loadtxt(fp)
    t, idx = np.unique(ddata[:, 0], return_index=True)
    disp = X_SYMMETRY_FACTOR * ddata[idx, 1]
    force = CROSS_SECTION_SYMMETRY_FACTOR * fdata[idx, 1]
    return t, disp, force


def load_replicates(mat):
    reps = []
    for r in (1, 2, 3):
        p = EXP_BASE / f"{mat}-{r}_cleaned.csv"
        if p.exists():
            df = pd.read_csv(p)
            reps.append((df["time_s"].values, df["position_mm"].values, -df["force_N"].values))
    return reps


def generate_comparison_plots():
    n_mat = len(MATERIALS)
    fig, axes = plt.subplots(2, n_mat, figsize=(19 / 2.54, 9.8 / 2.54), sharey="row", sharex="col")

    for i, mat in enumerate(MATERIALS):
        dname = DISPLAY_NAMES[mat]
        reps = load_replicates(mat)
        t_ramp = RAMP_TIMES.get(mat, 10.0)
        t_max = min(t.max() for t, _, _ in reps)
        xg = np.linspace(0, t_max, 1000)
        f_interp = [np.interp(xg, t, f) for t, _, f in reps]
        f_min = np.min(f_interp, axis=0)
        f_max = np.max(f_interp, axis=0)
        f_mean = np.mean(f_interp, axis=0)
        f_bar_peak = np.max(f_mean)
        i_mean_peak = np.argmax(f_mean)
        t_mean_peak = xg[i_mean_peak]

        # Row 0: Normalized by mean experimental peak force (full test duration)
        ax0 = axes[0, i]
        ax0.fill_between(xg, f_min / f_bar_peak, f_max / f_bar_peak, color="0.78", alpha=0.55, zorder=1)
        ax0.plot(xg, f_mean / f_bar_peak, color="0.25", linestyle="-", linewidth=1.1, zorder=2)
        ax0.set_ylim(-0.02, 1.65)
        ax0.set_title(
            f"{dname}\n" + r"($\bar{F}_{\mathrm{peak}} = " + f"{f_bar_peak:.1f}" + r"\,\mathrm{N}$)",
            fontsize=7.5,
        )
        ax0.grid(True, alpha=0.4)
        ax0.tick_params(labelbottom=False)

        # Loading ramp inset on Row 0
        ins_pos = [0.46, 0.08, 0.49, 0.35] if mat == "A0V100" else [0.46, 0.62, 0.49, 0.34]
        ax_ins = ax0.inset_axes(ins_pos)
        v_reps = [
            (d[t <= t_ramp + 0.05], f[t <= t_ramp + 0.05])
            for t, d, f in reps
            if len(d[t <= t_ramp + 0.05]) > 1
        ]
        d_max = min(d.max() for d, _ in v_reps)
        dg = np.linspace(0, d_max, 200)
        f_ramp_int = [np.interp(dg, d, f) for d, f in v_reps]
        ax_ins.fill_between(
            dg,
            np.min(f_ramp_int, axis=0) / f_bar_peak,
            np.max(f_ramp_int, axis=0) / f_bar_peak,
            color="0.78",
            alpha=0.55,
        )
        ax_ins.plot(dg, np.mean(f_ramp_int, axis=0) / f_bar_peak, color="0.25", linewidth=1.0)
        ax_ins.set_facecolor("white")
        ax_ins.tick_params(labelsize=4.0, pad=1, length=1.5, width=0.5)
        ax_ins.grid(True, alpha=0.25, linewidth=0.4)
        for spine in ax_ins.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#555555")
        ax_ins.set_xlabel(r"$u$ (mm)", fontsize=4.5, labelpad=0.5)
        ax_ins.set_ylabel(r"$F / \bar{F}_{\mathrm{peak}}$", fontsize=4.5, labelpad=0.5)
        ax_ins.set_ylim(-0.02, 1.48)
        ax_ins.set_yticks([0, 0.5, 1.0])

        # Row 1: Relaxation curve only (no ramp), starting at peak
        ax1 = axes[1, i]
        xg_hold = np.linspace(t_mean_peak, t_max, 800)
        rep_hold_interps = []
        for t_r, _, f_r in reps:
            i_pk_r = np.argmax(f_r)
            t_sub = t_r[i_pk_r:]
            f_sub = f_r[i_pk_r:] / f_r[i_pk_r]
            f_norm_interp = np.interp(xg_hold, t_sub, f_sub)
            rep_hold_interps.append(f_norm_interp)

        f_rep_min = np.min(rep_hold_interps, axis=0)
        f_rep_max = np.max(rep_hold_interps, axis=0)
        f_mean_hold = np.interp(xg_hold, xg, f_mean) / f_bar_peak

        ax1.fill_between(xg_hold, f_rep_min, f_rep_max, color="0.78", alpha=0.55, zorder=1)
        ax1.plot(xg_hold, f_mean_hold, color="0.25", linestyle="-", linewidth=1.1, zorder=2)
        ax1.set_ylim(-0.02, 1.08)
        ax1.grid(True, alpha=0.4)
        if i == 2:
            ax1.set_xlabel("Time in s", fontsize=8.5)

        # Plot FE models
        for sfx, lbl, ls in MODELS:
            res = load_fe(mat, sfx)
            if res is None:
                continue
            t_fe, disp_fe, force_fe = res
            pk_fe = np.max(force_fe)
            i_pk = np.argmax(force_fe)
            t_ramp_fe = t_fe[i_pk]

            t_dense_ramp = np.linspace(0, t_ramp_fe, 200)
            f_dense_ramp = np.interp(t_dense_ramp, t_fe[: i_pk + 1], force_fe[: i_pk + 1])
            u_hold = np.log(1.0 + (t_fe[i_pk:] - t_ramp_fe) / 5.0)
            t_dense_hold = np.linspace(t_ramp_fe, t_fe[-1], 800)
            u_dense_hold = np.log(1.0 + (t_dense_hold - t_ramp_fe) / 5.0)
            f_dense_hold = pchip_interpolate(u_hold, force_fe[i_pk:], u_dense_hold)
            t_smooth = np.concatenate([t_dense_ramp, t_dense_hold[1:]])
            f_smooth = np.concatenate([f_dense_ramp, f_dense_hold[1:]])

            # Row 0: Full curve normalized by f_bar_peak
            ax0.plot(t_smooth, f_smooth / f_bar_peak, color=colors[i], linestyle=ls, linewidth=1.2, zorder=3)
            mask_ins = t_fe <= t_ramp + 0.05
            ax_ins.plot(disp_fe[mask_ins], force_fe[mask_ins] / f_bar_peak, color=colors[i], linestyle=ls, linewidth=1.0, zorder=3)

            # Row 1: Relaxation curve only (t_dense_hold), normalized by pk_fe
            ax1.plot(t_dense_hold, f_dense_hold / pk_fe, color=colors[i], linestyle=ls, linewidth=1.2, zorder=3)

    axes[0, 0].set_ylabel(r"(a) $F / \bar{F}_{\mathrm{peak}}$", fontsize=8.0)
    axes[1, 0].set_ylabel(r"(b) $F / F_{\mathrm{peak}}$", fontsize=8.0)

    legend_handles = [
        Patch(facecolor="0.78", edgecolor="none", alpha=0.55, label="Experiment (scatter)"),
        Line2D([], [], color="0.25", linestyle="-", linewidth=1.1, label="Experiment (mean)"),
        Line2D([], [], color="black", linestyle="-", label="GM"),
        Line2D([], [], color="black", linestyle="--", label="BB"),
        Line2D([], [], color="black", linestyle="-.", label="FS"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        fontsize=7.5,
    )
    sns.despine(fig=fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for out_dir in [ANALYSIS_DIR, PAPER_FIGS]:
        for ext in ["pdf", "png"]:
            fig.savefig(out_dir / f"lattice_experiment_comparison_force_time.{ext}", dpi=300, bbox_inches="tight")
    print("Saved lattice_experiment_comparison_force_time.[pdf,png] successfully!")


if __name__ == "__main__":
    generate_comparison_plots()
