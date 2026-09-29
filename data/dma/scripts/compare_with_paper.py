#!/usr/bin/env python3
"""
Compare DMA master curves, thermomechanical damping (tan delta), and glass
transition temperatures from local DMA-new experimental tests with the published
results from Wirth, Chapuis & Shea (Nature Communications 2025, 16:11705):
"Inverse design of stochastic, voxelated thermo-viscoelastic digital materials".

Outputs:
  - dma_paper_comparison_dashboard.png: Comprehensive 4-panel visual comparison.
  - dma_paper_comparison_master_curves.png: Master curve overlay in reduced frequency.
  - dma_paper_comparison_paper_format.png: Time-domain master curve in paper format (Fig 3a/3d style).
  - dma_paper_comparison_thermal_damping.png: Tan(delta) vs T and Tg vs composition percolation.
  - dma_paper_comparison_summary.csv: Quantitative metrics comparison table.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import utilities from local compare_master_curves
from compare_master_curves import (
    load_master_curve, compute_mean_curve, median_temp_step, style_axis,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE, BASELINE, SURFACE
)

OUR_COLORS = {
    "A0V100": "#2a78d6",   # Blue (Pure Vero)
    "A25V75": "#1baf7a",   # Teal-Green (75% Vero)
    "A50V50": "#eda100",   # Amber (50/50)
    "A75V25": "#eb6834",   # Orange (25% Vero)
    "A100V0": "#e34948",   # Red (Pure Agilus)
}


def load_local_data(dma_root: Path, min_density: float = 7.5):
    """Load and process local DMA datasets using compare_master_curves logic."""
    local_data = {}
    material_order = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]
    for mat in material_order:
        pdir = dma_root / mat / "processed"
        if not pdir.is_dir():
            continue
        master_files = sorted(pdir.glob("*__direct_wlf_master_curve.csv"))
        qualifying = []
        for mf in master_files:
            sample = mf.name[:-len("__direct_wlf_master_curve.csv")]
            df = load_master_curve(pdir, sample)
            if df is None or df.empty:
                continue
            step = median_temp_step(df["temperature_C"].to_numpy())
            if step is not None and step < min_density:
                qualifying.append(df)
        if qualifying:
            mean_df = compute_mean_curve(qualifying)
            local_data[mat] = {
                "mean_df": mean_df,
                "replicates": qualifying,
                "n_reps": len(qualifying)
            }
    return local_data


def load_paper_data(paper_data_dir: Path):
    """Load extracted Source Data from the paper."""
    fig3a = pd.read_csv(paper_data_dir / "paper_fig3a_master_curves.csv")
    fig3d = pd.read_csv(paper_data_dir / "paper_fig3d_percolation_comparison.csv")
    fig4d_td = pd.read_csv(paper_data_dir / "paper_fig4d_tan_delta_curves.csv")
    fig4d_tg_fit = pd.read_csv(paper_data_dir / "paper_fig4d_tg_percolation_fit.csv")
    fig4d_tg_exp = pd.read_csv(paper_data_dir / "paper_fig4d_tg_experimental.csv")
    supp_pts = pd.read_csv(paper_data_dir / "paper_supp_shifted_frequency_points.csv")
    return {
        "fig3a": fig3a,
        "fig3d": fig3d,
        "tan_delta": fig4d_td,
        "tg_fit": fig4d_tg_fit,
        "tg_exp": fig4d_tg_exp,
        "supp_points": supp_pts
    }


def get_hold_tan_delta(csv_path: Path):
    """Cluster raw points into temperature holds and return mean T and tan(delta)."""
    df = pd.read_csv(csv_path)
    temps = df["Temperature (°C)"].to_numpy()
    step = median_temp_step(temps)
    if step is not None and step >= 7.5:
        return None
    # Cluster consecutive points within 1.5 C into holds
    order = np.argsort(temps)
    temps_sorted = temps[order]
    tan_sorted = df["Tan(delta)"].to_numpy()[order]
    hold_ids = np.concatenate(([0], np.cumsum(np.diff(temps_sorted) > 1.5)))
    df_holds = pd.DataFrame({
        "hold_id": hold_ids,
        "T": temps_sorted,
        "tan_delta": tan_sorted
    }).groupby("hold_id").agg({"T": "mean", "tan_delta": "mean"}).sort_values("T")
    return df_holds


def compute_local_tg(dma_root: Path):
    """Extract peak tan(delta) and corresponding temperature for each local material."""
    local_tg = {}
    material_vfrac = {
        "A0V100": (1.0, 0.0),
        "A25V75": (0.75, 0.25),
        "A50V50": (0.50, 0.50),
        "A75V25": (0.25, 0.75),
        "A100V0": (0.0, 1.0),
    }
    for mat, (vw_f, ab_f) in material_vfrac.items():
        pdir = dma_root / mat / "processed"
        sweep_files = sorted(pdir.glob("*__temp_sweep.csv"))
        peaks_t, peaks_td = [], []
        hold_dfs = []
        for sf in sweep_files:
            h_df = get_hold_tan_delta(sf)
            if h_df is not None:
                hold_dfs.append(h_df)
                idx = h_df["tan_delta"].idxmax()
                peaks_t.append(h_df.loc[idx, "T"])
                peaks_td.append(h_df.loc[idx, "tan_delta"])
        if peaks_t:
            local_tg[mat] = {
                "vw_frac": vw_f,
                "ab_frac": ab_f,
                "Tg_mean": float(np.mean(peaks_t)),
                "Tg_std": float(np.std(peaks_t)),
                "tan_delta_max": float(np.mean(peaks_td)),
                "n_samples": len(peaks_t),
                "representative_holds": hold_dfs[-1]  # latest dense run
            }
    return local_tg


def plot_master_curves_comparison(local_data, paper_data, out_path: Path):
    """Plot overlay in reduced frequency domain (log-log)."""
    fig, (ax_mod, ax_comp) = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE)

    # Left: All materials comparison
    for mat, d in local_data.items():
        color = OUR_COLORS[mat]
        df = d["mean_df"]
        ax_mod.fill_between(df["reduced_freq_hz"], df["storage_modulus_min_kPa"] / 1e3,
                            df["storage_modulus_max_kPa"] / 1e3, color=color, alpha=0.25, linewidth=0)
        ax_mod.plot(df["reduced_freq_hz"], df["storage_modulus_kPa"] / 1e3, color=color,
                    linewidth=2.2, label=f"Ours: {mat}")

    fig3a = paper_data["fig3a"]
    t_vals = fig3a["time_s"].to_numpy()
    f_eq = 1.0 / t_vals

    paper_lines = [
        ("E_VW_MPa", "100% VW (Sample VII)", "#000000", "-"),
        ("E_50VW_MPa", "50% VW (Sample III)", "#555555", "--"),
        ("E_TB_MPa", "0% VW / Agilus (Sample I)", "#888888", ":"),
    ]
    for col, label, color, ls in paper_lines:
        ax_mod.plot(f_eq, fig3a[col], color=color, linestyle=ls, linewidth=1.8,
                    label=f"Paper: {label}", zorder=5)

    ax_mod.set_xscale("log")
    ax_mod.set_yscale("log")
    ax_mod.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_mod.set_ylabel("Storage modulus, E' (MPa)")
    ax_mod.set_title("Overview: All Compositions (Ours vs. Paper Benchmarks)", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_mod)
    ax_mod.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    # Right: Direct 1-to-1 matching for identical compositions
    pairs = [
        ("A0V100", "E_VW_MPa", "Pure VeroWhite (100% VW)", OUR_COLORS["A0V100"]),
        ("A50V50", "E_50VW_MPa", "50% VW / 50% Agilus", OUR_COLORS["A50V50"]),
        ("A100V0", "E_TB_MPa", "Pure Agilus (100% AB)", OUR_COLORS["A100V0"]),
    ]
    for our_mat, paper_col, title, color in pairs:
        df_our = local_data[our_mat]["mean_df"]
        ax_comp.plot(df_our["reduced_freq_hz"], df_our["storage_modulus_kPa"] / 1e3,
                     color=color, linewidth=2.4, label=f"Ours: {our_mat}")
        ax_comp.plot(f_eq, fig3a[paper_col], color=color, linestyle="--", linewidth=1.8,
                     label=f"Paper: {our_mat} equiv.", alpha=0.9)

    supp_pts = paper_data["supp_points"]
    supp_samples = [
        ("VeroWhite_100VW", OUR_COLORS["A0V100"]),
        ("50VW_50AB", OUR_COLORS["A50V50"]),
        ("Agilus_0VW_100AB", OUR_COLORS["A100V0"])
    ]
    for s_name, col in supp_samples:
        sub = supp_pts[supp_pts["material"] == s_name]
        sub_sample = sub.iloc[::3]
        ax_comp.scatter(sub_sample["shifted_freq_hz"], sub_sample["storage_modulus_MPa"],
                        color=col, s=12, alpha=0.35, edgecolors="none", zorder=2)

    ax_comp.set_xscale("log")
    ax_comp.set_yscale("log")
    ax_comp.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_comp.set_ylabel("Storage modulus, E' (MPa)")
    ax_comp.set_title("Direct 1-to-1 Comparison: Identical Compositions", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_comp)
    ax_comp.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Master Curve Comparison: Local DMA-new vs. Wirth et al. (2025)",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


def plot_paper_format_comparison(local_data, paper_data, out_path: Path):
    """
    Plot in the exact format of the published paper (Fig 3a & 3d):
    Linear E' [MPa] vs Time t [s] from 10^-10 to 10^10 s.
    """
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE)

    fig3a = paper_data["fig3a"]
    t_paper = fig3a["time_s"].to_numpy()

    paper_cols = [
        ("E_VW_MPa", "Paper: VII (100% VW)", "#104281"),
        ("E_83.5VW_MPa", "Paper: V (83.5% VW)", "#008300"),
        ("E_66VW_MPa", "Paper: IV* (66.7% VW)", "#4a3aa7"),
        ("E_50VW_MPa", "Paper: III (50% VW)", "#b8860b"),
        ("E_33VW_MPa", "Paper: VI* (33.3% VW)", "#d95f02"),
        ("E_16.5VW_MPa", "Paper: II (16.5% VW)", "#a63603"),
        ("E_TB_MPa", "Paper: I (0% VW / Agilus)", "#7f0000"),
    ]
    for col, lbl, color in paper_cols:
        ax_lin.plot(t_paper, fig3a[col], color=color, linewidth=1.5, linestyle="--", alpha=0.8, label=lbl)
        ax_log.plot(t_paper, fig3a[col], color=color, linewidth=1.5, linestyle="--", alpha=0.8, label=lbl)

    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        df = local_data[mat]["mean_df"]
        t_our = 1.0 / df["reduced_freq_hz"].to_numpy()
        e_our = df["storage_modulus_kPa"].to_numpy() / 1e3
        order = np.argsort(t_our)
        ax_lin.plot(t_our[order], e_our[order], color=color, linewidth=2.4, label=f"Ours: {mat}", zorder=4)
        ax_log.plot(t_our[order], e_our[order], color=color, linewidth=2.4, label=f"Ours: {mat}", zorder=4)

    ax_lin.set_xscale("log")
    ax_lin.set_xlim(1e-10, 1e10)
    ax_lin.set_ylim(-50, 2500)
    ax_lin.set_xlabel("Time scale, t (s)")
    ax_lin.set_ylabel("Storage modulus, E' (MPa)")
    ax_lin.set_title("Linear E' Scale (Format of Paper Fig. 3a / 3d)", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_lin)
    ax_lin.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlim(1e-10, 1e10)
    ax_log.set_ylim(0.1, 3500)
    ax_log.set_xlabel("Time scale, t (s)")
    ax_log.set_ylabel("Storage modulus, E' (MPa)")
    ax_log.set_title("Log E' Scale (Resolving Rubbery Plateau & Modulus Drop)", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_log)
    ax_log.legend(loc="lower left", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    fig.suptitle("Viscoelastic Relaxation in Time Domain (T = 22–25 °C)",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


def plot_thermal_damping_comparison(dma_root: Path, local_tg, paper_data, out_path: Path):
    """
    Plot Tan(delta) vs Temperature and Tg vs Base Material Composition.
    """
    fig, (ax_td, ax_tg) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=SURFACE)

    p_td = paper_data["tan_delta"]
    for mat_name, grp in p_td.groupby("material"):
        ax_td.plot(grp["temperature_C"], grp["tan_delta"], linestyle="--", linewidth=1.8,
                   alpha=0.8, label=f"Paper: {mat_name}")

    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        h_df = local_tg[mat]["representative_holds"]
        ax_td.plot(h_df["T"], h_df["tan_delta"], color=color, marker="o", markersize=4,
                   linewidth=2.2, label=f"Ours: {mat}")

    ax_td.set_xlabel("Temperature, T (°C)")
    ax_td.set_ylabel("Loss factor, tan(δ)")
    ax_td.set_title("Damping Behavior & Transition Width: tan(δ) vs. Temperature",
                    fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_td.set_xlim(-25, 115)
    ax_td.set_ylim(-0.05, 2.4)
    style_axis(ax_td)
    ax_td.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    tg_fit = paper_data["tg_fit"]

    ax_tg.plot(tg_fit["rho_est"] * 100, tg_fit["Tg_C"], color="#0b0b0b", linestyle="-",
               linewidth=2.0, label="Paper Percolation Fit (ρc=16.3%, μ=0.555)", zorder=3)

    ax_tg.axvline(16.3, color="#e34948", linestyle=":", linewidth=1.4, alpha=0.8,
                  label="Percolation threshold ρc = 16.3% VW")

    paper_tg_points = [
        (0.0, 20.0, "Paper: 0% VW (Agilus)"),
        (33.3, 55.1, "Paper: 33.3% VW"),
        (66.7, 75.3, "Paper: 66.7% VW"),
        (100.0, 80.0, "Paper: 100% VW"),
    ]
    for x_vw, y_tg, lbl in paper_tg_points:
        ax_tg.scatter(x_vw, y_tg, color="#52514e", marker="s", s=65, zorder=5,
                      label=lbl if x_vw == 0.0 else None)
    ax_tg.scatter([], [], color="#52514e", marker="s", s=65, label="Paper Experimental Tg")

    for mat, info in local_tg.items():
        color = OUR_COLORS[mat]
        x_vw = info["vw_frac"] * 100
        y_tg = info["Tg_mean"]
        ax_tg.scatter(x_vw, y_tg, color=color, marker="o", s=85, edgecolors=INK_PRIMARY,
                      linewidths=1.2, zorder=6, label=f"Ours: {mat} ({y_tg:.1f}°C)")

    ax_tg.set_xlabel("VeroWhite volume fraction, ρ_VW (%)")
    ax_tg.set_ylabel("Glass transition temperature, Tg (°C)")
    ax_tg.set_title("Percolation of Glass Transition: Tg vs. Mixture Composition",
                    fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_tg.set_xlim(-5, 105)
    ax_tg.set_ylim(10, 95)
    style_axis(ax_tg)
    ax_tg.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Thermomechanical Transitions & Percolation Characterization",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


def plot_combined_dashboard(local_data, local_tg, paper_data, dma_root: Path, out_path: Path):
    """
    Generate an executive 4-panel visual comparison dashboard:
      Panel A: Master curve in reduced frequency (Log-Log)
      Panel B: Viscoelastic relaxation in time domain (Linear E' scale, Paper Fig 3a style)
      Panel C: Loss factor tan(delta) vs Temperature (Peak damping and width)
      Panel D: Modulus plateaus & Glass Transition Tg across composition
    """
    fig = plt.figure(figsize=(16, 12), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- Panel A: Reduced Frequency Master Curves ----
    for mat, d in local_data.items():
        color = OUR_COLORS[mat]
        df = d["mean_df"]
        ax_a.fill_between(df["reduced_freq_hz"], df["storage_modulus_min_kPa"] / 1e3,
                          df["storage_modulus_max_kPa"] / 1e3, color=color, alpha=0.25, linewidth=0)
        ax_a.plot(df["reduced_freq_hz"], df["storage_modulus_kPa"] / 1e3, color=color,
                  linewidth=2.2, label=f"Ours: {mat}")

    fig3a = paper_data["fig3a"]
    t_vals = fig3a["time_s"].to_numpy()
    f_eq = 1.0 / t_vals

    ax_a.plot(f_eq, fig3a["E_VW_MPa"], color="#0b0b0b", linestyle="--", linewidth=1.6, label="Paper: 100% VW (VII)")
    ax_a.plot(f_eq, fig3a["E_50VW_MPa"], color="#52514e", linestyle="--", linewidth=1.6, label="Paper: 50% VW (III)")
    ax_a.plot(f_eq, fig3a["E_TB_MPa"], color="#898781", linestyle="--", linewidth=1.6, label="Paper: 0% VW (I, AB)")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_a.set_ylabel("Storage modulus, E' (MPa)")
    ax_a.set_title("A. Master Curves Across Reduced Frequency (Log-Log)", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_a)
    ax_a.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY)

    # ---- Panel B: Time Domain (Paper Fig 3a/3d format) ----
    paper_cols = [
        ("E_VW_MPa", "Paper: 100% VW", "#104281"),
        ("E_50VW_MPa", "Paper: 50% VW", "#b8860b"),
        ("E_TB_MPa", "Paper: 0% VW (Agilus)", "#7f0000"),
    ]
    for col, lbl, color in paper_cols:
        ax_b.plot(t_vals, fig3a[col], color=color, linewidth=1.5, linestyle="--", alpha=0.9, label=lbl)

    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        df = local_data[mat]["mean_df"]
        t_our = 1.0 / df["reduced_freq_hz"].to_numpy()
        e_our = df["storage_modulus_kPa"].to_numpy() / 1e3
        order = np.argsort(t_our)
        ax_b.plot(t_our[order], e_our[order], color=color, linewidth=2.2, label=f"Ours: {mat}")

    ax_b.set_xscale("log")
    ax_b.set_xlim(1e-10, 1e10)
    ax_b.set_ylim(-50, 2450)
    ax_b.set_xlabel("Time scale, t (s)")
    ax_b.set_ylabel("Storage modulus, E' (MPa)")
    ax_b.set_title("B. Relaxation Modulus in Time Domain (Paper Fig. 3a Format)", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_b)
    ax_b.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    # ---- Panel C: Tan(delta) vs Temperature ----
    p_td = paper_data["tan_delta"]
    for mat_name, grp in p_td.groupby("material"):
        ax_c.plot(grp["temperature_C"], grp["tan_delta"], linestyle="--", linewidth=1.6, alpha=0.75, label=f"Paper: {mat_name}")

    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        h_df = local_tg[mat]["representative_holds"]
        ax_c.plot(h_df["T"], h_df["tan_delta"], color=color, marker="o", markersize=3,
                  linewidth=2.2, label=f"Ours: {mat}")

    ax_c.set_xlabel("Temperature, T (°C)")
    ax_c.set_ylabel("Loss factor, tan(δ)")
    ax_c.set_title("C. Loss Tangent & Transition Broadening: tan(δ) vs. T", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_c.set_xlim(-25, 115)
    ax_c.set_ylim(-0.05, 2.35)
    style_axis(ax_c)
    ax_c.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    # ---- Panel D: Glassy/Rubbery Plateau & Glass Transition Scaling ----
    vw_pcts_our = []
    eg_our = []
    er_our = []
    tg_our = []
    for mat in ["A100V0", "A75V25", "A50V50", "A25V75", "A0V100"]:
        vw_pct = local_tg[mat]["vw_frac"] * 100
        vw_pcts_our.append(vw_pct)
        df = local_data[mat]["mean_df"]
        eg_our.append(df["storage_modulus_kPa"].max() / 1e3)
        er_our.append(df["storage_modulus_kPa"].min() / 1e3)
        tg_our.append(local_tg[mat]["Tg_mean"])

    paper_summary = [
        (0.0, 481.0, 0.30, 20.0),
        (16.5, 564.2, 1.08, 20.0),
        (33.3, 1341.6, 5.05, 55.1),
        (50.0, 1565.1, 9.47, 56.3),
        (66.7, 1791.7, 17.85, 75.3),
        (83.5, 1935.5, 7.78, 71.6),
        (100.0, 2094.2, 23.78, 80.0)
    ]
    p_vw = [p[0] for p in paper_summary]
    p_eg = [p[1] for p in paper_summary]
    p_er = [p[2] for p in paper_summary]
    p_tg = [p[3] for p in paper_summary]

    ax_d2 = ax_d.twinx()

    l1 = ax_d.plot(vw_pcts_our, eg_our, color="#2a78d6", marker="o", linewidth=2.0, label="Ours: Glassy Modulus (E'_g)")
    l2 = ax_d.plot(p_vw, p_eg, color="#2a78d6", linestyle="--", marker="s", linewidth=1.5, alpha=0.8, label="Paper: Glassy Modulus (E'_g)")

    l3 = ax_d.plot(vw_pcts_our, [e * 10 for e in er_our], color="#eb6834", marker="o", linewidth=1.8, label="Ours: Rubbery Modulus ×10 (10·E'_r)")
    l4 = ax_d.plot(p_vw, [e * 10 for e in p_er], color="#eb6834", linestyle="--", marker="s", linewidth=1.5, alpha=0.8, label="Paper: Rubbery Modulus ×10 (10·E'_r)")

    l5 = ax_d2.plot(vw_pcts_our, tg_our, color="#1baf7a", marker="^", linewidth=2.0, label="Ours: Tg (°C)")
    l6 = ax_d2.plot(p_vw, p_tg, color="#1baf7a", linestyle="--", marker="v", linewidth=1.5, alpha=0.8, label="Paper: Tg (°C)")

    ax_d.set_xlabel("VeroWhite volume fraction, ρ_VW (%)")
    ax_d.set_ylabel("Storage modulus, E' (MPa)", color="#2a78d6")
    ax_d2.set_ylabel("Glass transition temperature, Tg (°C)", color="#1baf7a")
    ax_d.set_title("D. Property Evolution Across Composition: E'_g, E'_r, and Tg", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    style_axis(ax_d)
    ax_d2.tick_params(colors=INK_MUTED, labelsize=9)
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["left"].set_visible(False)
    ax_d2.spines["right"].set_color(BASELINE)
    ax_d2.spines["bottom"].set_color(BASELINE)

    lines = l1 + l2 + l3 + l4 + l5 + l6
    labels = [l.get_label() for l in lines]
    ax_d.legend(lines, labels, loc="center left", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle("Comprehensive Benchmark Comparison: Local DMA-new Results vs. Nature Communications 2025",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.07, right=0.93, hspace=0.28, wspace=0.25)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> Saved Dashboard: {out_path}")


def generate_summary_table(local_data, local_tg, paper_data, out_csv: Path):
    """Generate structured CSV of quantitative comparisons."""
    paper_benchmarks = {
        "A0V100": {"vw_pct": 100.0, "paper_sample": "VII (100% VW)", "Eg_paper": 2094.2, "Er_paper": 23.78, "Tg_paper": 80.0, "tan_delta_max_paper": 1.033},
        "A25V75": {"vw_pct": 75.0, "paper_sample": "Interpolated (IV* & V)", "Eg_paper": (1791.7 + 1935.5) / 2, "Er_paper": (17.85 + 7.78) / 2, "Tg_paper": 71.6, "tan_delta_max_paper": 0.60},
        "A50V50": {"vw_pct": 50.0, "paper_sample": "III (50% VW)", "Eg_paper": 1565.1, "Er_paper": 9.47, "Tg_paper": 56.3, "tan_delta_max_paper": 0.58},
        "A75V25": {"vw_pct": 25.0, "paper_sample": "Interpolated (II & VI*)", "Eg_paper": (564.2 + 1341.6) / 2, "Er_paper": (1.08 + 5.05) / 2, "Tg_paper": 37.6, "tan_delta_max_paper": 0.56},
        "A100V0": {"vw_pct": 0.0, "paper_sample": "I (0% VW / Agilus)", "Eg_paper": 481.0, "Er_paper": 0.30, "Tg_paper": 20.0, "tan_delta_max_paper": 2.108},
    }

    rows = []
    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        d_our = local_data[mat]["mean_df"]
        eg_our = float(d_our["storage_modulus_kPa"].max() / 1e3)
        er_our = float(d_our["storage_modulus_kPa"].min() / 1e3)
        tg_our = float(local_tg[mat]["Tg_mean"])
        td_our = float(local_tg[mat]["tan_delta_max"])

        pb = paper_benchmarks[mat]
        eg_p = pb["Eg_paper"]
        er_p = pb["Er_paper"]
        tg_p = pb["Tg_paper"]
        td_p = pb["tan_delta_max_paper"]

        rows.append({
            "Material": mat,
            "VW_vol_pct": pb["vw_pct"],
            "Paper_Benchmark_Sample": pb["paper_sample"],
            "Glassy_Modulus_Ours_MPa": round(eg_our, 1),
            "Glassy_Modulus_Paper_MPa": round(eg_p, 1),
            "Glassy_Modulus_Diff_pct": round((eg_our - eg_p) / eg_p * 100, 1),
            "Rubbery_Modulus_Ours_MPa": round(er_our, 2),
            "Rubbery_Modulus_Paper_MPa": round(er_p, 2),
            "Rubbery_Modulus_Diff_pct": round((er_our - er_p) / er_p * 100, 1),
            "Tg_Ours_C": round(tg_our, 1),
            "Tg_Paper_C": round(tg_p, 1),
            "Tg_Diff_C": round(tg_our - tg_p, 1),
            "Peak_TanDelta_Ours": round(td_our, 3),
            "Peak_TanDelta_Paper": round(td_p, 3),
            "Peak_TanDelta_Diff_pct": round((td_our - td_p) / td_p * 100, 1)
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_csv, index=False)
    print(f"  -> Saved Summary Table: {out_csv}")
    return df_summary


def main():
    root = Path(__file__).resolve().parent.parent
    dma_root = root / "raw"
    paper_data_dir = root / "reference"

    print("=" * 70)
    print("DMA Results vs. Paper (Wirth et al., Nat Commun 2025) Visual Comparison")
    print(f"DMA Root:       {dma_root}")
    print(f"Paper Data Dir: {paper_data_dir}")
    print("=" * 70)

    print("Loading local DMA-new datasets...")
    local_data = load_local_data(dma_root)
    print(f"Loaded {len(local_data)} qualified material groups.")

    print("Computing local thermomechanical transition metrics (Tg, peak tan delta)...")
    local_tg = compute_local_tg(dma_root)

    print("Loading paper Source Data benchmarks...")
    paper_data = load_paper_data(paper_data_dir)

    fig_master = root / "dma_paper_comparison_master_curves.png"
    fig_paper_fmt = root / "dma_paper_comparison_paper_format.png"
    fig_damping = root / "dma_paper_comparison_thermal_damping.png"
    fig_dashboard = root / "dma_paper_comparison_dashboard.png"
    csv_summary = root / "dma_paper_comparison_summary.csv"

    print("\nGenerating Visual Comparisons:")
    plot_master_curves_comparison(local_data, paper_data, fig_master)
    plot_paper_format_comparison(local_data, paper_data, fig_paper_fmt)
    plot_thermal_damping_comparison(dma_root, local_tg, paper_data, fig_damping)
    plot_combined_dashboard(local_data, local_tg, paper_data, dma_root, fig_dashboard)

    summary_df = generate_summary_table(local_data, local_tg, paper_data, csv_summary)
    print("\nQuantitative Summary Table:")
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print("Visual comparison generation completed successfully!")


if __name__ == "__main__":
    main()
