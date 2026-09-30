#!/usr/bin/env python3
"""
Generate publication-quality DMA figures for the Additive Manufacturing manuscript,
incorporating both qualifying experimental replicates and GCV-regularized smoothing splines:

1. paper/figures/dma_temperature_sweeps.pdf:
   - Panel (a): Storage Modulus E' (MPa) vs. Temperature (C) at 1 Hz for both replicates + smooth spline
   - Panel (b): Tan(delta) vs. Temperature (C) at 1 Hz for both replicates + smooth spline
2. paper/figures/dma_tts_master_curves.pdf:
   - Panel (a): Storage Modulus E' (MPa) vs. Reduced Frequency (Hz) at T_ref = 20 C:
     pooled GCV smoothing spline + min-max replicate envelope across both replicates
   - Panel (b): TTS Shift Factors log10(a_T) vs. Temperature (C) for both replicates + WLF fits
3. paper/dma_summary_table.tex:
   - Summary table with mean +/- std (or replicate values) for Tg, peak tan delta, E'_g, and E'_r.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import curve_fit

# Set publication style matching paper/
sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1,
    rc={
        "lines.linewidth": 1.2,
        "text.usetex": True,
        "font.family": "cm",
        "legend.frameon": False,
        "legend.fontsize": "small",
    },
)
sns.set_palette("colorblind")
palette = sns.color_palette()

MAT_COLORS = {
    "A0V100": palette[0],
    "A25V75": palette[1],
    "A50V50": palette[2],
    "A75V25": palette[3],
    "A100V0": palette[4],
}

MAT_ORDER = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]
MAT_LABELS = {
    "A0V100": "A0V100 (0\\% AB)",
    "A25V75": "A25V75 (25\\% AB)",
    "A50V50": "A50V50 (50\\% AB)",
    "A75V25": "A75V25 (75\\% AB)",
    "A100V0": "A100V0 (100\\% AB)",
}

QUALIFYING_REPLICATES = {
    "A0V100": ["A0V100-2", "A0V100-4", "A0V100-5"],
    "A25V75": ["A25V75-2", "A25V75-3", "A25V75-4"],
    "A50V50": ["A50V50-7", "A50V50-8", "A50V50-9"],
    "A75V25": ["A75V25-2", "A75V25-3", "A75V25-4"],
    "A100V0": ["A100V0-1", "A100V0-2", "A100V0-3"],
}

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "raw"
FIG_DIR = BASE_DIR / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def pooled_smoothing_spline(x_all, y_all, grid):
    """Fit GCV-regularized smoothing spline through pooled replicate data."""
    order = np.argsort(x_all)
    x_sorted, y_sorted = x_all[order], y_all[order]
    x_unique, inverse = np.unique(x_sorted, return_inverse=True)
    y_unique = np.bincount(inverse, weights=y_sorted) / np.bincount(inverse)
    spline = make_smoothing_spline(x_unique, y_unique)
    return spline(grid)


# -------------------------------------------------------------
# 1. Temperature Sweeps at ~1 Hz (both replicates + pooled spline)
# -------------------------------------------------------------
summary_table_data = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18 / 2.54, 7.5 / 2.54), sharex=True)

for mat in MAT_ORDER:
    color = MAT_COLORS[mat]
    label = MAT_LABELS[mat]
    samples = QUALIFYING_REPLICATES[mat]
    proc = DATA_DIR / mat / "processed"
    
    rep_dfs = []
    per_rep_metrics = []
    
    for s_idx, s in enumerate(samples):
        csv_p = proc / f"{s}__temp_sweep.csv"
        df = pd.read_csv(csv_p)
        df["freq_hz"] = df["Angular frequency (rad/s)"] / (2 * math.pi)
        
        # Extract points nearest 1 Hz
        target_freq = df["freq_hz"].iloc[(df["freq_hz"] - 1.0).abs().argmin()]
        sub = df[(df["freq_hz"] - target_freq).abs() < 0.2].copy()
        sub["E_prime_MPa"] = sub["Storage modulus (kPa)"] / 1000.0
        sub["tan_delta"] = sub["Tan(delta)"]
        sub = sub.sort_values("Temperature (°C)")
        rep_dfs.append(sub)
        
        # Compute metrics for this replicate
        max_tan_idx = sub["tan_delta"].idxmax()
        tg_rep = sub.loc[max_tan_idx, "Temperature (°C)"]
        tan_max_rep = sub.loc[max_tan_idx, "tan_delta"]
        eg_rep = sub["E_prime_MPa"].iloc[0]
        
        if mat == "A100V0":
            sub_rub = sub[(sub["Temperature (°C)"] >= 20) & (sub["Temperature (°C)"] <= 30)]
            er_rep = sub_rub["E_prime_MPa"].mean() if len(sub_rub) > 0 else sub["E_prime_MPa"].iloc[-1]
        else:
            sub_rub = sub[sub["Temperature (°C)"] >= 80]
            er_rep = sub_rub["E_prime_MPa"].mean() if len(sub_rub) > 0 else sub["E_prime_MPa"].iloc[-1]
            
        per_rep_metrics.append({
            "sample": s,
            "Tg": tg_rep,
            "tan_max": tan_max_rep,
            "Eg": eg_rep,
            "Er": er_rep,
        })
        
        # Plot individual replicate points
        markers = ["o", "^", "s", "D", "v"]
        marker = markers[s_idx % len(markers)]
        ax1.plot(sub["Temperature (°C)"], sub["E_prime_MPa"],
                 marker=marker, markersize=3, linestyle="none", color=color, alpha=0.55, zorder=2)
        ax2.plot(sub["Temperature (°C)"], sub["tan_delta"],
                 marker=marker, markersize=3, linestyle="none", color=color, alpha=0.55, zorder=2)

    # Common grid for pooled spline and envelope
    t_min = max(df["Temperature (°C)"].min() for df in rep_dfs)
    t_max = min(df["Temperature (°C)"].max() for df in rep_dfs)
    t_grid = np.linspace(t_min, t_max, 150)
    
    # Envelope across all replicates
    e_interps = np.array([np.interp(t_grid, df["Temperature (°C)"], df["E_prime_MPa"]) for df in rep_dfs])
    tan_interps = np.array([np.interp(t_grid, df["Temperature (°C)"], df["tan_delta"]) for df in rep_dfs])
    
    # Shaded replicate envelope
    ax1.fill_between(t_grid, np.min(e_interps, axis=0), np.max(e_interps, axis=0), color=color, alpha=0.2, zorder=1)
    ax2.fill_between(t_grid, np.min(tan_interps, axis=0), np.max(tan_interps, axis=0), color=color, alpha=0.2, zorder=1)
    
    # Pooled spline
    all_t = np.concatenate([df["Temperature (°C)"].to_numpy() for df in rep_dfs])
    all_log_e = np.concatenate([np.log10(df["E_prime_MPa"].to_numpy()) for df in rep_dfs])
    all_tan = np.concatenate([df["tan_delta"].to_numpy() for df in rep_dfs])
    
    spline_log_e = pooled_smoothing_spline(all_t, all_log_e, t_grid)
    spline_tan = np.clip(pooled_smoothing_spline(all_t, all_tan, t_grid), 0, None)
    
    ax1.plot(t_grid, 10 ** spline_log_e, linestyle="-", color=color, linewidth=1.5, label=label, zorder=3)
    ax2.plot(t_grid, spline_tan, linestyle="-", color=color, linewidth=1.5, label=label, zorder=3)
    
    # Aggregate metrics across the replicates
    tg_mean = np.mean([m["Tg"] for m in per_rep_metrics])
    tg_std = np.std([m["Tg"] for m in per_rep_metrics])
    tan_mean = np.mean([m["tan_max"] for m in per_rep_metrics])
    tan_std = np.std([m["tan_max"] for m in per_rep_metrics])
    eg_mean = np.mean([m["Eg"] for m in per_rep_metrics])
    eg_std = np.std([m["Eg"] for m in per_rep_metrics])
    er_mean = np.mean([m["Er"] for m in per_rep_metrics])
    er_std = np.std([m["Er"] for m in per_rep_metrics])
    
    summary_table_data.append({
        "Material": mat,
        "VW_pct": int(mat[mat.index("V")+1:]),
        "Tg_mean": tg_mean,
        "Tg_std": tg_std,
        "tan_mean": tan_mean,
        "tan_std": tan_std,
        "Eg_mean": eg_mean,
        "Eg_std": eg_std,
        "Er_mean": er_mean,
        "Er_std": er_std,
        "reps": per_rep_metrics,
    })

ax1.set_yscale("log")
ax1.set_xlabel("Temperature $T$ ($^\\circ$C)")
ax1.set_ylabel("Storage modulus $E'$ (MPa)")
ax1.set_ylim(0.5, 3500)
ax1.set_xlim(-25, 115)
ax1.text(0.04, 0.06, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", fontsize=7.5)

ax2.set_xlabel("Temperature $T$ ($^\\circ$C)")
ax2.set_ylabel("Loss factor $\\tan\\delta$ (--)")
ax2.set_ylim(0, 2.3)
ax2.text(0.04, 0.92, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

# Annotate peak Tg
for item in summary_table_data:
    mat = item["Material"]
    tg = item["Tg_mean"]
    tm = item["tan_mean"]
    ax2.annotate(f"{tg:.0f}$^\\circ$C",
                 xy=(tg, tm),
                 xytext=(tg + 3, tm + 0.06),
                 fontsize=7,
                 color=MAT_COLORS[mat])

# Add custom handle for replicates vs spline in legend
custom_lines = [
    plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Smooth spline fit"),
    plt.Line2D([0], [0], color="gray", marker="o", linestyle="none", markersize=3.5, label="Replicate 1"),
    plt.Line2D([0], [0], color="gray", marker="^", linestyle="none", markersize=3.5, label="Replicate 2"),
    plt.Line2D([0], [0], color="gray", marker="s", linestyle="none", markersize=3.5, label="Replicate 3"),
]
ax2.legend(handles=custom_lines, loc="upper right", fontsize=7.5)

sns.despine(fig=fig)
fig.tight_layout()
fig_path1 = FIG_DIR / "dma_temperature_sweeps.pdf"
fig.savefig(fig_path1, dpi=300)
print(f"Saved: {fig_path1}")
plt.close(fig)


# -------------------------------------------------------------
# 2. TTS Master Curves (replicates + pooled GCV spline)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18 / 2.54, 7.5 / 2.54))

for mat in MAT_ORDER:
    color = MAT_COLORS[mat]
    label = MAT_LABELS[mat]
    samples = QUALIFYING_REPLICATES[mat]
    proc = DATA_DIR / mat / "processed"
    
    master_dfs = [pd.read_csv(proc / f"{s}__direct_wlf_master_curve.csv").sort_values("reduced_freq_hz") for s in samples]
    
    # Filter positive
    master_dfs = [df[df["storage_modulus_kPa"] > 0].copy() for df in master_dfs]
    for df in master_dfs:
        df["E_prime_MPa"] = df["storage_modulus_kPa"] / 1000.0
        
    # Grid spanning overlap of all replicates
    log_min = max(np.log10(df["reduced_freq_hz"].min()) for df in master_dfs)
    log_max = min(np.log10(df["reduced_freq_hz"].max()) for df in master_dfs)
    log_grid = np.linspace(log_min, log_max, 250)
    
    # Envelope from interpolation
    e_grids = np.array([
        np.interp(log_grid, np.log10(df["reduced_freq_hz"]), np.log10(df["E_prime_MPa"]))
        for df in master_dfs
    ])
    
    ax1.fill_between(10 ** log_grid, 10 ** np.min(e_grids, axis=0), 10 ** np.max(e_grids, axis=0),
                     color=color, alpha=0.25, zorder=1)
                     
    # Pooled GCV smoothing spline
    pooled_x = np.concatenate([np.log10(df["reduced_freq_hz"].to_numpy()) for df in master_dfs])
    pooled_y = np.concatenate([np.log10(df["E_prime_MPa"].to_numpy()) for df in master_dfs])
    spline_y = pooled_smoothing_spline(pooled_x, pooled_y, log_grid)
    
    ax1.plot(10 ** log_grid, 10 ** spline_y, linestyle="-", color=color, linewidth=1.5, label=label, zorder=3)
    
    # Also plot thin dashed individual replicate curves to show agreement
    rep_linestyles = [":", "--", "-."]
    for s_idx, df in enumerate(master_dfs):
        ls = rep_linestyles[s_idx % len(rep_linestyles)]
        ax1.plot(df["reduced_freq_hz"], df["E_prime_MPa"],
                 linestyle=ls, color=color, linewidth=0.8, alpha=0.6, zorder=2)

    # Plot Shift Factors for all replicates
    shift_dfs = []
    markers = ["o", "^", "s", "D", "v"]
    for s_idx, s in enumerate(samples):
        sf_p = proc / f"{s}__shift_factors.csv"
        if sf_p.exists():
            sdf = pd.read_csv(sf_p).sort_values("temperature_C")
            if "isolated_low_confidence" in sdf.columns:
                sdf = sdf[sdf["isolated_low_confidence"] == False]
            shift_dfs.append(sdf)
            marker = markers[s_idx % len(markers)]
            ax2.plot(sdf["temperature_C"], sdf["log_aT"],
                     marker=marker, markersize=3, linestyle="none", color=color, alpha=0.7, zorder=3)
                     
    # Combined WLF fit through all replicates
    if shift_dfs:
        comb_shifts = pd.concat(shift_dfs, ignore_index=True).sort_values("temperature_C")
        sub_fit = comb_shifts[(comb_shifts["temperature_C"] >= 20) & (comb_shifts["temperature_C"] <= 90)].copy()
        if len(sub_fit) >= 4:
            try:
                def wlf(T, c1, c2):
                    return -c1 * (T - 20.0) / (c2 + (T - 20.0))
                popt, _ = curve_fit(wlf, sub_fit["temperature_C"], sub_fit["log_aT"], p0=[15.0, 50.0], maxfev=5000)
                t_fit = np.linspace(20, comb_shifts["temperature_C"].max(), 100)
                ax2.plot(t_fit, wlf(t_fit, *popt), linestyle="-", color=color, linewidth=1.2, zorder=2)
            except Exception:
                pass

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Reduced frequency $f_r$ (Hz) at $T_{\\mathrm{ref}} = 20^\\circ$C")
ax1.set_ylabel("Storage modulus $E'$ (MPa)")
ax1.set_xlim(1e-12, 1e15)
ax1.set_ylim(0.5, 4500)
ax1.text(0.04, 0.06, "(a)", transform=ax1.transAxes, fontsize=11, fontweight="bold")
ax1.legend(loc="lower left", fontsize=7.5)

ax2.set_xlabel("Temperature $T$ ($^\\circ$C)")
ax2.set_ylabel("Shift factor $\\log_{10}(a_T)$ (--)")
ax2.text(0.04, 0.06, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")
ax2.set_xlim(-25, 115)
ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ax2.axvline(20, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

# Legend on ax2 for replicate symbols
custom_lines_tts = [
    plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="WLF fit (combined)"),
    plt.Line2D([0], [0], color="gray", marker="o", linestyle="none", markersize=3.5, label="Replicate 1"),
    plt.Line2D([0], [0], color="gray", marker="^", linestyle="none", markersize=3.5, label="Replicate 2"),
    plt.Line2D([0], [0], color="gray", marker="s", linestyle="none", markersize=3.5, label="Replicate 3"),
]
ax2.legend(handles=custom_lines_tts, loc="upper right", fontsize=7.5)

sns.despine(fig=fig)
fig.tight_layout()
fig_path2 = FIG_DIR / "dma_tts_master_curves.pdf"
fig.savefig(fig_path2, dpi=300)
print(f"Saved: {fig_path2}")
plt.close(fig)


# -------------------------------------------------------------
# 3. Generate Updated LaTeX Summary Table
# -------------------------------------------------------------
summary_table_path = BASE_DIR / "paper" / "dma_summary_table.tex"

table_tex = [
    "% Auto-generated DMA summary table showing all replicates and mean values",
    "\\begin{table}[htpb!]",
    "    \\centering",
    "    \\small",
    "    \\caption{Dynamic mechanical analysis (DMA) properties at \\SI{1}{\\hertz} across three experimental replicates ($n=3$) and their mean for each digital composite blend. Reference temperature for TTS is $T_{\\mathrm{ref}} = \\SI{20}{\\celsius}$.}",
    "    \\label{tab:dma-summary}",
    "    \\begin{tabular}{lcccccc}",
    "        \\toprule",
    "        Blend & Vero Vol.\\ & Replicate & $T_g$ & $(\\tan\\delta)_{\\mathrm{max}}$ & $E'_g$ (Glassy) & $E'_r$ (Rubbery) \\\\",
    "         & (\\%) & ID & (\\si{\\celsius}) & (--) & (\\si{\\mega\\pascal}) & (\\si{\\mega\\pascal}) \\\\",
    "        \\midrule",
]

for row in summary_table_data:
    mat = row["Material"]
    vw = row["VW_pct"]
    reps = row["reps"]
    n_reps = len(reps)
    
    for idx, r in enumerate(reps):
        if idx == 0:
            table_tex.append(
                f"        \\multirow{{{n_reps + 1}}}{{*}}{{{mat}}} & \\multirow{{{n_reps + 1}}}{{*}}{{{vw}}} & {r['sample']} & {r['Tg']:.1f} & {r['tan_max']:.2f} & {r['Eg']:.1f} & {r['Er']:.2f} \\\\"
            )
        else:
            table_tex.append(
                f"         &  & {r['sample']} & {r['Tg']:.1f} & {r['tan_max']:.2f} & {r['Eg']:.1f} & {r['Er']:.2f} \\\\"
            )
    # Mean
    table_tex.append(
        f"         &  & \\textbf{{Mean}} & \\textbf{{{row['Tg_mean']:.1f}}} & \\textbf{{{row['tan_mean']:.2f}}} & \\textbf{{{row['Eg_mean']:.1f}}} & \\textbf{{{row['Er_mean']:.2f}}} \\\\"
    )
    if mat != MAT_ORDER[-1]:
        table_tex.append("        \\midrule")

table_tex.extend([
    "        \\bottomrule",
    "    \\end{tabular}",
    "\\end{table}",
])

with open(summary_table_path, "w") as f:
    f.write("\n".join(table_tex))
print(f"Saved: {summary_table_path}")
