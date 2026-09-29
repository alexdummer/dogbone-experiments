#!/usr/bin/env python3
"""
Compare RAW, UNSHIFTED DMA experimental data from local DMA-new testing
against the published raw DMA measurements from Wirth et al. (Nature Communications 2025).

No WLF fitting, no Arrhenius shifting, no TTS master curve construction:
this directly compares:
  1. Raw Storage Modulus (E') vs. Temperature (°C) across the full sweep.
  2. Raw Loss Factor (tan δ) vs. Temperature (°C).
  3. Raw Storage Modulus vs. Frequency at Isothermal Holds.
  4. 1-to-1 matching for identical compositions (100% VW, 50% VW, 0% VW).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utilities from compare_master_curves
from compare_master_curves import (
    median_temp_step, style_axis,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE, BASELINE, SURFACE
)

OUR_COLORS = {
    "A0V100": "#2a78d6",   # Blue (Pure Vero)
    "A25V75": "#1baf7a",   # Teal-Green (75% Vero)
    "A50V50": "#eda100",   # Amber (50/50)
    "A75V25": "#eb6834",   # Orange (25% Vero)
    "A100V0": "#e34948",   # Red (Pure Agilus)
}

PAPER_COLORS = {
    "100% VW": "#104281",
    "83.5% VW": "#008300",
    "66.7% VW": "#4a3aa7",
    "50% VW": "#b8860b",
    "33.3% VW": "#d95f02",
    "16.5% VW": "#a63603",
    "0% VW (Agilus)": "#7f0000",
}


def load_all_raw_data(dma_root: Path, paper_data_dir: Path):
    """Load local and paper raw experimental data and group by nominal hold temperature."""
    local_pts = []
    local_holds = []
    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        pdir = dma_root / mat / "processed"
        for f in sorted(pdir.glob("*__temp_sweep.csv")):
            if any(x in f.name for x in ["-1", "-6"]) and mat in ["A0V100", "A25V75", "A50V50"]:
                continue  # skip coarse shakeouts
            sample = f.name[:-len("__temp_sweep.csv")]
            df = pd.read_csv(f)
            temps = df["Temperature (°C)"].to_numpy()
            step = median_temp_step(temps)
            if step is not None and step >= 7.5:
                continue

            for _, r in df.iterrows():
                local_pts.append({
                    "material": mat,
                    "sample": sample,
                    "temperature_C": float(r["Temperature (°C)"]),
                    "freq_hz": float(r["Angular frequency (rad/s)"]) / (2 * np.pi),
                    "storage_modulus_MPa": float(r["Storage modulus (kPa)"]) / 1000.0,
                    "loss_modulus_MPa": float(r["Loss modulus (kPa)"]) / 1000.0,
                    "tan_delta": float(r["Tan(delta)"])
                })

            order = np.argsort(temps)
            temps_s = temps[order]
            tan_s = df["Tan(delta)"].to_numpy()[order]
            e_s = (df["Storage modulus (kPa)"].to_numpy() / 1000.0)[order]
            f_s = (df["Angular frequency (rad/s)"].to_numpy() / (2 * np.pi))[order]

            # Cluster consecutive points within 1.5 C
            hold_ids = np.concatenate(([0], np.cumsum(np.diff(temps_s) > 1.5)))
            h_df = pd.DataFrame({
                "hold_id": hold_ids,
                "T": temps_s,
                "E_MPa": e_s,
                "tan_delta": tan_s,
                "freq_hz": f_s
            }).groupby("hold_id").agg({
                "T": "mean",
                "E_MPa": ["mean", "min", "max"],
                "tan_delta": ["mean", "min", "max"]
            })
            h_df.columns = ["T", "E_mean", "E_min", "E_max", "tan_d_mean", "tan_d_min", "tan_d_max"]
            h_df["T_nominal"] = np.round(h_df["T"] / 5.0) * 5.0
            h_df["material"] = mat
            h_df["sample"] = sample
            local_holds.append(h_df)

    df_local_pts = pd.DataFrame(local_pts)
    df_local_holds = pd.concat(local_holds, ignore_index=True)

    # Paper raw data
    df_paper_raw = pd.read_csv(paper_data_dir / "paper_raw_dma_measurements.csv")
    paper_holds = []
    for mat, grp in df_paper_raw.groupby("material"):
        temps = grp["temperature_C"].to_numpy()
        order = np.argsort(temps)
        temps_s = temps[order]
        tan_s = grp["tan_delta"].to_numpy()[order]
        e_s = grp["storage_modulus_MPa"].to_numpy()[order]

        hold_ids = np.concatenate(([0], np.cumsum(np.diff(temps_s) > 1.5)))
        h_df = pd.DataFrame({
            "hold_id": hold_ids,
            "T": temps_s,
            "E_MPa": e_s,
            "tan_delta": tan_s
        }).groupby("hold_id").agg({
            "T": "mean",
            "E_MPa": ["mean", "min", "max"],
            "tan_delta": ["mean", "min", "max"]
        })
        h_df.columns = ["T", "E_mean", "E_min", "E_max", "tan_d_mean", "tan_d_min", "tan_d_max"]
        h_df["T_nominal"] = np.round(h_df["T"] / 5.0) * 5.0
        h_df["material"] = mat
        paper_holds.append(h_df)

    df_paper_holds = pd.concat(paper_holds, ignore_index=True)

    return df_local_pts, df_local_holds, df_paper_raw, df_paper_holds


def plot_raw_dashboard(df_local_pts, df_local_holds, df_paper_raw, df_paper_holds, out_path: Path):
    """Create comprehensive 4-panel dashboard of raw unshifted DMA data."""
    fig = plt.figure(figsize=(16, 12), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)

    ax_e_log = fig.add_subplot(gs[0, 0])
    ax_e_lin = fig.add_subplot(gs[0, 1])
    ax_td = fig.add_subplot(gs[1, 0])
    ax_comp = fig.add_subplot(gs[1, 1])

    # ---- Panel A: Raw Storage Modulus vs. Temperature (Log Scale) ----
    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        sub = df_local_holds[df_local_holds["material"] == mat]
        grp = sub.groupby("T_nominal").agg({"E_mean": "mean", "E_min": "min", "E_max": "max"}).sort_index()
        ax_e_log.fill_between(grp.index, grp["E_min"], grp["E_max"], color=color, alpha=0.2, linewidth=0)
        ax_e_log.plot(grp.index, grp["E_mean"], color=color, linewidth=2.2, label=f"Ours: {mat}")

    for mat in ["100% VW", "50% VW", "0% VW (Agilus)"]:
        color = PAPER_COLORS[mat]
        sub = df_paper_holds[df_paper_holds["material"] == mat]
        grp = sub.groupby("T_nominal").agg({"E_mean": "mean", "E_min": "min", "E_max": "max"}).sort_index()
        ax_e_log.plot(grp.index, grp["E_mean"], color=color, linestyle="--", marker="s",
                      markersize=4, linewidth=1.6, alpha=0.85, label=f"Paper: {mat}")

    ax_e_log.set_yscale("log")
    ax_e_log.set_xlabel("Temperature, T (°C)")
    ax_e_log.set_ylabel("Raw Storage Modulus, E' (MPa)")
    ax_e_log.set_title("A. Raw Storage Modulus vs. Temperature (Log E')", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_e_log.set_xlim(-25, 115)
    ax_e_log.set_ylim(0.5, 3000)
    style_axis(ax_e_log)
    ax_e_log.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    # ---- Panel B: Raw Storage Modulus vs. Temperature (Linear Scale) ----
    paper_order = ["100% VW", "83.5% VW", "66.7% VW", "50% VW", "33.3% VW", "16.5% VW", "0% VW (Agilus)"]
    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        sub = df_local_holds[df_local_holds["material"] == mat]
        grp = sub.groupby("T_nominal")["E_mean"].mean().sort_index()
        ax_e_lin.plot(grp.index, grp.values, color=color, marker="o", markersize=4,
                      linewidth=2.0, label=f"Ours: {mat}")

    for mat in paper_order:
        color = PAPER_COLORS[mat]
        sub = df_paper_holds[df_paper_holds["material"] == mat]
        grp = sub.groupby("T_nominal")["E_mean"].mean().sort_index()
        ax_e_lin.plot(grp.index, grp.values, color=color, linestyle="--", marker="s",
                      markersize=3.5, linewidth=1.4, alpha=0.8, label=f"Paper: {mat}")

    ax_e_lin.set_xlabel("Temperature, T (°C)")
    ax_e_lin.set_ylabel("Raw Storage Modulus, E' (MPa)")
    ax_e_lin.set_title("B. Raw Storage Modulus vs. Temperature (Linear E')", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_e_lin.set_xlim(-25, 115)
    ax_e_lin.set_ylim(-50, 2400)
    style_axis(ax_e_lin)
    ax_e_lin.legend(loc="upper right", fontsize=7.0, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    # ---- Panel C: Raw Tan(delta) vs. Temperature ----
    for mat in ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]:
        color = OUR_COLORS[mat]
        sub = df_local_holds[df_local_holds["material"] == mat]
        grp = sub.groupby("T_nominal")["tan_d_mean"].mean().sort_index()
        ax_td.plot(grp.index, grp.values, color=color, marker="o", markersize=4,
                   linewidth=2.2, label=f"Ours: {mat}")

    for mat in ["100% VW", "66.7% VW", "50% VW", "33.3% VW", "0% VW (Agilus)"]:
        color = PAPER_COLORS[mat]
        sub = df_paper_holds[df_paper_holds["material"] == mat]
        grp = sub.groupby("T_nominal")["tan_d_mean"].mean().sort_index()
        ax_td.plot(grp.index, grp.values, color=color, linestyle="--", marker="s",
                   markersize=3.5, linewidth=1.5, alpha=0.85, label=f"Paper: {mat}")

    ax_td.set_xlabel("Temperature, T (°C)")
    ax_td.set_ylabel("Raw Loss Factor, tan(δ)")
    ax_td.set_title("C. Raw Loss Tangent (Damping Factor) vs. Temperature", fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_td.set_xlim(-25, 115)
    ax_td.set_ylim(-0.05, 2.4)
    style_axis(ax_td)
    ax_td.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    # ---- Panel D: Direct 1-to-1 Raw Scatter & Isothermal Spreads ----
    comp_pairs = [
        ("A0V100", "100% VW", OUR_COLORS["A0V100"], "100% VW"),
        ("A50V50", "50% VW", OUR_COLORS["A50V50"], "50% VW"),
        ("A100V0", "0% VW (Agilus)", OUR_COLORS["A100V0"], "0% VW (Agilus)"),
    ]
    for our_mat, paper_mat, color, lbl in comp_pairs:
        sub_our = df_local_pts[df_local_pts["material"] == our_mat]
        ax_comp.scatter(sub_our["temperature_C"], sub_our["storage_modulus_MPa"],
                        color=color, alpha=0.35, s=14, edgecolors="none", label=f"Ours raw: {lbl}")
        sub_paper = df_paper_raw[df_paper_raw["material"] == paper_mat]
        ax_comp.scatter(sub_paper["temperature_C"], sub_paper["storage_modulus_MPa"],
                        color=color, marker="^", alpha=0.45, s=18, edgecolors="black", linewidths=0.5,
                        label=f"Paper raw: {lbl}")

    ax_comp.set_yscale("log")
    ax_comp.set_xlabel("Temperature, T (°C)")
    ax_comp.set_ylabel("Raw Storage Modulus, E' (MPa)")
    ax_comp.set_title("D. Direct Raw Scatter: All Measured Points Across All Frequencies",
                      fontsize=11, fontweight="bold", color=INK_PRIMARY)
    ax_comp.set_xlim(-25, 115)
    ax_comp.set_ylim(0.5, 3000)
    style_axis(ax_comp)
    ax_comp.legend(loc="lower left", fontsize=7.5, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    fig.suptitle("Direct RAW Experimental Data Comparison (Without WLF / TTS Fits)\n"
                 "Local DMA-new Testing vs. Wirth et al. (Nature Communications 2025)",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.07, right=0.93, hspace=0.28, wspace=0.24)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> Saved Raw Dashboard: {out_path}")


def compute_raw_summary(df_local_holds, df_paper_holds, out_csv: Path):
    """Compute direct raw metrics at key benchmark temperatures."""
    bench_temps = [-10.0, 20.0, 50.0, 80.0]
    materials_pair = [
        ("A0V100", "100% VW"),
        ("A50V50", "50% VW"),
        ("A100V0", "0% VW (Agilus)")
    ]

    rows = []
    for our_mat, paper_mat in materials_pair:
        sub_our = df_local_holds[df_local_holds["material"] == our_mat].groupby("T_nominal").agg({
            "E_mean": "mean", "tan_d_mean": "mean"
        }).reset_index()
        sub_paper = df_paper_holds[df_paper_holds["material"] == paper_mat].groupby("T_nominal").agg({
            "E_mean": "mean", "tan_d_mean": "mean"
        }).reset_index()

        for t_target in bench_temps:
            idx_our = (sub_our["T_nominal"] - t_target).abs().argmin()
            r_our = sub_our.iloc[idx_our]
            t_actual_our = r_our["T_nominal"]
            e_our = r_our["E_mean"]
            td_our = r_our["tan_d_mean"]

            idx_p = (sub_paper["T_nominal"] - t_target).abs().argmin()
            r_p = sub_paper.iloc[idx_p]
            t_actual_p = r_p["T_nominal"]
            e_p = r_p["E_mean"]
            td_p = r_p["tan_d_mean"]

            if abs(t_actual_our - t_target) <= 6.0 and abs(t_actual_p - t_target) <= 6.0:
                e_diff_pct = (e_our - e_p) / e_p * 100.0
                rows.append({
                    "Material": our_mat,
                    "Paper_Material": paper_mat,
                    "Target_Temp_C": t_target,
                    "Ours_Actual_Temp_C": round(t_actual_our, 1),
                    "Paper_Actual_Temp_C": round(t_actual_p, 1),
                    "Ours_Raw_E_mean_MPa": round(e_our, 2),
                    "Paper_Raw_E_mean_MPa": round(e_p, 2),
                    "Raw_E_Diff_pct": round(e_diff_pct, 1),
                    "Ours_Raw_tan_delta": round(td_our, 3),
                    "Paper_Raw_tan_delta": round(td_p, 3)
                })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_csv, index=False)
    print(f"  -> Saved Raw Summary Table: {out_csv}")
    return df_summary


def main():
    root = Path(__file__).resolve().parent.parent
    dma_root = root / "raw"
    paper_data_dir = root / "reference"

    print("=" * 70)
    print("Raw DMA Data Comparison (No WLF, No Shifting, No Models)")
    print("=" * 70)

    df_local_pts, df_local_holds, df_paper_raw, df_paper_holds = load_all_raw_data(dma_root, paper_data_dir)
    print(f"Total Local Raw Points: {len(df_local_pts)} across {len(df_local_holds)} temperature holds")
    print(f"Total Paper Raw Points: {len(df_paper_raw)} across {len(df_paper_holds)} temperature holds")

    dashboard_path = root / "dma_raw_comparison_dashboard.png"
    summary_path = root / "dma_raw_comparison_summary.csv"

    plot_raw_dashboard(df_local_pts, df_local_holds, df_paper_raw, df_paper_holds, dashboard_path)
    summary_df = compute_raw_summary(df_local_holds, df_paper_holds, summary_path)

    print("\nRaw Data Benchmark Table (at selected actual test temperatures):")
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
