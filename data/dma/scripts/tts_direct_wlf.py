#!/usr/bin/env python3
"""
One-step WLF-constrained TTS fit -- an alternative to the free per-hold
shifting in tts_superposition.py.

Rather than fitting an independent shift factor per temperature hold and
*then* fitting WLF's C1/C2 to those shifts (which can go "frozen"/degenerate
when a hold's own frequency-dependence is too weak to constrain a free
shift -- see the tts-shifting-algorithm-limits memory note), this fits C1
and C2 directly against the ENTIRE raw dataset in one regression. Every
point's reduced frequency is a direct function of (C1, C2); there is no
per-hold free parameter left to get stuck on a degenerate local minimum.

Method: profile/alternating optimization over (C1, C2) --
  1. Given a trial (C1, C2), compute every point's reduced frequency.
  2. Fit the best non-decreasing curve through (log10(reduced_freq), log10(E'))
     via isotonic regression (pool-adjacent-violators) -- this is the implied
     master curve shape, with no free parameters of its own.
  3. Residual = actual log10(E') - isotonic-fitted value; total sum of
     squares is the objective scipy.optimize.minimize adjusts (C1, C2) to
     minimize.
This uses every valid point across every hold at once, so a hold with weak
curvature is still constrained by its neighbors through the shared shape fit
-- unlike per-hold free shifting, there's no way for one hold to wander off
independently.

Caveat: this assumes WLF holds across the fitted temperature range. Use
--wlf-min-temp/--wlf-max-temp to exclude a sub-Tg/secondary-relaxation region
that wouldn't follow WLF (same idea as in tts_superposition.py, but here it
restricts which raw points enter the regression, not which holds enter a
downstream fit).

Outputs (per sample, in <Material>/processed/):
  <Sample>__direct_wlf_master_curve.csv   reduced_freq_hz, temperature_C, E', E'', tan(delta)
  <Sample>__direct_wlf_fit.png            data (colored by T) + fitted master-curve shape

Usage:
  python3 tts_direct_wlf.py [DMA_ROOT] [--ref-temp 25] [--wlf-min-temp T] [--wlf-max-temp T]
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEQ_BLUE_STOPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
ACCENT_FIT = "#e34948"


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def isotonic_regression(x, y, increasing=True):
    """Pool-adjacent-violators fit of the best monotonic curve through (x, y)."""
    order = np.argsort(x)
    vals = y[order].astype(float)
    if not increasing:
        vals = -vals
    levels = list(vals)
    weights = [1.0] * len(levels)
    i = 0
    while i < len(levels) - 1:
        if levels[i] > levels[i + 1]:
            merged = (levels[i] * weights[i] + levels[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
            levels[i:i + 2] = [merged]
            weights[i:i + 2] = [weights[i] + weights[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted_blocks = []
    for lvl, w in zip(levels, weights):
        fitted_blocks.extend([lvl] * int(round(w)))
    fitted = np.empty(len(y))
    fitted[order] = fitted_blocks
    if not increasing:
        fitted = -fitted
    return fitted


def wlf_log_at(t_minus_tref, c1, c2, pole_margin=1.0):
    denom = c2 + t_minus_tref
    if np.any(denom <= pole_margin):
        return None  # too close to (or past) the WLF pole -- not a valid parameterization
    return -c1 * t_minus_tref / denom


def fit_direct_wlf(log_f, log_e, t_minus_tref, c1_init=17.4, c2_init=100.0):
    def objective(params):
        c1, c2 = params
        if c1 <= 0 or c2 <= 0:
            return 1e12
        log_at = wlf_log_at(t_minus_tref, c1, c2)
        if log_at is None:
            return 1e12
        reduced_x = log_f + log_at
        fitted = isotonic_regression(reduced_x, log_e, increasing=True)
        return float(np.sum((log_e - fitted) ** 2))

    res = minimize(objective, x0=[c1_init, c2_init], method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    return res.x, res.fun


def process_sample(csv_path: Path, sample: str, material: str, ref_temp: float,
                    wlf_min_temp=None, wlf_max_temp=None):
    df = pd.read_csv(csv_path)
    df["freq_hz"] = df["Angular frequency (rad/s)"] / (2 * math.pi)

    valid = (df["Storage modulus (kPa)"] > 0) & (df["Loss modulus (kPa)"] > 0)
    n_dropped = int((~valid).sum())
    df_valid = df.loc[valid].copy()

    temps = df_valid["Temperature (°C)"].to_numpy()
    ref_temp_actual = temps[np.argmin(np.abs(temps - ref_temp))]

    fit_mask = np.ones(len(df_valid), dtype=bool)
    if wlf_min_temp is not None:
        fit_mask &= temps >= wlf_min_temp
    if wlf_max_temp is not None:
        fit_mask &= temps <= wlf_max_temp

    log_f_all = np.log10(df_valid["freq_hz"].to_numpy())
    log_e_all = np.log10(df_valid["Storage modulus (kPa)"].to_numpy())
    t_minus_tref_all = temps - ref_temp_actual

    (c1, c2), sse = fit_direct_wlf(log_f_all[fit_mask], log_e_all[fit_mask], t_minus_tref_all[fit_mask])

    log_at_all = wlf_log_at(t_minus_tref_all, c1, c2)
    reduced_freq_all = df_valid["freq_hz"].to_numpy() * 10 ** log_at_all

    out_df = pd.DataFrame({
        "reduced_freq_hz": reduced_freq_all,
        "temperature_C": temps,
        "storage_modulus_kPa": df_valid["Storage modulus (kPa)"].to_numpy(),
        "loss_modulus_kPa": df_valid["Loss modulus (kPa)"].to_numpy(),
        "tan_delta": df_valid["Tan(delta)"].to_numpy(),
        "used_in_fit": fit_mask,
    }).sort_values("reduced_freq_hz")

    out_dir = csv_path.parent
    out_csv = out_dir / f"{sample}__direct_wlf_master_curve.csv"
    out_df.to_csv(out_csv, index=False)

    rmse = math.sqrt(sse / fit_mask.sum())
    print(f"  [{material}/{sample}] Tref={ref_temp_actual:.1f}°C, {len(df_valid)} points "
          f"({fit_mask.sum()} used in fit) -- C1={c1:.3g}, C2={c2:.3g}, RMSE(log10 E')={rmse:.3f}")
    if n_dropped:
        print(f"    dropped {n_dropped} non-positive storage/loss-modulus point(s)")

    out_png = out_dir / f"{sample}__direct_wlf_fit.png"
    plot_direct_wlf(out_df, sample, material, ref_temp_actual, c1, c2, out_png)
    print(f"    -> {out_csv.name}, {out_png.name}")


def plot_direct_wlf(out_df: pd.DataFrame, sample: str, material: str, ref_temp_actual: float,
                     c1: float, c2: float, out_path: Path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STOPS)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    tmin, tmax = out_df["temperature_C"].min(), out_df["temperature_C"].max()
    norm = plt.Normalize(tmin, tmax)

    used = out_df["used_in_fit"]
    sc = ax.scatter(out_df.loc[used, "reduced_freq_hz"], out_df.loc[used, "storage_modulus_kPa"],
                     c=out_df.loc[used, "temperature_C"], cmap=cmap, norm=norm,
                     marker="o", s=22, linewidths=0, zorder=3, label="Used in fit")
    if (~used).any():
        ax.scatter(out_df.loc[~used, "reduced_freq_hz"], out_df.loc[~used, "storage_modulus_kPa"],
                   facecolors="none", edgecolors=ACCENT_FIT, linewidths=1.2, s=30, zorder=4,
                   label="Excluded from fit (--wlf-min/max-temp)")

    x = np.log10(out_df["reduced_freq_hz"].to_numpy())
    y = np.log10(out_df["storage_modulus_kPa"].to_numpy())
    fitted = isotonic_regression(x, y, increasing=True)
    order = np.argsort(x)
    ax.plot(10 ** x[order], 10 ** fitted[order], color=ACCENT_FIT, linewidth=1.8, zorder=5,
            label="Fitted master-curve shape")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax.set_ylabel("Storage modulus, E' (kPa)")
    style_axis(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.suptitle(f"{material} — {sample}: direct WLF fit "
                 f"(Tref={ref_temp_actual:.1f}°C, C1={c1:.3g}, C2={c2:.3g})",
                 color=INK_PRIMARY, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=None,
                         help="DMA root directory (default: parent directory of this script)")
    parser.add_argument("--ref-temp", type=float, default=25.0,
                         help="Reference temperature in °C (default 25)")
    parser.add_argument("--wlf-min-temp", type=float, default=None,
                         help="Exclude raw points below this temperature (°C) from the fit")
    parser.add_argument("--wlf-max-temp", type=float, default=None,
                         help="Exclude raw points above this temperature (°C) from the fit")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    print(f"Scanning DMA root: {root} (reference temperature target: {args.ref_temp} °C)")

    for material_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "scripts"):
        processed_dir = material_dir / "processed"
        if not processed_dir.is_dir():
            continue
        sweep_files = sorted(processed_dir.glob("*__temp_sweep.csv"))
        if not sweep_files:
            continue
        print(f"Material: {material_dir.name}")
        for sweep_csv in sweep_files:
            sample = sweep_csv.name[: -len("__temp_sweep.csv")]
            process_sample(sweep_csv, sample, material_dir.name, args.ref_temp,
                            args.wlf_min_temp, args.wlf_max_temp)


if __name__ == "__main__":
    main()
