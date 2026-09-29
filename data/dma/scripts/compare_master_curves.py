#!/usr/bin/env python3
"""
Compare TTS master curves produced by tts_direct_wlf.py (the one-step
WLF-constrained fit -- preferred over tts_superposition.py's free per-hold
shifting, which is prone to "frozen"/degenerate shifts in flat-curvature
regions; see the tts-shifting-algorithm-limits memory note).

Produces two kinds of overlay plots (storage modulus + tan delta vs reduced
frequency, log-log for modulus):

  <Material>/processed/<Material>__replicate_comparison.png
      All replicates/samples of one material overlaid -- color = sample
      (categorical), for checking repeatability between replicates.

  <DMA_ROOT>/materials_master_curve_comparison.png
      All materials overlaid -- color = material (categorical); every
      replicate of a material shares its material's color, so replicate
      spread shows up as same-hue scatter.

tts_direct_wlf.py already drops non-positive storage/loss-modulus points
before writing its CSV, so no further point-level filtering is needed here.

Samples measured with a coarse temperature step (shakeout runs, ~10C spacing,
used only to confirm a temperature range works before the dense rerun) are
excluded entirely, detected from the actual temperature spacing in
__direct_wlf_master_curve.csv rather than hardcoded sample names. Pass
--min-density to change the cutoff, or --include-coarse to keep them in.

Usage:
  python3 compare_master_curves.py [DMA_ROOT] [--min-density 7.5] [--include-coarse]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_smoothing_spline

CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


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


def median_temp_step(temperatures: np.ndarray, gap_tol: float = 1.0):
    """Median spacing between temperature holds, clustering raw per-point
    temperatures into holds first (consecutive values within gap_tol belong
    to the same hold -- multiple frequencies at ~the same nominal T)."""
    temps = np.sort(temperatures)
    if len(temps) < 2:
        return None
    hold_ids = np.concatenate(([0], np.cumsum(np.diff(temps) > gap_tol)))
    hold_means = pd.Series(temps).groupby(hold_ids).mean().to_numpy()
    if len(hold_means) < 2:
        return None
    return float(np.median(np.diff(hold_means)))


def load_master_curve(processed_dir: Path, sample: str):
    master_path = processed_dir / f"{sample}__direct_wlf_master_curve.csv"
    if not master_path.exists():
        return None
    return pd.read_csv(master_path).sort_values("reduced_freq_hz")


def _pooled_smoothing_spline(x_all, y_all, grid):
    """Fit scipy's automatic (GCV-regularized) smoothing spline through all
    replicates' raw points pooled together, rather than averaging pre-
    interpolated per-replicate curves point-by-point -- lets one replicate's
    denser/sparser coverage at a given x contribute proportionally, and gives
    a genuinely smooth curve instead of a noisy pointwise average."""
    order = np.argsort(x_all)
    x_sorted, y_sorted = x_all[order], y_all[order]
    # make_smoothing_spline needs strictly increasing x -- average y where
    # replicates land on (near-)identical reduced frequency.
    x_unique, inverse = np.unique(x_sorted, return_inverse=True)
    y_unique = np.bincount(inverse, weights=y_sorted) / np.bincount(inverse)
    spline = make_smoothing_spline(x_unique, y_unique)
    return spline(grid)


def compute_mean_curve(dfs, n_grid=200):
    """Combine multiple replicates' master curves onto a common log-spaced
    reduced-frequency grid. The representative line is a smoothing spline
    fit through every replicate's pooled raw points (log space for storage
    modulus, linear for tan delta, since it has a peak rather than a
    monotonic trend). The min/max envelope (for the fill_between band) is
    still computed per-replicate via simple interpolation -- each replicate
    only contributes within its own measured range (no extrapolation), so
    both the spline and the envelope are undefined, and the line breaks,
    wherever no replicate has data."""
    log_min = min(np.log10(df["reduced_freq_hz"].min()) for df in dfs)
    log_max = max(np.log10(df["reduced_freq_hz"].max()) for df in dfs)
    log_grid = np.linspace(log_min, log_max, n_grid)

    storage_stack, tan_stack = [], []
    pooled_log_x, pooled_log_storage, pooled_tan = [], [], []
    for df in dfs:
        x = np.log10(df["reduced_freq_hz"].to_numpy())
        order = np.argsort(x)
        x_sorted = x[order]
        log_storage_sorted = np.log10(df["storage_modulus_kPa"].to_numpy())[order]
        tan_sorted = df["tan_delta"].to_numpy()[order]
        storage_stack.append(np.interp(log_grid, x_sorted, log_storage_sorted, left=np.nan, right=np.nan))
        tan_stack.append(np.interp(log_grid, x_sorted, tan_sorted, left=np.nan, right=np.nan))
        pooled_log_x.append(x_sorted)
        pooled_log_storage.append(log_storage_sorted)
        pooled_tan.append(tan_sorted)
    storage_arr = np.array(storage_stack)
    tan_arr = np.array(tan_stack)

    pooled_log_x = np.concatenate(pooled_log_x)
    spline_log_storage = _pooled_smoothing_spline(pooled_log_x, np.concatenate(pooled_log_storage), log_grid)
    # Tan(delta) can't be physically negative; individual noisy points may
    # dip slightly below zero (still shown honestly in the envelope), but
    # clip the smoothing spline's edge overshoot so the representative curve
    # doesn't imply a nonphysical value.
    spline_tan = np.clip(_pooled_smoothing_spline(pooled_log_x, np.concatenate(pooled_tan), log_grid), 0, None)

    with np.errstate(invalid="ignore"):
        all_nan_storage = np.all(np.isnan(storage_arr), axis=0)
        all_nan_tan = np.all(np.isnan(tan_arr), axis=0)
        # min/max envelope across replicates -- the "surface" of experimental
        # spread a fill_between band is drawn over, per reduced-frequency grid point.
        min_log_storage = np.where(all_nan_storage, np.nan, np.nanmin(storage_arr, axis=0))
        max_log_storage = np.where(all_nan_storage, np.nan, np.nanmax(storage_arr, axis=0))
        min_tan = np.where(all_nan_tan, np.nan, np.nanmin(tan_arr, axis=0))
        max_tan = np.where(all_nan_tan, np.nan, np.nanmax(tan_arr, axis=0))
        mean_log_storage = np.where(all_nan_storage, np.nan, spline_log_storage)
        mean_tan = np.where(all_nan_tan, np.nan, spline_tan)
    return pd.DataFrame({
        "reduced_freq_hz": 10 ** log_grid,
        "storage_modulus_kPa": 10 ** mean_log_storage,
        "storage_modulus_min_kPa": 10 ** min_log_storage,
        "storage_modulus_max_kPa": 10 ** max_log_storage,
        "tan_delta": mean_tan,
        "tan_delta_min": min_tan,
        "tan_delta_max": max_tan,
    }).dropna(subset=["storage_modulus_kPa"], how="all")


def representative_curve(dfs):
    """The curve to represent a material with: the mean across replicates
    when there are >=2 (dense, qualifying) samples, or just the single
    sample's own curve when there's only one -- there's nothing to average."""
    return compute_mean_curve(dfs) if len(dfs) >= 2 else dfs[0]


def plot_replicate_overlay(dfs, color, out_path: Path, title: str):
    """The min-max envelope across replicates drawn as a faint (alpha=0.5)
    fill_between band; their mean drawn as a solid, full-opacity line on top."""
    fig, (ax_mod, ax_tan) = plt.subplots(
        1, 2, figsize=(13, 5.5), facecolor=SURFACE,
    )

    mean_df = compute_mean_curve(dfs)
    envelope_label = f"Replicate range (n={len(dfs)})"
    ax_mod.fill_between(mean_df["reduced_freq_hz"], mean_df["storage_modulus_min_kPa"],
                         mean_df["storage_modulus_max_kPa"], color=color, alpha=0.5,
                         linewidth=0, label=envelope_label, zorder=2)
    ax_tan.fill_between(mean_df["reduced_freq_hz"], mean_df["tan_delta_min"],
                         mean_df["tan_delta_max"], color=color, alpha=0.5,
                         linewidth=0, label=envelope_label, zorder=2)

    mean_label = f"Spline fit (n={len(dfs)})"
    ax_mod.plot(mean_df["reduced_freq_hz"], mean_df["storage_modulus_kPa"], color=color, linewidth=2.4,
                linestyle="-", alpha=1.0, label=mean_label, zorder=4)
    ax_tan.plot(mean_df["reduced_freq_hz"], mean_df["tan_delta"], color=color, linewidth=2.4,
                linestyle="-", alpha=1.0, label=mean_label, zorder=4)

    ax_mod.set_xscale("log")
    ax_mod.set_yscale("log")
    ax_mod.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_mod.set_ylabel("Storage modulus, E' (kPa)")
    style_axis(ax_mod)
    ax_mod.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    ax_tan.set_xscale("log")
    ax_tan.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_tan.set_ylabel("Tan(delta)")
    style_axis(ax_tan)
    ax_tan.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_overlay(groups, color_for, out_path: Path, title: str, legend_title: str, background_groups=()):
    """groups: list of (label, color_key, dataframe) to draw as the main,
    labeled lines. background_groups: list of (color_key, envelope_dataframe)
    -- envelope_dataframe must carry the min/max columns compute_mean_curve
    produces -- drawn as a faint (alpha=0.5, no legend entry) fill_between
    band underneath, e.g. a material's replicate spread behind its mean."""
    fig, (ax_mod, ax_tan) = plt.subplots(
        1, 2, figsize=(13, 5.5), facecolor=SURFACE,
    )

    for color_key, df in background_groups:
        color = color_for(color_key)
        ax_mod.fill_between(df["reduced_freq_hz"], df["storage_modulus_min_kPa"],
                             df["storage_modulus_max_kPa"], color=color, alpha=0.5,
                             linewidth=0, zorder=1)
        ax_tan.fill_between(df["reduced_freq_hz"], df["tan_delta_min"],
                             df["tan_delta_max"], color=color, alpha=0.5,
                             linewidth=0, zorder=1)

    seen_labels = set()
    for label, color_key, df in groups:
        color = color_for(color_key)
        storage = df["storage_modulus_kPa"].where(df["storage_modulus_kPa"] > 0)
        show_label = label not in seen_labels
        seen_labels.add(label)
        ax_mod.plot(df["reduced_freq_hz"], storage, color=color, linewidth=1.6,
                    marker="o", markersize=3, linestyle="-", alpha=0.9,
                    label=label if show_label else None, zorder=3)
        ax_tan.plot(df["reduced_freq_hz"], df["tan_delta"], color=color, linewidth=1.6,
                    marker="o", markersize=3, linestyle="-", alpha=0.9,
                    label=label if show_label else None, zorder=3)

    ax_mod.set_xscale("log")
    ax_mod.set_yscale("log")
    ax_mod.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_mod.set_ylabel("Storage modulus, E' (kPa)")
    style_axis(ax_mod)
    ax_mod.legend(title=legend_title, loc="upper left", fontsize=8,
                  title_fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    ax_tan.set_xscale("log")
    ax_tan.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_tan.set_ylabel("Tan(delta)")
    style_axis(ax_tan)
    ax_tan.legend(title=legend_title, loc="upper left", fontsize=8,
                  title_fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(title, color=INK_PRIMARY, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=None,
                         help="DMA root directory (default: parent directory of this script)")
    parser.add_argument("--min-density", type=float, default=7.5,
                         help="Minimum temperature-step density (°C) to include a sample -- "
                              "samples with a coarser median step (shakeout runs) are excluded "
                              "(default 7.5, i.e. excludes ~10°C-step runs, keeps ~5°C-step runs)")
    parser.add_argument("--include-coarse", action="store_true",
                         help="Include coarse-step (shakeout) samples anyway")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    print(f"Scanning DMA root: {root}")

    material_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "scripts")
    material_color = {m.name: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(material_dirs)}

    all_material_groups = []
    all_material_background = []

    for material_dir in material_dirs:
        processed_dir = material_dir / "processed"
        if not processed_dir.is_dir():
            continue
        master_files = sorted(processed_dir.glob("*__direct_wlf_master_curve.csv"))
        if not master_files:
            continue

        replicate_groups = []
        for mf in master_files:
            sample = mf.name[: -len("__direct_wlf_master_curve.csv")]

            df = load_master_curve(processed_dir, sample)
            if df is None or df.empty:
                continue

            step = median_temp_step(df["temperature_C"].to_numpy())
            if step is not None and step >= args.min_density and not args.include_coarse:
                print(f"  [{material_dir.name}/{sample}] excluded: coarse temperature step "
                      f"(~{step:.1f}°C median, shakeout run) -- not a dense final run")
                continue
            replicate_groups.append((sample, sample, df))

        if replicate_groups:
            rep_df = representative_curve([g[2] for g in replicate_groups])
            all_material_groups.append((material_dir.name, material_dir.name, rep_df))
            if len(replicate_groups) >= 2:
                # rep_df already carries the min/max envelope columns from
                # compute_mean_curve -- only meaningful with >=2 replicates.
                all_material_background.append((material_dir.name, rep_df))

        if len(replicate_groups) < 2:
            print(f"Material: {material_dir.name} -- fewer than 2 samples with master curves, skipping replicate comparison")
            continue

        print(f"Material: {material_dir.name} -- comparing {len(replicate_groups)} replicate(s)")
        out_path = processed_dir / f"{material_dir.name}__replicate_comparison.png"
        plot_replicate_overlay(
            [g[2] for g in replicate_groups], material_color[material_dir.name], out_path,
            title=f"{material_dir.name}: replicate master curve comparison",
        )
        print(f"  -> {out_path}")

    if len({g[0] for g in all_material_groups}) >= 2:
        out_path = root / "materials_master_curve_comparison.png"
        print(f"Comparing {len({g[0] for g in all_material_groups})} material(s)")
        plot_overlay(
            all_material_groups, lambda k: material_color[k], out_path,
            title="Master curve comparison across materials",
            legend_title="Material", background_groups=all_material_background,
        )
        print(f"  -> {out_path}")
    else:
        print("Fewer than 2 materials with master curves -- skipping cross-material comparison")


if __name__ == "__main__":
    main()
