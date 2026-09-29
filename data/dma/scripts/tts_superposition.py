#!/usr/bin/env python3
"""
Time-Temperature Superposition (TTS) on the oscillatory temperature-sweep data
produced by parse_dma.py.

For each <Sample>__temp_sweep.csv, groups the sweep into its per-temperature
frequency scans (7 frequencies per hold in this procedure), picks a reference
temperature, and horizontally shifts each hold's (log frequency, log E')
curve to build a single master curve -- the standard sequential/incremental
TTS algorithm (shift outward from the reference, always fitting against the
master curve accumulated so far). The same shift factor a_T is then applied
to the loss modulus and tan(delta) (horizontal shift only; no vertical
density/modulus correction b_T is applied).

A WLF fit (log10(aT) = -C1*(T-Tref) / (C2+(T-Tref))) is attempted on the
resulting shift factors as a sanity check / interpolation aid.

Outputs (per sample, in <Material>/processed/):
  <Sample>__shift_factors.csv     Temperature, log_aT, aT
  <Sample>__master_curve.csv      reduced_freq_hz, source_temperature_C, E', E'', tan(delta)
  <Sample>__tts_master_curve.png  master curve (log-log) + shift-factor/WLF plot

Usage:
  python3 tts_superposition.py [DMA_ROOT] [--ref-temp 25]
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

# --- dataviz palette (sequential blue ramp, for the continuous Temperature encoding) ---
SEQ_BLUE_STOPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
ACCENT_FIT = "#e34948"  # WLF fit line (status-distinct red, single series contrast to blue ramp)


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


def interp_extrap(x, xp, fp):
    """Piecewise-linear interpolation with linear extrapolation past the ends."""
    order = np.argsort(xp)
    xp, fp = np.asarray(xp)[order], np.asarray(fp)[order]
    y = np.interp(x, xp, fp)
    x = np.asarray(x, dtype=float)
    if len(xp) >= 2:
        below = x < xp[0]
        if np.any(below):
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            y = np.where(below, fp[0] + slope * (x - xp[0]), y)
        above = x > xp[-1]
        if np.any(above):
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            y = np.where(above, fp[-1] + slope * (x - xp[-1]), y)
    return y


def group_by_temperature_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a hold/group id: a new hold starts whenever frequency resets
    back down to the lowest value in the sweep (each temperature hold cycles
    through the same ascending frequency list)."""
    df = df.sort_values("Step time (s)").reset_index(drop=True)
    first_freq = df["freq_hz"].iloc[0]
    is_reset = pd.Series(np.isclose(df["freq_hz"], first_freq, rtol=1e-3))
    is_reset.iloc[0] = True
    df["hold_id"] = is_reset.cumsum() - 1
    return df


def wlf(t_minus_tref, c1, c2):
    return -c1 * t_minus_tref / (c2 + t_minus_tref)


def flag_isolated_holds(temperatures: np.ndarray, gap_factor: float = 2.5):
    """A hold is 'isolated' if its nearest-neighbor temperature gap is far
    wider than the typical spacing -- its shift factor would rest on
    extrapolation rather than real curve overlap, so it should be excluded
    from any WLF fit (still reported, just flagged low-confidence)."""
    order = np.argsort(temperatures)
    t_sorted = temperatures[order]
    if len(t_sorted) < 3:
        return np.zeros(len(temperatures), dtype=bool)
    consecutive_gaps = np.diff(t_sorted)
    median_gap = np.median(consecutive_gaps)
    nearest_gap = np.full(len(t_sorted), np.inf)
    nearest_gap[1:] = np.minimum(nearest_gap[1:], consecutive_gaps)
    nearest_gap[:-1] = np.minimum(nearest_gap[:-1], consecutive_gaps)
    isolated_sorted = nearest_gap > gap_factor * median_gap
    isolated = np.zeros(len(temperatures), dtype=bool)
    isolated[order] = isolated_sorted
    return isolated


def _isotonic_nonincreasing(x, y):
    """Pool-adjacent-violators fit of the largest non-increasing curve through
    (x, y) in weighted least squares -- dependency-free (no sklearn needed)."""
    order = np.argsort(x)
    levels = list(y[order].astype(float))
    weights = [1.0] * len(levels)
    i = 0
    while i < len(levels) - 1:
        if levels[i] < levels[i + 1]:
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
    return fitted


def flag_nonmonotonic_holds(temperatures: np.ndarray, log_at: np.ndarray, is_ref: np.ndarray,
                             residual_threshold: float = 1.0):
    """A physically valid single-mechanism shift factor is non-increasing in
    T. Fit the best non-increasing curve through the data (isotonic
    regression) and flag any point whose residual from it exceeds
    `residual_threshold` decades -- these don't belong to a consistent
    monotonic trend (aliased/mis-shifted holds from sparse data, or a genuine
    secondary relaxation) and shouldn't be trusted for a WLF fit. The
    reference hold (log_aT fixed at 0 by definition, not fit) is included so
    it can anchor/constrain the fit, but is never itself flagged."""
    valid = np.isfinite(log_at)
    flags = np.zeros(len(log_at), dtype=bool)
    if valid.sum() < 4:
        return flags
    fitted = _isotonic_nonincreasing(temperatures[valid], log_at[valid])
    residual = np.abs(log_at[valid] - fitted)
    valid_flags = residual > residual_threshold
    valid_flags[is_ref[valid]] = False
    flags[valid] = valid_flags
    return flags


def flag_frozen_holds(temperatures: np.ndarray, log_at: np.ndarray, is_ref: np.ndarray,
                       tol: float = 1e-4):
    """A genuine WLF-like shift factor is never exactly (or near-exactly)
    repeated at two different temperatures -- if the per-hold curve has too
    little frequency-dependence to discriminate a shift (e.g. deep in a flat
    rubbery/melt plateau), the bounded optimizer can get stuck re-finding the
    same degenerate local minimum for consecutive holds. This looks perfectly
    monotonic (flat counts as non-increasing) so flag_nonmonotonic_holds
    misses it entirely; catch it separately by looking for near-identical
    consecutive values in temperature order."""
    order = np.argsort(temperatures)
    flags_sorted = np.zeros(len(temperatures), dtype=bool)
    valid_sorted = np.isfinite(log_at[order])
    is_ref_sorted = is_ref[order]
    idx = np.where(valid_sorted)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if is_ref_sorted[a] or is_ref_sorted[b]:
            continue
        if abs(log_at[order][a] - log_at[order][b]) < tol:
            flags_sorted[a] = True
            flags_sorted[b] = True
    flags = np.zeros(len(temperatures), dtype=bool)
    flags[order] = flags_sorted
    return flags


def _fit_shift(log_f, log_e, master_x, master_y, bounds=(-15, 15)):
    def objective(log_at):
        pred = interp_extrap(log_f + log_at, master_x, master_y)
        return float(np.sum((log_e - pred) ** 2))
    return minimize_scalar(objective, bounds=bounds, method="bounded").x


def build_master_curve(df: pd.DataFrame, ref_temp: float):
    holds = (
        df.groupby("hold_id")
        .agg(temperature=("Temperature (°C)", "mean"))
        .reset_index()
    )
    holds["isolated"] = flag_isolated_holds(holds["temperature"].to_numpy())
    ref_hold_id = holds.iloc[(holds["temperature"] - ref_temp).abs().argsort().iloc[0]]["hold_id"]
    ref_temp_actual = holds.loc[holds["hold_id"] == ref_hold_id, "temperature"].iloc[0]

    order = holds.assign(dist=(holds["temperature"] - ref_temp_actual).abs()).sort_values("dist")
    hold_order = [int(h) for h in order["hold_id"]]
    isolated_by_hold = dict(zip(holds["hold_id"], holds["isolated"]))
    temp_by_hold = dict(zip(holds["hold_id"], holds["temperature"]))

    # A non-positive storage OR loss modulus is a single-point instrument-noise
    # artifact -- log10 of a non-positive storage modulus is undefined, and a
    # negative loss modulus (even alongside a positive storage modulus, as
    # seen in A0V100-2) is just as clear a sign the point is garbage, often
    # dragging its storage-modulus reading off-trend too. Both must be
    # dropped before shift-fitting, not just masked for display. The raw
    # CSV/output rows still carry the original values either way.
    hold_points = {}  # hold_id -> (log_f, log_e) of valid points
    for hold_id in hold_order:
        hold = df[df["hold_id"] == hold_id].sort_values("freq_hz")
        valid = (hold["Storage modulus (kPa)"] > 0) & (hold["Loss modulus (kPa)"] > 0)
        hold_points[hold_id] = (
            np.log10(hold.loc[valid, "freq_hz"].to_numpy()),
            np.log10(hold.loc[valid, "Storage modulus (kPa)"].to_numpy()),
        )
        if not valid.all():
            print(f"    dropped {(~valid).sum()} non-positive storage/loss-modulus point(s) "
                  f"at {temp_by_hold[hold_id]:.1f}°C from shift-fitting")

    # Sequential shift: each hold, outward from the reference, is fit against
    # whatever partial master curve has been accumulated so far.
    log_at = {}
    master_x, master_y = [], []
    for hold_id in hold_order:
        log_f, log_e = hold_points[hold_id]
        if hold_id == ref_hold_id:
            log_at[hold_id] = 0.0
        elif len(log_f) == 0:
            log_at[hold_id] = np.nan
        else:
            log_at[hold_id] = _fit_shift(log_f, log_e, master_x, master_y)
        if np.isfinite(log_at[hold_id]):
            master_x.extend((log_f + log_at[hold_id]).tolist())
            master_y.extend(log_e.tolist())

    # NOTE: a leave-one-out refinement pass (re-fitting each hold against a
    # master curve built from every other hold) was tried here and reverted --
    # tested empirically, it made known-bad cases worse, not better, because a
    # smooth, sparsely-sampled modulus curve is easy to alias (shifting by
    # roughly one log-frequency spacing can match another part of the curve
    # almost as well), and joint refinement can converge to a *different*,
    # mutually self-consistent but wrong set of shifts. The one-pass sequential
    # estimate above is kept as final; flag_nonmonotonic_holds() below instead
    # detects, after the fact, which holds it still got wrong.
    hold_ids_sorted = list(log_at.keys())
    temps_arr = np.array([temp_by_hold[h] for h in hold_ids_sorted])
    log_at_arr = np.array([log_at[h] for h in hold_ids_sorted])
    is_ref_arr = np.array([h == ref_hold_id for h in hold_ids_sorted])
    nonmonotonic = flag_nonmonotonic_holds(temps_arr, log_at_arr, is_ref_arr)
    nonmonotonic_by_hold = dict(zip(hold_ids_sorted, nonmonotonic))
    frozen = flag_frozen_holds(temps_arr, log_at_arr, is_ref_arr)
    frozen_by_hold = dict(zip(hold_ids_sorted, frozen))

    shift_rows = []
    all_rows = []
    for hold_id in hold_order:
        hold = df[df["hold_id"] == hold_id].sort_values("freq_hz")
        temp_actual = temp_by_hold[hold_id]
        la = log_at[hold_id]
        at = 10 ** la if np.isfinite(la) else np.nan
        if not np.isfinite(la) and hold_id != ref_hold_id:
            print(f"    WARNING: hold at {temp_actual:.1f}°C has no valid (positive) "
                  f"storage modulus points -- shift factor could not be determined.")

        is_isolated = bool(isolated_by_hold[hold_id])
        is_nonmonotonic = bool(nonmonotonic_by_hold[hold_id])
        is_frozen = bool(frozen_by_hold[hold_id])
        is_invalid = not np.isfinite(la)
        reasons = []
        if is_isolated:
            reasons.append("isolated")
        if is_nonmonotonic:
            reasons.append("nonmonotonic")
        if is_frozen:
            reasons.append("frozen")
        if is_invalid:
            reasons.append("no_valid_points")

        shift_rows.append({
            "temperature_C": temp_actual, "log_aT": la, "aT": at,
            "isolated_low_confidence": is_isolated or is_nonmonotonic or is_frozen or is_invalid,
            "low_confidence_reason": ";".join(reasons),
        })

        reduced_freq = hold["freq_hz"].to_numpy() * at
        for rf, t, ep, epp, tand in zip(
            reduced_freq, [temp_actual] * len(hold),
            hold["Storage modulus (kPa)"], hold["Loss modulus (kPa)"], hold["Tan(delta)"]
        ):
            all_rows.append({
                "reduced_freq_hz": rf,
                "source_temperature_C": t,
                "storage_modulus_kPa": ep,
                "loss_modulus_kPa": epp,
                "tan_delta": tand,
            })

    shift_df = pd.DataFrame(shift_rows).sort_values("temperature_C").reset_index(drop=True)
    master_df = pd.DataFrame(all_rows).sort_values("reduced_freq_hz").reset_index(drop=True)
    return shift_df, master_df, ref_temp_actual


def fit_wlf(shift_df: pd.DataFrame, ref_temp_actual: float, wlf_min_temp=None, wlf_max_temp=None):
    t_minus_tref = (shift_df["temperature_C"] - ref_temp_actual).to_numpy()
    log_at = shift_df["log_aT"].to_numpy()
    mask = (t_minus_tref != 0) & (~shift_df["isolated_low_confidence"].to_numpy())
    # A secondary (sub-Tg) relaxation has a different temperature dependence than
    # the primary transition; mixing both into one WLF fit is what breaks it.
    # --wlf-min-temp/--wlf-max-temp let the caller restrict the fit to the
    # single-mechanism region without touching the master-curve shifting itself.
    if wlf_min_temp is not None:
        mask &= shift_df["temperature_C"].to_numpy() >= wlf_min_temp
    if wlf_max_temp is not None:
        mask &= shift_df["temperature_C"].to_numpy() <= wlf_max_temp
    if mask.sum() < 4:
        return None
    try:
        popt, pcov = curve_fit(wlf, t_minus_tref[mask], log_at[mask], p0=[10.0, 100.0], maxfev=10000)
    except Exception:
        return None
    perr = np.sqrt(np.diag(pcov))
    # WLF's two parameters trade off (only their ratio is pinned) when the fitted
    # temperature range has too little curvature to separate them -- catch that
    # ill-conditioned case rather than reporting meaningless C1/C2 values.
    if not np.all(np.isfinite(perr)) or np.any(perr > 5 * np.abs(popt)):
        return None
    return popt  # C1, C2


def temperature_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STOPS)


def plot_tts(sample: str, material: str, master_df: pd.DataFrame, shift_df: pd.DataFrame,
             ref_temp_actual: float, wlf_params, out_path: Path):
    fig, (ax_master, ax_shift) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

    cmap = temperature_colormap()
    tmin, tmax = master_df["source_temperature_C"].min(), master_df["source_temperature_C"].max()
    norm = plt.Normalize(tmin, tmax)

    for storage_or_loss, marker, ls in (("storage_modulus_kPa", "o", "-"), ("loss_modulus_kPa", "^", "--")):
        vals = master_df[storage_or_loss].where(master_df[storage_or_loss] > 0)
        sc = ax_master.scatter(
            master_df["reduced_freq_hz"], vals,
            c=master_df["source_temperature_C"], cmap=cmap, norm=norm,
            marker=marker, s=22, linewidths=0, zorder=3,
        )
    ax_master.set_xscale("log")
    ax_master.set_yscale("log")
    ax_master.set_xlabel("Reduced frequency, f · aT (Hz)")
    ax_master.set_ylabel("Modulus (kPa)")
    style_axis(ax_master)
    marker_handles = [
        plt.Line2D([0], [0], color=INK_SECONDARY, marker="o", linestyle="none", markersize=6, label="Storage modulus (E')"),
        plt.Line2D([0], [0], color=INK_SECONDARY, marker="^", linestyle="none", markersize=6, label="Loss modulus (E'')"),
    ]
    ax_master.legend(handles=marker_handles, loc="lower left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    cbar = fig.colorbar(sc, ax=ax_master, pad=0.02)
    cbar.set_label("Temperature (°C)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    t_minus_tref = shift_df["temperature_C"] - ref_temp_actual
    isolated = shift_df["isolated_low_confidence"].to_numpy()
    ax_shift.scatter(t_minus_tref[~isolated], shift_df["log_aT"][~isolated],
                      color=SEQ_BLUE_STOPS[3], s=28, zorder=3, label="Fitted shift factor")
    if isolated.any():
        ax_shift.scatter(t_minus_tref[isolated], shift_df["log_aT"][isolated],
                          facecolors="none", edgecolors=ACCENT_FIT, linewidths=1.5, s=50, zorder=4,
                          label="Low-confidence (isolated/non-monotonic, excluded from fit)")
    if wlf_params is not None:
        c1, c2 = wlf_params
        xs = np.linspace(t_minus_tref.min(), t_minus_tref.max(), 200)
        # avoid the WLF pole if c2 falls inside the data range
        xs = xs[np.abs(c2 + xs) > 1e-6]
        ax_shift.plot(xs, wlf(xs, c1, c2), color=ACCENT_FIT, linewidth=1.8, zorder=2,
                      label=f"WLF fit (C1={c1:.3g}, C2={c2:.3g})")
    ax_shift.axvline(0, color=BASELINE, linewidth=1, zorder=1)
    ax_shift.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax_shift.set_xlabel(f"T − Tref (°C), Tref = {ref_temp_actual:.1f} °C")
    ax_shift.set_ylabel("log10(aT)")
    style_axis(ax_shift)
    ax_shift.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(f"{material} — {sample}: TTS master curve (Tref = {ref_temp_actual:.1f} °C)",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def process_sample(csv_path: Path, sample: str, material: str, ref_temp: float,
                    wlf_min_temp=None, wlf_max_temp=None, temp_min=None, temp_max=None,
                    hold_stride=1):
    df = pd.read_csv(csv_path)
    df["freq_hz"] = df["Angular frequency (rad/s)"] / (2 * math.pi)
    df = group_by_temperature_hold(df)

    if temp_min is not None:
        df = df[df["Temperature (°C)"] >= temp_min]
    if temp_max is not None:
        df = df[df["Temperature (°C)"] <= temp_max]

    if hold_stride > 1:
        # Simulate a coarser temperature step (e.g. stride=2 turns 5C steps into
        # effective 10C steps) by keeping every Nth hold, in temperature order.
        hold_temps = df.groupby("hold_id")["Temperature (°C)"].mean().sort_values()
        kept_holds = hold_temps.index[::hold_stride]
        df = df[df["hold_id"].isin(kept_holds)]

    n_holds = df["hold_id"].nunique()
    if n_holds < 2:
        print(f"  [{material}/{sample}] only {n_holds} temperature hold(s) -- skipping TTS")
        return

    shift_df, master_df, ref_temp_actual = build_master_curve(df, ref_temp)
    wlf_params = fit_wlf(shift_df, ref_temp_actual, wlf_min_temp, wlf_max_temp)

    suffix = ""
    if temp_min is not None or temp_max is not None:
        lo = f"{temp_min:.0f}" if temp_min is not None else "min"
        hi = f"{temp_max:.0f}" if temp_max is not None else "max"
        suffix += f"__T{lo}-{hi}"
    if hold_stride > 1:
        suffix += f"__stride{hold_stride}"

    out_dir = csv_path.parent
    shift_path = out_dir / f"{sample}__shift_factors{suffix}.csv"
    master_path = out_dir / f"{sample}__master_curve{suffix}.csv"
    plot_path = out_dir / f"{sample}__tts_master_curve{suffix}.png"

    shift_df.to_csv(shift_path, index=False)
    master_df.to_csv(master_path, index=False)
    plot_tts(sample, material, master_df, shift_df, ref_temp_actual, wlf_params, plot_path)

    wlf_msg = f"C1={wlf_params[0]:.3g}, C2={wlf_params[1]:.3g}" if wlf_params is not None else "fit failed"
    print(f"  [{material}/{sample}] Tref={ref_temp_actual:.1f}°C, {n_holds} holds, WLF: {wlf_msg}")
    flagged = shift_df.loc[shift_df["isolated_low_confidence"]]
    if not flagged.empty:
        for _, row in flagged.iterrows():
            print(f"    WARNING: low-confidence shift at {row['temperature_C']:.1f}°C "
                  f"({row['low_confidence_reason']}) -- excluded from the WLF fit.")
        n_flagged, n_total = len(flagged), len(shift_df)
        if n_flagged > n_total * 0.3:
            print(f"    NOTE: {n_flagged}/{n_total} holds flagged low-confidence -- the sequential "
                  f"shift-fitting has broken down across most of this curve. This can be caused by "
                  f"coarse/sparse temperature sampling, but a dense sample can fail the same way if "
                  f"the material has an unusually sharp/steep relaxation (high per-hold curvature is "
                  f"itself hard for this algorithm to shift reliably) -- check the raw temperature-"
                  f"domain plot before assuming denser sampling alone will fix it.")
    print(f"    -> {shift_path.name}, {master_path.name}, {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=None,
                         help="DMA root directory (default: parent directory of this script)")
    parser.add_argument("--ref-temp", type=float, default=25.0,
                         help="Reference temperature in °C (closest available hold is used; default 25)")
    parser.add_argument("--wlf-min-temp", type=float, default=None,
                         help="Exclude holds below this temperature (°C) from the WLF fit only "
                              "(e.g. to drop a sub-Tg secondary relaxation region); master curve is unaffected")
    parser.add_argument("--wlf-max-temp", type=float, default=None,
                         help="Exclude holds above this temperature (°C) from the WLF fit only; "
                              "master curve is unaffected")
    parser.add_argument("--temp-min", type=float, default=None,
                         help="Simulate a narrower physical sweep: drop holds below this temperature (°C) "
                              "entirely, before shifting -- affects the master curve too, not just the WLF fit")
    parser.add_argument("--temp-max", type=float, default=None,
                         help="Simulate a narrower physical sweep: drop holds above this temperature (°C) "
                              "entirely, before shifting -- affects the master curve too, not just the WLF fit")
    parser.add_argument("--hold-stride", type=int, default=1,
                         help="Simulate a coarser temperature step by keeping only every Nth hold "
                              "(e.g. 2 turns 5°C steps into effective 10°C steps)")
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
                            args.wlf_min_temp, args.wlf_max_temp, args.temp_min, args.temp_max,
                            args.hold_stride)


if __name__ == "__main__":
    main()
