"""Stress-vs-time plot for the A25V75 relaxation tests (all held at ~1 mm),
one subplot per loading rate. Tests 1, 2, and 9 ramp at the nominal baseline
crosshead speed (~0.167 mm/s; test 3 was stopped early and is excluded);
tests 4-6 ramp at 10x that rate (~1.667 mm/s); tests 10-12 ramp at ~10x slower
than baseline (~0.017 mm/s) -- plotted as a bold mean curve per rate group with
individual specimens shown faded (alpha=0.3) behind it. Panel titles
report the actual measured mean engineering strain rate over the ramp
(not the nominal crosshead speed above), the same convention used for
compare_materials.py's rate-sensitivity analysis and the BB/QLV-Marmot
model-validation grids.

Mean curves are built by averaging stress independently on a common time
grid per rate group (strain isn't perfectly monotonic during the hold, due
to small viscoelastic drift, so channels are averaged independently rather
than one against another directly).

Stress is computed from force and the specimen cross-section (6 mm x 2 mm),
not the instrument's own Ch:Stress channel.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plotstyle import colors, figsize_double  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM
INDIVIDUAL_ALPHA = 0.3

# test_id -> rate tier
TESTS = {
    10: "Very slow",
    11: "Very slow",
    12: "Very slow",
    1: "Slow",
    2: "Slow",
    9: "Slow",
    4: "Fast",
    5: "Fast",
    6: "Fast",
}

RATE_LABELS = list(dict.fromkeys(TESTS.values()))
RATE_COLOR = {label: colors[i] for i, label in enumerate(RATE_LABELS)}


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A25V75-{test_id}_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


def ramp_bounds(position):
    baseline = position.iloc[0]
    deviation = (position - baseline).abs()
    sustained = deviation.rolling(5).min() > 0.0002
    nonzero = sustained.to_numpy().nonzero()[0]
    onset = max(nonzero[0] - 4, 0) if len(nonzero) else 0
    n = len(position)
    hold_value = position.iloc[-max(1, n // 10):].median()
    reached = (position >= 0.99 * hold_value).to_numpy().nonzero()[0]
    candidates = reached[reached > onset]
    ramp_end = candidates[0] if len(candidates) else n - 1
    return onset, ramp_end


def mean_strain_rate(dfs):
    """Actual measured mean engineering strain rate over the ramp
    (extensometer strain change / ramp duration), averaged across a rate
    group's replicates -- not the nominal crosshead displacement rate."""
    rates = []
    for df in dfs:
        onset, ramp_end = ramp_bounds(df["position_mm"])
        t, strain = df["time_s"].to_numpy(), df["strain"].to_numpy()
        rates.append((strain[ramp_end] - strain[onset]) / (t[ramp_end] - t[onset]))
    return np.mean(rates)


def interp_or_nan(common_time, t, y):
    """Like np.interp, but NaN past this specimen's own duration instead of
    clamping to its last value -- so a shorter-duration specimen shrinks
    the mean's effective N near the tail rather than either truncating the
    whole mean early or silently biasing it with a flat extrapolation."""
    values = np.interp(common_time, t, y)
    values[common_time > t.max()] = np.nan
    return values


def mean_stress(dfs):
    common_time = np.linspace(0, max(df["time_s"].max() for df in dfs), 5000)
    stresses = np.array([interp_or_nan(common_time, df["time_s"], df["stress"]) for df in dfs])
    return common_time, np.nanmean(stresses, axis=0)


def ramp_strain_stress(df):
    onset, ramp_end = ramp_bounds(df["position_mm"])
    sl = slice(onset, ramp_end + 1)
    return 100 * df["strain"].to_numpy()[sl], df["stress"].to_numpy()[sl]


def mean_stress_vs_strain(ramp_curves):
    max_strain = min(strain.max() for strain, _ in ramp_curves)
    common_strain = np.linspace(0, max_strain, 200)
    stresses = np.array([np.interp(common_strain, strain, stress) for strain, stress in ramp_curves])
    return common_strain, stresses.mean(axis=0)


def main():
    fig, axes = plt.subplots(
        1, 4, figsize=figsize_double, sharey=True
    )

    dfs_by_test = {test_id: load(test_id) for test_id in TESTS}

    for ax, label in zip(axes[:3], RATE_LABELS):
        color = RATE_COLOR[label]
        first_rep = True
        for test_id, lbl in TESTS.items():
            if lbl != label:
                continue
            df = dfs_by_test[test_id]
            ax.plot(
                df["time_s"], df["stress"], color=color, alpha=INDIVIDUAL_ALPHA,
                linewidth=0.8, linestyle="-",
                label="replicate" if first_rep else None,
            )
            first_rep = False

        dfs = [dfs_by_test[t] for t, lbl in TESTS.items() if lbl == label]
        t, stress = mean_stress(dfs)
        ax.plot(t, stress, color=color, linewidth=1.8, linestyle="--", label="mean")

        rate = mean_strain_rate(dfs)
        ax.set_xlabel("Time in s")
        ax.set_title(rf"{label} ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)", fontsize=8.5)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7.5, loc="lower right")

    # Panel 4: loading ramp stress vs strain
    ax_ss = axes[3]
    ramp_by_test = {test_id: ramp_strain_stress(df) for test_id, df in dfs_by_test.items()}
    for label in RATE_LABELS:
        color = RATE_COLOR[label]
        for test_id, lbl in TESTS.items():
            if lbl != label:
                continue
            strain, stress = ramp_by_test[test_id]
            ax_ss.plot(strain, stress, color=color, alpha=INDIVIDUAL_ALPHA, linewidth=0.8, linestyle="-")

        ramp_curves = [ramp_by_test[t] for t, lbl in TESTS.items() if lbl == label]
        strain, stress = mean_stress_vs_strain(ramp_curves)
        ax_ss.plot(strain, stress, color=color, linewidth=1.8, linestyle="--", label=label)

    ax_ss.set_xlabel(r"Strain in \%")
    ax_ss.set_title("Ramp stress--strain", fontsize=8.5)
    ax_ss.grid(True, alpha=0.4)
    ax_ss.legend(fontsize=7.5, loc="lower right")

    axes[0].set_ylabel("Engineering stress in MPa")

    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
