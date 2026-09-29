"""Stress-vs-time plot for the A25V75 relaxation tests (all held at ~1 mm),
one subplot per loading rate. Tests 1-3 and 9 ramp at the nominal baseline
crosshead speed (~0.167 mm/s); tests 4-6 ramp at 10x that rate
(~1.667 mm/s); tests 10-12 ramp at ~10x slower than baseline
(~0.017 mm/s) -- plotted as a bold mean curve per rate group with
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

sys.path.insert(0, str(Path(__file__).parent.parent))
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
    3: "Slow",
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


def main():
    fig, axes = plt.subplots(
        1, 3, figsize=figsize_double, sharey=True
    )

    dfs_by_test = {test_id: load(test_id) for test_id in TESTS}

    for ax, label in zip(axes, RATE_LABELS):
        color = RATE_COLOR[label]
        for test_id, lbl in TESTS.items():
            if lbl != label:
                continue
            df = dfs_by_test[test_id]
            ax.plot(df["time_s"], df["stress"], color=color, alpha=INDIVIDUAL_ALPHA, linewidth=0.8,
                    label=f"test {test_id}")

        dfs = [dfs_by_test[t] for t, lbl in TESTS.items() if lbl == label]
        t, stress = mean_stress(dfs)
        ax.plot(t, stress, color=color, linewidth=1.8, linestyle="--", label="mean")

        rate = mean_strain_rate(dfs)
        ax.set_xlabel("Time in s")
        ax.set_title(rf"{label} rate ($\dot\varepsilon\approx${100 * rate:.3g}\%/s)", fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Engineering stress in MPa")

    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
