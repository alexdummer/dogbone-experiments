"""Stress-vs-time plot for the A50V50 relaxation tests, one subplot per
loading rate, restricted to the 2 mm-hold retest series: tests 13-15 ramp
at ~10x slower than baseline (~0.017 mm/s), tests 4-6 ramp at the baseline
rate (~0.167 mm/s), and tests 7/11/12 ramp at ~10x that rate (~1.67 mm/s)
-- the same three rate groups used for A50V50 elsewhere in the
cross-material comparison (compare_materials.py, standup_slides.py).
Plotted as a bold mean curve per rate group with individual specimens
shown faded (alpha=0.3) behind it.

Test 16 was meant to be a 4th "slow rate" replicate, but is consistently
and substantially stiffer than tests 4/5/6 from the very start of the ramp
-- a real specimen-level difference, not a late-stage artifact (see the
original plot_stress_time.py docstring for details). It is shown here as a
faded, dotted individual trace for transparency but excluded from the mean
curve.

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

# test_id -> (rate label, included in the group's mean curve)
TESTS = {
    13: ("Very slow rate (0.017 mm/s)", True),
    14: ("Very slow rate (0.017 mm/s)", True),
    15: ("Very slow rate (0.017 mm/s)", True),
    4: ("Slow rate (0.17 mm/s)", True),
    5: ("Slow rate (0.17 mm/s)", True),
    6: ("Slow rate (0.17 mm/s)", True),
    16: ("Slow rate (0.17 mm/s)", False),  # outlier, substantially stiffer; recommend retest
    7: ("Fast rate (1.67 mm/s)", True),
    11: ("Fast rate (1.67 mm/s)", True),
    12: ("Fast rate (1.67 mm/s)", True),
}

RATE_LABELS = list(dict.fromkeys(label for label, _ in TESTS.values()))
RATE_COLOR = {label: colors[i] for i, label in enumerate(RATE_LABELS)}


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A50V50-{test_id}_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


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
        for test_id, (lbl, in_mean) in TESTS.items():
            if lbl != label:
                continue
            df = dfs_by_test[test_id]
            ax.plot(
                df["time_s"], df["stress"], color=color, alpha=INDIVIDUAL_ALPHA,
                linewidth=0.8, linestyle="-" if in_mean else ":",
                label=f"test {test_id} (excluded)" if not in_mean else None,
            )

        dfs = [dfs_by_test[t] for t, (lbl, in_mean) in TESTS.items() if lbl == label and in_mean]
        t, stress = mean_stress(dfs)
        ax.plot(t, stress, color=color, linewidth=1.8, label="mean")

        ax.set_xlabel("Time (s)")
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Engineering stress (MPa)")

    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
