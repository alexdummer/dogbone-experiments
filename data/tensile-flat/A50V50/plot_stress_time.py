"""Stress-vs-time, stress-vs-strain, stress-vs-position, and position-vs-strain
plots for all 8 A50V50 relaxation tests, plus the two later single-shot
retests of samples 1 and 2 pushed to new target positions.

Stress is computed from force and the specimen cross-section
(6 mm x 2 mm, shared by all specimens), not the instrument's own
Ch:Stress channel. Tests 1-2, 3-4, and 5-6 are replicate pairs of three
configurations, plotted as a bold mean curve with individual specimens
shown faded (alpha=0.3) behind it; tests 7 and 8 each used their own
altered strain rate (n=1, so the "mean" curve is just that one test) and
are plotted as separate configurations. The retests have no valid strain
channel (extensometer was disconnected), so they are omitted from the
strain-based panels and shown as single dotted reference lines rather than
mean curves.

Since strain isn't perfectly monotonic during the hold (small viscoelastic
drift), mean curves are built by averaging stress/strain/position
independently on a common time grid per configuration, then pairing the
resulting mean channels -- not by averaging one channel against another
directly.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotstyle import colors, figsize_single  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM
INDIVIDUAL_ALPHA = 0.3

# test_id -> config label
CONFIGS = {
    1: "Config 1",
    2: "Config 1",
    3: "Config 2",
    4: "Config 2",
    5: "Config 3",
    6: "Config 3",
    7: "Config 4 (altered rate)",
    8: "Config 5 (altered rate)",
}

CONFIG_LABELS = list(dict.fromkeys(CONFIGS.values()))
CONFIG_COLOR = {label: colors[i] for i, label in enumerate(CONFIG_LABELS)}

# retested sample_id -> (label, hold position for reference)
RETESTS = {
    1: "Sample 1 retest (6 mm)",
    2: "Sample 2 retest (2 mm)",
}
RETEST_COLOR = {
    1: colors[len(CONFIG_LABELS)],
    2: colors[len(CONFIG_LABELS) + 1],
}


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


def mean_channels(dfs):
    common_time = np.linspace(0, max(df["time_s"].max() for df in dfs), 5000)
    stresses = np.array([interp_or_nan(common_time, df["time_s"], df["stress"]) for df in dfs])
    strains = np.array([interp_or_nan(common_time, df["time_s"], df["strain"]) for df in dfs])
    positions = np.array([interp_or_nan(common_time, df["time_s"], df["position_mm"]) for df in dfs])
    return common_time, np.nanmean(stresses, axis=0), np.nanmean(strains, axis=0), np.nanmean(positions, axis=0)


def main():
    fig, ((ax_time, ax_strain), (ax_position, ax_position_strain)) = plt.subplots(
        2, 2, figsize=(2 * figsize_single[0], 2 * figsize_single[1])
    )

    dfs_by_test = {test_id: load(test_id) for test_id in CONFIGS}

    for test_id, label in CONFIGS.items():
        df = dfs_by_test[test_id]
        style = dict(color=CONFIG_COLOR[label], alpha=INDIVIDUAL_ALPHA, linewidth=0.8)
        ax_time.plot(df["time_s"], df["stress"], **style)
        ax_strain.plot(df["strain"], df["stress"], **style)
        ax_position.plot(df["position_mm"], df["stress"], **style)
        ax_position_strain.plot(df["strain"], df["position_mm"], **style)

    for label in CONFIG_LABELS:
        dfs = [dfs_by_test[t] for t, lbl in CONFIGS.items() if lbl == label]
        t, stress, strain, position = mean_channels(dfs)
        style = dict(color=CONFIG_COLOR[label], linewidth=1.8, label=label)
        ax_time.plot(t, stress, **style)
        ax_strain.plot(strain, stress, **style)
        ax_position.plot(position, stress, **style)
        ax_position_strain.plot(strain, position, **style)

    for sample_id, label in RETESTS.items():
        df = pd.read_csv(CLEANED_DIR / f"A50V50-{sample_id}_retest_cleaned.csv")
        stress_mpa = -df["force_N"] / AREA_MM2
        style = dict(color=RETEST_COLOR[sample_id], linestyle=":", label=label)
        ax_time.plot(df["time_s"], stress_mpa, **style)
        ax_position.plot(df["position_mm"], stress_mpa, **style)

    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Stress (MPa)")

    ax_strain.set_xlabel("Strain (in/in)")
    ax_strain.set_ylabel("Stress (MPa)")
    ax_strain.set_xlim(left=0.0, right=0.025)

    ax_position.set_xlabel("Position (mm)")
    ax_position.set_ylabel("Stress (MPa)")

    ax_position_strain.set_xlabel("Strain (in/in)")
    ax_position_strain.set_ylabel("Position (mm)")
    ax_position_strain.set_xlim(left=0.0, right=0.025)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
