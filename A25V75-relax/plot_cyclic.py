"""Visualize the two A25V75 cyclic tests (7, 8): the raw stress-time trace,
the cycle-by-cycle peak/valley stress (cyclic softening trend), and the
stress-position hysteresis loops colored by cycle number.

Stress is computed from force and the specimen cross-section (6 mm x 2 mm).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotstyle import colors, figsize_single  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

TESTS = {7: colors[0], 8: colors[1]}
PROMINENCE_MM = 0.02


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A25V75-{test_id}_cyclic_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


def cycle_extrema(df):
    peaks, _ = find_peaks(df["position_mm"], prominence=PROMINENCE_MM)
    troughs, _ = find_peaks(-df["position_mm"], prominence=PROMINENCE_MM)
    return peaks, troughs


def plot_hysteresis(ax, df, troughs):
    n_cycles = len(troughs) - 1
    cmap = plt.get_cmap("viridis")
    for k in range(n_cycles):
        sl = slice(troughs[k], troughs[k + 1] + 1)
        points = np.array([df["position_mm"].iloc[sl], df["stress"].iloc[sl]]).T
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, color=cmap(k / max(n_cycles - 1, 1)), linewidth=0.6)
        ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Stress (MPa)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_cycles))
    plt.colorbar(sm, ax=ax, label="Cycle")


def main():
    fig, ((ax_time, ax_degrad), (ax_hyst7, ax_hyst8)) = plt.subplots(
        2, 2, figsize=(2 * figsize_single[0], 2 * figsize_single[1])
    )

    for test_id, color in TESTS.items():
        df = load(test_id)
        ax_time.plot(df["time_s"], df["stress"], color=color, label=f"test {test_id}")

        peaks, troughs = cycle_extrema(df)
        ax_degrad.plot(
            np.arange(len(peaks)), df["stress"].iloc[peaks],
            color=color, marker="o", markersize=2, linestyle="-", label=f"test {test_id} peak",
        )
        ax_degrad.plot(
            np.arange(len(troughs)), df["stress"].iloc[troughs],
            color=color, marker="o", markersize=2, linestyle="--", label=f"test {test_id} valley",
        )

    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Stress (MPa)")
    ax_time.legend()

    ax_degrad.set_xlabel("Cycle number")
    ax_degrad.set_ylabel("Stress (MPa)")
    ax_degrad.legend(fontsize=7)

    df7 = load(7)
    _, troughs7 = cycle_extrema(df7)
    plot_hysteresis(ax_hyst7, df7, troughs7)
    ax_hyst7.set_title("Test 7", fontsize=9)

    df8 = load(8)
    _, troughs8 = cycle_extrema(df8)
    plot_hysteresis(ax_hyst8, df8, troughs8)
    ax_hyst8.set_title("Test 8", fontsize=9)

    fig.tight_layout()

    out_path = Path(__file__).parent / "cyclic_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
