"""Visualize the three A0V100 cyclic tests (7, 8, 9): the raw stress-time
trace and the cycle-by-cycle peak/valley stress (cyclic softening trend).

Specimen 9's cyclic test required two retests before a clean run was
captured (see clean_data.py); treat it with a bit more caution than 7/8.

Stress is computed from force and the specimen cross-section (6 mm x 2 mm).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotstyle import colors, figsize_single  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

TESTS = {7: colors[0], 8: colors[1], 9: colors[2]}
CAUTION_TESTS = {9}
PROMINENCE_MM = 0.02


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A0V100-{test_id}_cyclic_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


def cycle_extrema(df):
    peaks, _ = find_peaks(df["position_mm"], prominence=PROMINENCE_MM)
    troughs, _ = find_peaks(-df["position_mm"], prominence=PROMINENCE_MM)
    return peaks, troughs


def main():
    fig, (ax_time, ax_degrad) = plt.subplots(
        1, 2, figsize=(2 * figsize_single[0], figsize_single[1])
    )

    for test_id, color in TESTS.items():
        df = load(test_id)
        label = f"test {test_id}" + (" (caution)" if test_id in CAUTION_TESTS else "")
        ax_time.plot(df["time_s"], df["stress"], color=color, label=label)

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
    ax_time.legend(fontsize=7)
    ax_time.grid(True, alpha=0.4)

    ax_degrad.set_xlabel("Cycle number")
    ax_degrad.set_ylabel("Stress (MPa)")
    ax_degrad.legend(fontsize=6)
    ax_degrad.grid(True, alpha=0.4)

    fig.tight_layout()

    out_path = Path(__file__).parent / "cyclic_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
