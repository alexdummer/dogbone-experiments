"""Stress-vs-time and stress-vs-position comparison of Config 3 (tests 5, 6)
against the two later single-shot retests of samples 1 and 2, to check
whether Config 3's reduced stiffness is reproduced by specimens with a
prior load history."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotstyle import colors, figsize_single  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

SERIES = [
    (5, "Config 3 - test 5 (orig, 6 mm)", colors[2], "-"),
    (6, "Config 3 - test 6 (orig, 6 mm)", colors[2], "--"),
    ("1_retest", "Sample 1 retest (6 mm)", colors[5], ":"),
    ("2_retest", "Sample 2 retest (2 mm)", colors[6], ":"),
]


def main():
    fig, (ax_time, ax_position) = plt.subplots(
        1, 2, figsize=(2 * figsize_single[0], figsize_single[1])
    )

    for test_id, label, color, ls in SERIES:
        df = pd.read_csv(CLEANED_DIR / f"A50V50-{test_id}_cleaned.csv")
        stress_mpa = -df["force_N"] / AREA_MM2
        ax_time.plot(df["time_s"], stress_mpa, color=color, linestyle=ls, label=label)
        ax_position.plot(df["position_mm"], stress_mpa, color=color, linestyle=ls, label=label)

    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Stress (MPa)")

    ax_position.set_xlabel("Position (mm)")
    ax_position.set_ylabel("Stress (MPa)")

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()

    out_path = Path(__file__).parent / "config3_vs_retest.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
