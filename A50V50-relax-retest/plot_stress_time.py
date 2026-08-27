"""Stress-vs-time, stress-vs-strain, stress-vs-position, and position-vs-strain
plots for the A50V50 configuration retest: 3 specimens at 1 mm/low rate,
3 at 2 mm/low rate, and 3 at 2 mm/high rate (10x). The original 6 mm/low
rate specimens (Config 3, tests 5-6) and the original altered-rate
specimens (test 7, 10x rate; test 8, 5x rate; both 6 mm) from the first
A50V50-relax study are included for comparison.

Stress is computed from force and the specimen cross-section (6 mm x 2 mm),
not the instrument's own Ch:Stress channel.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotstyle import colors, figsize_single, linestyles  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"
ORIGINAL_CLEANED_DIR = Path(__file__).parent.parent / "A50V50-relax" / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

# test_id -> (group label, replicate index within group, cleaned-data directory)
TESTS = {
    1: ("1 mm, low rate", 0, CLEANED_DIR),
    2: ("1 mm, low rate", 1, CLEANED_DIR),
    3: ("1 mm, low rate", 2, CLEANED_DIR),
    4: ("2 mm, low rate", 0, CLEANED_DIR),
    5: ("2 mm, low rate", 1, CLEANED_DIR),
    6: ("2 mm, low rate", 2, CLEANED_DIR),
    7: ("2 mm, high rate", 0, CLEANED_DIR),
    11: ("2 mm, high rate", 1, CLEANED_DIR),
    12: ("2 mm, high rate", 2, CLEANED_DIR),
    "orig-5": ("6 mm, low rate (original)", 0, ORIGINAL_CLEANED_DIR),
    "orig-6": ("6 mm, low rate (original)", 1, ORIGINAL_CLEANED_DIR),
    "orig-7": ("6 mm, 10x rate (original)", 0, ORIGINAL_CLEANED_DIR),
    "orig-8": ("6 mm, 5x rate (original)", 0, ORIGINAL_CLEANED_DIR),
}

GROUP_LABELS = list(dict.fromkeys(label for label, _, _ in TESTS.values()))
GROUP_COLOR = {label: colors[i] for i, label in enumerate(GROUP_LABELS)}


def main():
    fig, ((ax_time, ax_strain), (ax_position, ax_position_strain)) = plt.subplots(
        2, 2, figsize=(2 * figsize_single[0], 2 * figsize_single[1])
    )

    for test_id, (label, rep, cleaned_dir) in TESTS.items():
        file_id = test_id.split("-")[1] if isinstance(test_id, str) else test_id
        df = pd.read_csv(cleaned_dir / f"A50V50-{file_id}_cleaned.csv")
        stress_mpa = -df["force_N"] / AREA_MM2
        style = dict(
            color=GROUP_COLOR[label],
            linestyle=linestyles[rep % len(linestyles)],
            label=f"{label} (test {file_id})",
        )
        ax_time.plot(df["time_s"], stress_mpa, **style)
        ax_strain.plot(df["strain"], stress_mpa, **style)
        ax_position.plot(df["position_mm"], stress_mpa, **style)
        ax_position_strain.plot(df["strain"], df["position_mm"], **style)

    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Stress (MPa)")

    ax_strain.set_xlabel("Strain (in/in)")
    ax_strain.set_ylabel("Stress (MPa)")

    ax_position.set_xlabel("Position (mm)")
    ax_position.set_ylabel("Stress (MPa)")

    ax_position_strain.set_xlabel("Strain (in/in)")
    ax_position_strain.set_ylabel("Position (mm)")

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
