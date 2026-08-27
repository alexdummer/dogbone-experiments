"""Stress-vs-time, stress-vs-strain, stress-vs-position, and position-vs-strain
plots for the 6 A25V75 relaxation tests (all held at ~1 mm). Tests 1-3 ramp
at the baseline rate (~0.167 mm/s); tests 4-6 ramp at 10x that rate
(~1.667 mm/s) -- three replicates per rate.

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

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

# test_id -> (rate label, replicate index within rate group)
TESTS = {
    1: ("Slow rate (0.17 mm/s)", 0),
    2: ("Slow rate (0.17 mm/s)", 1),
    3: ("Slow rate (0.17 mm/s)", 2),
    4: ("Fast rate (1.67 mm/s)", 0),
    5: ("Fast rate (1.67 mm/s)", 1),
    6: ("Fast rate (1.67 mm/s)", 2),
}

RATE_LABELS = list(dict.fromkeys(label for label, _ in TESTS.values()))
RATE_COLOR = {label: colors[i] for i, label in enumerate(RATE_LABELS)}


def main():
    fig, ((ax_time, ax_strain), (ax_position, ax_position_strain)) = plt.subplots(
        2, 2, figsize=(2 * figsize_single[0], 2 * figsize_single[1])
    )

    for test_id, (label, rep) in TESTS.items():
        df = pd.read_csv(CLEANED_DIR / f"A25V75-{test_id}_cleaned.csv")
        stress_mpa = -df["force_N"] / AREA_MM2
        style = dict(
            color=RATE_COLOR[label],
            linestyle=linestyles[rep % len(linestyles)],
            label=f"{label} (test {test_id})",
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
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
