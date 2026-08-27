"""Stress-vs-time, stress-vs-strain, stress-vs-position, and position-vs-strain
plots for all 8 A50V50 relaxation tests, plus the two later single-shot
retests of samples 1 and 2 pushed to new target positions.

Stress is computed from force and the specimen cross-section
(6 mm x 2 mm, shared by all specimens), not the instrument's own
Ch:Stress channel. Tests 1-2, 3-4, and 5-6 are replicate pairs of
three configurations; tests 7 and 8 each used their own altered
strain rate and are plotted as separate configurations. The retests
have no valid strain channel (extensometer was disconnected), so they
are omitted from the strain-based panels.
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

# test_id -> (config label, replicate index within config)
CONFIGS = {
    1: ("Config 1", 0),
    2: ("Config 1", 1),
    3: ("Config 2", 0),
    4: ("Config 2", 1),
    5: ("Config 3", 0),
    6: ("Config 3", 1),
    7: ("Config 4 (altered rate)", 0),
    8: ("Config 5 (altered rate)", 0),
}

CONFIG_LABELS = list(dict.fromkeys(label for label, _ in CONFIGS.values()))
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


def main():
    fig, ((ax_time, ax_strain), (ax_position, ax_position_strain)) = plt.subplots(
        2, 2, figsize=(2 * figsize_single[0], 2 * figsize_single[1])
    )

    for test_id, (label, rep) in CONFIGS.items():
        df = pd.read_csv(CLEANED_DIR / f"A50V50-{test_id}_cleaned.csv")
        stress_mpa = -df["force_N"] / AREA_MM2
        style = dict(
            color=CONFIG_COLOR[label],
            linestyle=linestyles[rep % len(linestyles)],
            label=f"{label} (test {test_id})",
        )
        ax_time.plot(df["time_s"], stress_mpa, **style)
        ax_strain.plot(df["strain"], stress_mpa, **style)
        ax_position.plot(df["position_mm"], stress_mpa, **style)
        ax_position_strain.plot(df["strain"], df["position_mm"], **style)

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
