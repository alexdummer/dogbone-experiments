"""Stress-vs-strain plot for the A0V100 relaxation tests, one subplot per
loading rate, restricted to the loading ramp (the hold/relaxation phase
that follows is excluded, since strain there is dominated by viscoelastic
drift rather than the imposed deformation).

See plot_stress_time.py for the rate-group/replicate structure and the
test-2 outlier note.
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
    10: ("Very slow rate (0.017 mm/s)", True),
    11: ("Very slow rate (0.017 mm/s)", True),
    12: ("Very slow rate (0.017 mm/s)", True),
    1: ("Slow rate (0.17 mm/s)", True),
    2: ("Slow rate (0.17 mm/s)", False),  # outlier, retested as 13; kept only for transparency
    3: ("Slow rate (0.17 mm/s)", True),
    13: ("Slow rate (0.17 mm/s)", True),
    4: ("Fast rate (1.62 mm/s)", True),
    5: ("Fast rate (1.62 mm/s)", True),
    6: ("Fast rate (1.62 mm/s)", True),
}

RATE_LABELS = list(dict.fromkeys(label for label, _ in TESTS.values()))
RATE_COLOR = {label: colors[i] for i, label in enumerate(RATE_LABELS)}


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A0V100-{test_id}_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    return df


def ramp_bounds(position):
    """Onset and end index of the loading ramp, from the crosshead position
    trace: onset is where position first sustains a real deviation from
    its baseline; the ramp ends once position first reaches (within 1%)
    the hold value it settles at for the final 10% of the record."""
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


def ramp_strain_stress(df):
    onset, ramp_end = ramp_bounds(df["position_mm"])
    sl = slice(onset, ramp_end + 1)
    return df["strain"].to_numpy()[sl], df["stress"].to_numpy()[sl]


def mean_stress_vs_strain(ramp_curves):
    max_strain = min(strain.max() for strain, _ in ramp_curves)
    common_strain = np.linspace(0, max_strain, 200)
    stresses = np.array([np.interp(common_strain, strain, stress) for strain, stress in ramp_curves])
    return common_strain, stresses.mean(axis=0)


def main():
    fig, axes = plt.subplots(
        1, 3, figsize=(figsize_double[0] * 1.6, figsize_double[1]), sharey=True
    )

    ramp_by_test = {test_id: ramp_strain_stress(load(test_id)) for test_id in TESTS}

    for ax, label in zip(axes, RATE_LABELS):
        color = RATE_COLOR[label]
        for test_id, (lbl, in_mean) in TESTS.items():
            if lbl != label:
                continue
            strain, stress = ramp_by_test[test_id]
            ax.plot(
                strain, stress, color=color, alpha=INDIVIDUAL_ALPHA,
                linewidth=0.8, linestyle="-" if in_mean else ":",
                label=f"test {test_id} (excluded)" if not in_mean else None,
            )

        ramp_curves = [ramp_by_test[t] for t, (lbl, in_mean) in TESTS.items() if lbl == label and in_mean]
        strain, stress = mean_stress_vs_strain(ramp_curves)
        ax.plot(strain, stress, color=color, linewidth=1.8, label="mean")

        ax.set_xlabel("Strain (in/in)")
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Stress (MPa)")

    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_strain.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
