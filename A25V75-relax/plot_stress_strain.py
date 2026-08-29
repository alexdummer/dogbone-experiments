"""Stress-vs-strain plot for the A25V75 relaxation tests, all loading rates
overlaid on a single axis, restricted to the loading ramp (the
hold/relaxation phase that follows is excluded, since strain there is
dominated by viscoelastic drift rather than the imposed deformation).

See plot_stress_time.py for the rate-group/replicate structure.
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

# test_id -> rate label
TESTS = {
    10: "Very slow rate (0.017 mm/s)",
    11: "Very slow rate (0.017 mm/s)",
    12: "Very slow rate (0.017 mm/s)",
    1: "Slow rate (0.17 mm/s)",
    2: "Slow rate (0.17 mm/s)",
    3: "Slow rate (0.17 mm/s)",
    9: "Slow rate (0.17 mm/s)",
    4: "Fast rate (1.67 mm/s)",
    5: "Fast rate (1.67 mm/s)",
    6: "Fast rate (1.67 mm/s)",
}

RATE_LABELS = list(dict.fromkeys(TESTS.values()))
RATE_COLOR = {label: colors[i] for i, label in enumerate(RATE_LABELS)}


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A25V75-{test_id}_cleaned.csv")
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
    fig, ax = plt.subplots(figsize=figsize_double)

    ramp_by_test = {test_id: ramp_strain_stress(load(test_id)) for test_id in TESTS}

    for label in RATE_LABELS:
        color = RATE_COLOR[label]
        for test_id, lbl in TESTS.items():
            if lbl != label:
                continue
            strain, stress = ramp_by_test[test_id]
            ax.plot(strain, stress, color=color, alpha=INDIVIDUAL_ALPHA, linewidth=0.8)

        ramp_curves = [ramp_by_test[t] for t, lbl in TESTS.items() if lbl == label]
        strain, stress = mean_stress_vs_strain(ramp_curves)
        ax.plot(strain, stress, color=color, linewidth=1.8, label=label)

    ax.set_xlabel("Strain (in/in)")
    ax.set_ylabel("Stress (MPa)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_strain.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
