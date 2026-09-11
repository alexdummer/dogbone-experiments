"""Stress-vs-strain plot for the A75V25 relaxation tests, all loading rates
overlaid on a single axis, restricted to the loading ramp (same convention
as every other material -- see plot_stress_time.py for the rate-group
structure and the partial-dataset/rate-labeling caveats). Strain here is
the two-point videoextensometer measurement (sync_and_clean_data.py),
valid only over the ramp -- exactly the region this plot uses.

Legend entries report the actual measured mean engineering strain rate
over the ramp (not the nominal crosshead displacement rate) -- see
plot_stress_time.py's module docstring for why.
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

# test_id -> (rate tier, included in the group's mean curve)
TESTS = {
    1: ("Slow", True),
    2: ("Slow", True),
    3: ("Slow", True),
    4: ("Fast", True),
    5: ("Fast", True),
    6: ("Fast", True),
    7: ("Very slow", True),
    8: ("Very slow", True),
    9: ("Very slow", True),
}

RATE_LABELS = list(dict.fromkeys(label for label, _ in TESTS.values()))
RATE_COLOR_ALL = ["Very slow", "Slow", "Fast"]
RATE_COLOR = {label: colors[i] for i, label in enumerate(RATE_COLOR_ALL)}


def load(test_id):
    df = pd.read_csv(CLEANED_DIR / f"A75V25-{test_id}_cleaned.csv")
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
    return 100 * df["strain"].to_numpy()[sl], df["stress"].to_numpy()[sl]


def strain_rate(df):
    """Actual measured mean engineering strain rate over the ramp
    (extensometer strain change / ramp duration) -- not the nominal
    crosshead displacement rate, see plot_stress_time.py's docstring."""
    onset, ramp_end = ramp_bounds(df["position_mm"])
    t, strain = df["time_s"].to_numpy(), df["strain"].to_numpy()
    return (strain[ramp_end] - strain[onset]) / (t[ramp_end] - t[onset])


def mean_stress_vs_strain(ramp_curves):
    max_strain = min(strain.max() for strain, _ in ramp_curves)
    common_strain = np.linspace(0, max_strain, 200)
    stresses = np.array([np.interp(common_strain, strain, stress) for strain, stress in ramp_curves])
    return common_strain, stresses.mean(axis=0)


def main():
    fig, ax = plt.subplots(figsize=figsize_single)

    dfs_by_test = {test_id: load(test_id) for test_id in TESTS}
    ramp_by_test = {test_id: ramp_strain_stress(df) for test_id, df in dfs_by_test.items()}

    for label in RATE_LABELS:
        color = RATE_COLOR[label]
        for test_id, (lbl, in_mean) in TESTS.items():
            if lbl != label:
                continue
            strain, stress = ramp_by_test[test_id]
            ax.plot(
                strain, stress, color=color, alpha=INDIVIDUAL_ALPHA,
                linewidth=0.8, linestyle="-" if in_mean else ":",
                label=f"test {test_id} (excluded)" if not in_mean else f"test {test_id}",
            )

        ramp_curves = [ramp_by_test[t] for t, (lbl, in_mean) in TESTS.items() if lbl == label and in_mean]
        strain, stress = mean_stress_vs_strain(ramp_curves)
        rate = np.mean([strain_rate(dfs_by_test[t]) for t, (lbl, in_mean) in TESTS.items()
                         if lbl == label and in_mean])
        ax.plot(strain, stress, color=color, linewidth=1.8, linestyle="--",
                label=rf"{label} rate ($\dot\varepsilon\approx${100 * rate:.3g}\%/s, n={len(ramp_curves)})")

    ax.set_xlabel("Engineering strain (\\%)")
    ax.set_ylabel("Engineering stress (MPa)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()

    out_path = Path(__file__).parent / "stress_strain.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
