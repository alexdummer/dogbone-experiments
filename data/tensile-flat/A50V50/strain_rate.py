"""Estimate the approximate loading strain rate for each relaxation test.

The ramp is actuator-position-controlled at a constant rate, so the ramp
window is bounded using the (low-noise) position channel: from ramp onset
(reused from clean_data.py's onset detection) to the first time position
reaches its steady-state hold value. The strain rate is then the slope of
a linear fit of strain vs. time over that window.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clean_data import find_ramp_onset

CLEANED_DIR = Path(__file__).parent / "cleaned"

HOLD_FRACTION = 0.1  # trailing fraction of the test used to estimate the hold position
REACH_FRACTION = 0.99  # fraction of the hold position that marks ramp end

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


def find_ramp_end(position, onset):
    n = len(position)
    hold_value = position.iloc[-int(n * HOLD_FRACTION):].median()
    reached = (position >= REACH_FRACTION * hold_value).to_numpy().nonzero()[0]
    reached = reached[reached > onset]
    return reached[0], hold_value


def main():
    rows = []
    for test_id, label in CONFIGS.items():
        df = pd.read_csv(CLEANED_DIR / f"A50V50-{test_id}_cleaned.csv")
        onset = find_ramp_onset(df["position_mm"])
        ramp_end, hold_position = find_ramp_end(df["position_mm"], onset)

        t = df["time_s"].iloc[onset:ramp_end].to_numpy()
        eps = df["strain"].iloc[onset:ramp_end].to_numpy()
        pos = df["position_mm"].iloc[onset:ramp_end].to_numpy()
        strain_rate, _ = np.polyfit(t, eps, 1)
        position_rate, _ = np.polyfit(t, pos, 1)

        rows.append(
            {
                "test": test_id,
                "config": label,
                "ramp_start_s": df["time_s"].iloc[onset],
                "ramp_end_s": df["time_s"].iloc[ramp_end],
                "hold_position_mm": hold_position,
                "position_rate_mm_per_s": position_rate,
                "strain_rate_1_per_s": strain_rate,
                "strain_rate_pct_per_s": strain_rate * 100,
            }
        )

    summary = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    print(summary.to_string(index=False))

    out_path = Path(__file__).parent / "strain_rate_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
