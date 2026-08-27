"""Concatenate the raw chunks of each A0V100 relaxation test (tests 1-6,
variable number of _N continuation chunks), extract time, position, force,
and strain, and zero position/force/strain against the pre-ramp baseline
(before position starts moving) rather than the raw first sample."""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
CLEANED_DIR = Path(__file__).parent / "cleaned"

COLUMNS = {
    "Time (sec)": "time_s",
    "Ch:Position (mm)": "position_mm",
    "Ch:Load (N)": "force_N",
    "Ch:Strain (in/in)": "strain",
}

ZERO_COLUMNS = ["position_mm", "force_N", "strain"]

# the actuator position is flat (effectively noise-free) until the ramp
# starts, so the ramp onset is the first point that deviates from the
# initial position by more than half its quantization step, sustained for
# SUSTAIN samples in a row
POSITION_STEP_MM = 0.0004
SUSTAIN = 5

RELAX_PATTERN = re.compile(r"^(A0V100-(\d+)_\d{8}_\d{6})\.csv$")


def find_relax_tests():
    tests = {}
    for path in RAW_DIR.glob("A0V100-*.csv"):
        m = RELAX_PATTERN.match(path.name)
        if m:
            tests[int(m.group(2))] = (path, m.group(1))
    return dict(sorted(tests.items()))


def load_relax_test(main_path, stem):
    chunks = [main_path]
    n = 1
    while (chunk := RAW_DIR / f"{stem}_{n}.csv").exists():
        chunks.append(chunk)
        n += 1
    df = pd.concat(
        [pd.read_csv(c, usecols=COLUMNS.keys()) for c in chunks],
        ignore_index=True,
    )
    return df.rename(columns=COLUMNS)


def find_ramp_onset(position):
    baseline = position.iloc[0]
    deviation = (position - baseline).abs()
    sustained = deviation.rolling(SUSTAIN).min() > POSITION_STEP_MM / 2
    onset_positions = sustained.to_numpy().nonzero()[0]
    onset = onset_positions[0] - SUSTAIN + 1
    return onset


def zero_to_baseline(df):
    onset = find_ramp_onset(df["position_mm"])
    baseline = df[ZERO_COLUMNS].iloc[:onset].mean()
    df[ZERO_COLUMNS] = df[ZERO_COLUMNS] - baseline
    return df, onset


def main():
    CLEANED_DIR.mkdir(exist_ok=True)
    for test_id, (main_path, stem) in find_relax_tests().items():
        df = load_relax_test(main_path, stem)
        df, onset = zero_to_baseline(df)
        out_path = CLEANED_DIR / f"A0V100-{test_id}_cleaned.csv"
        df.to_csv(out_path, index=False)
        print(
            f"test {test_id}: {len(df)} rows, zeroed on first "
            f"{onset} samples (t < {df['time_s'].iloc[onset]:.3f} s) -> {out_path}"
        )


if __name__ == "__main__":
    main()
