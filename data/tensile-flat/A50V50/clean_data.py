"""Concatenate the raw (main + _1 + _2) chunks of each relaxation test,
extract time, position, force, and strain, and zero position/force/strain
against the pre-ramp baseline (before position starts moving) rather than
the raw first sample."""

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
RETEST_DIR = RAW_DIR / "AsylumTests"
CLEANED_DIR = Path(__file__).parent / "cleaned"

COLUMNS = {
    "Time (sec)": "time_s",
    "Ch:Position (mm)": "position_mm",
    "Ch:Load (N)": "force_N",
    "Ch:Strain (in/in)": "strain",
}

# sample-to-file mapping for the later single-shot retests: the extensometer
# was not connected for these runs, so strain is unavailable (the raw strain
# columns contain garbage placeholder values); confirmed by actual hold
# position, not by the (misleading) filenames' test-order numbering
RETEST_FILES = {
    1: "A50_50_20Test_1_08242026_172439.csv",  # ramps to ~6 mm
    2: "A50_50_5_test2_08242026_173021.csv",  # ramps to ~2 mm
}

ZERO_COLUMNS = ["position_mm", "force_N", "strain"]

# the actuator position is flat (effectively noise-free) until the ramp
# starts, so the ramp onset is the first point that deviates from the
# initial position by more than half its quantization step, sustained for
# SUSTAIN samples in a row
POSITION_STEP_MM = 0.0004
SUSTAIN = 5

TEST_PATTERN = re.compile(r"^(A50V50-(\d+)_\d{8}_\d{6})\.csv$")


def find_tests():
    tests = {}
    for path in RAW_DIR.glob("A50V50-*.csv"):
        m = TEST_PATTERN.match(path.name)
        if m:
            tests[int(m.group(2))] = (path, m.group(1))
    return dict(sorted(tests.items()))


def load_test(main_path, stem):
    chunks = [main_path, RAW_DIR / f"{stem}_1.csv", RAW_DIR / f"{stem}_2.csv"]
    df = pd.concat(
        [pd.read_csv(c, usecols=COLUMNS.keys()) for c in chunks],
        ignore_index=True,
    )
    return df.rename(columns=COLUMNS)


def load_retest(path):
    position_columns = {k: v for k, v in COLUMNS.items() if v != "strain"}
    df = pd.read_csv(path, usecols=position_columns.keys())
    df = df.rename(columns=position_columns)
    df["strain"] = float("nan")
    return df


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
    for test_id, (main_path, stem) in find_tests().items():
        df = load_test(main_path, stem)
        df, onset = zero_to_baseline(df)
        out_path = CLEANED_DIR / f"A50V50-{test_id}_cleaned.csv"
        df.to_csv(out_path, index=False)
        print(
            f"test {test_id}: {len(df)} rows, zeroed on first "
            f"{onset} samples (t < {df['time_s'].iloc[onset]:.3f} s) -> {out_path}"
        )

    for test_id, filename in RETEST_FILES.items():
        df = load_retest(RETEST_DIR / filename)
        df, onset = zero_to_baseline(df)
        out_path = CLEANED_DIR / f"A50V50-{test_id}_retest_cleaned.csv"
        df.to_csv(out_path, index=False)
        print(
            f"retest {test_id}: {len(df)} rows, zeroed on first "
            f"{onset} samples (t < {df['time_s'].iloc[onset]:.3f} s), "
            f"strain unavailable -> {out_path}"
        )


if __name__ == "__main__":
    main()
