"""Clean the 2026-09-14 dogbone-lattice physical validation tests (one test
per material, run at each material's own "slow"-tier ramp/hold protocol --
same ramp duration and grip displacement as that material's flat-dogbone
"slow" rate-group test, reused directly as the dogbone-lattice FE job's
target protocol -- see dogbone-lattices/edelweissfe/generate_jobs.py).

No video/strain tracking here (unlike the flat-dogbone *-relax pipelines):
the FE comparison target is grip displacement/force, not gauge strain, so
only Ch:Position (mm) and Ch:Load (N) are needed.

A0V100 and A100V0 each have a rejected first attempt (wrong rate/short
duration -- see MOV count and short raw duration) alongside a "-retest"
("-retest20" for A100V0, reaching the intended ~20mm) file; only the
retest is used for those two materials.

Force sign convention: like every other material in this project except
A100V0, raw Ch:Load is NEGATIVE in tension here; A100V0's raw Ch:Load is
POSITIVE in tension (same quirk as A100V0-relax/sync_and_clean_data.py).
Negate so cleaned force_N is negative-in-tension for every material,
matching the flat-dogbone cleaned-file convention.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
CLEANED_DIR = Path(__file__).parent / "cleaned"

POSITION_STEP_MM = 0.0002

TESTS = {
    "A0V100": "A0V100-lattice-retest_09142026_180603.csv",
    "A25V75": "A25V75-lattice_09142026_182332.csv",
    "A50V50": "A50V50-lattice_09142026_184002.csv",
    "A75V25": "A75V25-lattice_09142026_185406.csv",
    "A100V0": "A100V0-lattice-retest20_09142026_192131.csv",
}

# raw Ch:Load sign convention differs only for A100V0 (see docstring)
POSITIVE_IN_TENSION_RAW = {"A100V0"}


def find_ramp_onset(position, step_threshold=POSITION_STEP_MM):
    """Same onset-detection convention used throughout this project
    (fit_qlv_model.py, the *-relax sync_and_clean_data.py scripts)."""
    baseline = position[0]
    deviation = np.abs(position - baseline)
    sustained = pd.Series(deviation).rolling(5).min() > step_threshold
    nonzero = sustained.to_numpy().nonzero()[0]
    return max(nonzero[0] - 4, 0) if len(nonzero) else 0


def load_relax_chunks(machine_name):
    """Concatenate a test's base raw CSV with its "_1", "_2", ... continuation
    chunks (contiguous Time (sec) values), the same pattern used by every
    other material's clean_data.py in this project."""
    stem = machine_name[: -len(".csv")]
    chunks = [RAW_DIR / machine_name]
    n = 1
    while (chunk := RAW_DIR / f"{stem}_{n}.csv").exists():
        chunks.append(chunk)
        n += 1
    return pd.concat(
        [pd.read_csv(c, usecols=["Time (sec)", "Ch:Position (mm)", "Ch:Load (N)"]) for c in chunks],
        ignore_index=True,
    )


def process(material, machine_name):
    df = load_relax_chunks(machine_name)
    df = df.rename(columns={"Time (sec)": "time_s", "Ch:Position (mm)": "position_mm",
                             "Ch:Load (N)": "load_N"})

    onset = find_ramp_onset(df["position_mm"].to_numpy())
    sign = -1.0 if material in POSITIVE_IN_TENSION_RAW else 1.0
    df["force_N"] = sign * df["load_N"]

    baseline = df[["position_mm", "force_N"]].iloc[: max(onset, 1)].mean()
    df[["position_mm", "force_N"]] = df[["position_mm", "force_N"]] - baseline

    out = df[["time_s", "position_mm", "force_N"]]
    out_path = CLEANED_DIR / f"{material}_cleaned.csv"
    out.to_csv(out_path, index=False)
    print(f"{material}: {len(out)} rows, t in [{out['time_s'].iloc[0]:.3f}, "
          f"{out['time_s'].iloc[-1]:.3f}]s, position in [{out['position_mm'].min():.4f}, "
          f"{out['position_mm'].max():.4f}]mm, peak |force|={out['force_N'].abs().max():.3f}N "
          f"-> {out_path}")


def main():
    CLEANED_DIR.mkdir(exist_ok=True)
    for material, machine_name in TESTS.items():
        process(material, machine_name)


if __name__ == "__main__":
    main()
