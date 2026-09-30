"""Clean the dogbone-lattice physical validation tests across all replicates:
- Replicate 1: 2026-09-14
- Replicates 2 & 3: 2026-09-30

Tests are run at each material's own "slow"-tier ramp/hold protocol --
same ramp duration and grip displacement as that material's flat-dogbone
"slow" rate-group test, reused directly as the dogbone-lattice FE job's
target protocol (see simulations/fe-tension/generate_jobs.py).

No video/strain tracking here (unlike the flat-dogbone *-relax pipelines):
the FE comparison target is grip displacement/force, not gauge strain, so
only Ch:Position (mm) and Ch:Load (N) are needed.

Retests / exclusions:
- A0V100 rep 1: 09142026_180603 retest used (initial run was aborted/short).
- A0V100 rep 2: 09302026_090943 retest used (initial run had grip slippage note in .cmm).
- A25V75 rep 2 & 3: raw CSV chunks converted from streaming TDMS (machine CSVs were 0 bytes).
- A50V50 rep 2: 09302026_101711 retest used (initial run aborted after <0.5MB).
- A100V0 rep 1: 09142026_192131 retest20 used (reached intended ~20mm).
- A100V0 rep 2: 09302026_121434 retest used (initial run aborted after 3.5s).

Force sign convention: like every other material in this project except
A100V0, raw Ch:Load is NEGATIVE in tension here; A100V0's raw Ch:Load is
POSITIVE in tension (same quirk as A100V0-relax/sync_and_clean_data.py).
Negate so cleaned force_N is negative-in-tension for every material,
matching the flat-dogbone cleaned-file convention.

Outputs per material:
- {material}-{rep}_cleaned.csv for rep in (1, 2, 3)
- {material}_cleaned.csv (mean curve across all 3 replicates)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
CLEANED_DIR = Path(__file__).parent / "cleaned"

POSITION_STEP_MM = 0.0002

REPLICATES = {
    "A0V100": {
        1: "A0V100-lattice-retest_09142026_180603.csv",
        2: "A0V100-lattice-2-retest_09302026_090943.csv",
        3: "A0V100-lattice-3_09302026_085459.csv",
    },
    "A25V75": {
        1: "A25V75-lattice_09142026_182332.csv",
        2: "A25V75-lattice-2_09302026_092517.csv",
        3: "A25V75-lattice-3_09302026_093822.csv",
    },
    "A50V50": {
        1: "A50V50-lattice_09142026_184002.csv",
        2: "A50V50-lattice-2-retest_09302026_101711.csv",
        3: "A50V50-lattice-3_09302026_102948.csv",
    },
    "A75V25": {
        1: "A75V25-lattice_09142026_185406.csv",
        2: "A75V25-lattice-2_09302026_104603.csv",
        3: "A75V25-lattice-3_09302026_105937.csv",
    },
    "A100V0": {
        1: "A100V0-lattice-retest20_09142026_192131.csv",
        2: "A100V0-lattice-2-retest_09302026_121434.csv",
        3: "A100V0-lattice-3_09302026_122826.csv",
    },
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


def process_replicate(material, rep, machine_name):
    df = load_relax_chunks(machine_name)
    df = df.rename(columns={"Time (sec)": "time_s", "Ch:Position (mm)": "position_mm",
                             "Ch:Load (N)": "load_N"})

    onset = find_ramp_onset(df["position_mm"].to_numpy())
    sign = -1.0 if material in POSITIVE_IN_TENSION_RAW else 1.0
    df["force_N"] = sign * df["load_N"]

    baseline = df[["position_mm", "force_N"]].iloc[: max(onset, 1)].mean()
    df[["position_mm", "force_N"]] = df[["position_mm", "force_N"]] - baseline

    out = df[["time_s", "position_mm", "force_N"]].copy()
    out_path = CLEANED_DIR / f"{material}-{rep}_cleaned.csv"
    out.to_csv(out_path, index=False)
    print(f"  Rep {rep}: {len(out)} rows, t in [{out['time_s'].iloc[0]:.3f}, "
          f"{out['time_s'].iloc[-1]:.3f}]s, position in [{out['position_mm'].min():.4f}, "
          f"{out['position_mm'].max():.4f}]mm, peak |force|={out['force_N'].abs().max():.3f}N "
          f"-> {out_path.name}")
    return out


def compute_mean_response(material, rep_dfs):
    """Interpolate all replicates onto a common uniform 1 kHz time grid and compute mean."""
    min_end = min(df["time_s"].iloc[-1] for df in rep_dfs)
    t_grid = np.arange(0.001, min_end, 0.001)

    interp_pos = [np.interp(t_grid, df["time_s"], df["position_mm"]) for df in rep_dfs]
    interp_force = [np.interp(t_grid, df["time_s"], df["force_N"]) for df in rep_dfs]

    mean_pos = np.mean(interp_pos, axis=0)
    mean_force = np.mean(interp_force, axis=0)

    out = pd.DataFrame({"time_s": t_grid, "position_mm": mean_pos, "force_N": mean_force})
    out_path = CLEANED_DIR / f"{material}_cleaned.csv"
    out.to_csv(out_path, index=False)
    print(f"  Mean ({len(rep_dfs)} reps): {len(out)} rows, peak |force|={out['force_N'].abs().max():.3f}N -> {out_path.name}")


def main():
    CLEANED_DIR.mkdir(exist_ok=True)
    for material, reps in REPLICATES.items():
        print(f"Processing {material}...")
        rep_dfs = []
        for rep, machine_name in sorted(reps.items()):
            df = process_replicate(material, rep, machine_name)
            rep_dfs.append(df)
        compute_mean_response(material, rep_dfs)


if __name__ == "__main__":
    main()

