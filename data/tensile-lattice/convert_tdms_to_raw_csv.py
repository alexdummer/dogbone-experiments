"""Convert binary TDMS test machine files to raw CSV chunks.

Used for runs where the machine failed to write or export the raw CSV chunks
(e.g., A25V75-lattice-2 and A25V75-lattice-3 on 2026-09-30).
"""
import struct
import tempfile
from pathlib import Path

import numpy as np
import nptdms
import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"

CHANNEL_MAP = {
    "Time": "Time (sec)",
    "S:Load": "S:Load (lbs)",
    "S:Strain": "S:Strain (in)",
    "S:HL1": "S:HL1 (in)",
    "S:HL2": "S:HL2 (in)",
    "S:Position": "S:Position (in)",
    "Ch:Position": "Ch:Position (mm)",
    "Ch:Load": "Ch:Load (N)",
    "Ch:Stress": "Ch:Stress (PSI)",
    "Ch:Strain": "Ch:Strain (in/in)",
    "Set Point_X1": "Set Point_X1 ()",
    "Set Point_Command1": "Set Point_Command1 ()",
    "Set Point_Command2": "Set Point_Command2 ()",
    "CSP_X1": "CSP_X1 ()",
    "CSP_Command1": "CSP_Command1 ()",
    "CSP_Command2": "CSP_Command2 ()",
    "Cycles_X1": "Cycles_X1 ()",
    "Cycles_Command1": "Cycles_Command1 ()",
    "Cycles_Command2": "Cycles_Command2 ()",
    "Ch:Strain_Orig": "Ch:Strain_Orig ()",
}


def read_tdms(path: Path) -> nptdms.TdmsFile:
    """Read a TDMS file, handling trailing null-byte corruption if present."""
    try:
        return nptdms.TdmsFile.read(path)
    except Exception:
        with open(path, "rb") as f:
            data = f.read()
        lead_in = data[:28]
        tag, toc_mask, version, next_seg_offset, raw_data_offset = struct.unpack("<4sIIQQ", lead_in)
        valid_len = 28 + next_seg_offset
        with tempfile.NamedTemporaryFile(suffix=".tdms") as tf:
            tf.write(data[:valid_len])
            tf.flush()
            return nptdms.TdmsFile.read(tf.name)


def convert_run(stem: str, suffixes: list[str]) -> None:
    for s in suffixes:
        tdms_path = RAW_DIR / f"{stem}{s}.tdms"
        csv_path = RAW_DIR / f"{stem}{s}.csv"
        if not tdms_path.exists():
            continue
        print(f"Converting {tdms_path.name} -> {csv_path.name}...")
        tf = read_tdms(tdms_path)
        grp = tf["Real-Time"]
        time_len = len(grp["Time"])
        data = {}
        for ch_name, col_name in CHANNEL_MAP.items():
            if ch_name in grp and len(grp[ch_name]) == time_len:
                vals = grp[ch_name][:]
                if ch_name == "Time":
                    vals = vals / 1000.0
                data[col_name] = vals
            else:
                data[col_name] = np.zeros(time_len, dtype=float)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  Written: {len(df)} rows -> {csv_path.name}")


def main():
    convert_run("A25V75-lattice-2_09302026_092517", ["", "_1", "_2"])
    convert_run("A25V75-lattice-3_09302026_093822", ["", "_1", "_2"])


if __name__ == "__main__":
    main()
