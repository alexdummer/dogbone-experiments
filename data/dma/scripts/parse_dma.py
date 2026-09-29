#!/usr/bin/env python3
"""
Parse TA Instruments TRIOS multi-step DMA/TMA export (.txt) files into tidy CSVs.

Layout assumed:
  <DMA_ROOT>/<Material>/<Sample>.txt      (raw TRIOS export, one per replicate)

Output:
  <DMA_ROOT>/<Material>/processed/<Sample>__<step_kind>.csv   (one per procedure step)
  <DMA_ROOT>/<Material>/processed/<Sample>__metadata.csv      (header key/value info)
  <DMA_ROOT>/processed_summary.csv                            (one row per input file)

Usage:
  python3 parse_dma.py [DMA_ROOT]        # default: parent of this script's dir
  python3 parse_dma.py --file path/to/one/file.txt   # process a single file
"""
import argparse
import csv
import re
from pathlib import Path

STEP_KIND_PATTERNS = [
    (re.compile(r"ramp", re.I), "tma_ramp"),
    (re.compile(r"axial", re.I), "axial_preload"),
    (re.compile(r"sweep", re.I), "temp_sweep"),
]


def classify_step(step_name: str) -> str:
    for pattern, kind in STEP_KIND_PATTERNS:
        if pattern.search(step_name):
            return kind
    return re.sub(r"[^A-Za-z0-9]+", "_", step_name).strip("_").lower() or "step"


def parse_trios_file(path: Path):
    """Returns (metadata: dict, steps: list of {name, kind, columns, units, rows}) """
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln.rstrip("\r") for ln in raw_lines]

    metadata = {}
    steps = []
    section = None
    i = 0
    n = len(lines)

    # --- header block, up to first [step] ---
    while i < n and lines[i].strip() != "[step]":
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        elif stripped and "\t" in line:
            key, _, value = line.partition("\t")
            key = key.strip()
            value = value.strip()
            if key:
                full_key = f"{section}.{key}" if section else key
                metadata[full_key] = value
        elif stripped:
            # e.g. multi-line "Procedure name" continuation rows (no tab)
            key = f"{section}.note" if section else "note"
            metadata[key] = (metadata.get(key, "") + " | " + stripped).strip(" |")
        i += 1

    # --- step blocks ---
    while i < n:
        if lines[i].strip() != "[step]":
            i += 1
            continue
        i += 1  # skip [step]
        if i >= n:
            break
        step_name = lines[i].strip()
        i += 1
        if i >= n:
            break
        columns = [c.strip() for c in lines[i].split("\t")]
        i += 1
        units = [u.strip() for u in lines[i].split("\t")] if i < n else []
        i += 1

        rows = []
        while i < n and lines[i].strip() != "" and lines[i].strip() != "[step]":
            rows.append([v.strip() for v in lines[i].split("\t")])
            i += 1
        # skip blank separator line(s)
        while i < n and lines[i].strip() == "":
            i += 1

        steps.append({
            "name": step_name,
            "kind": classify_step(step_name),
            "columns": columns,
            "units": units,
            "rows": rows,
        })

    return metadata, steps


def write_step_csv(out_path: Path, step: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            f"{col} ({unit})" if unit else col
            for col, unit in zip(step["columns"], step["units"] + [""] * len(step["columns"]))
        ]
        writer.writerow(header)
        writer.writerows(step["rows"])


def write_metadata_csv(out_path: Path, metadata: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in metadata.items():
            writer.writerow([k, v])


def process_file(path: Path, material: str, summary_rows: list):
    metadata, steps = parse_trios_file(path)
    out_dir = path.parent / "processed"
    sample = path.stem

    for step in steps:
        n_rows = len(step["rows"])
        n_cols = len(step["columns"])
        out_path = out_dir / f"{sample}__{step['kind']}.csv"
        write_step_csv(out_path, step)
        print(f"  [{material}/{sample}] step '{step['name']}' -> {out_path.name} "
              f"({n_rows} rows x {n_cols} cols)")

    write_metadata_csv(out_dir / f"{sample}__metadata.csv", metadata)

    summary_rows.append({
        "material": material,
        "sample": sample,
        "file": str(path),
        "rundate": metadata.get("File Parameters.Run date", metadata.get("rundate", "")),
        "geometry": metadata.get("Geometry Parameters.Name", ""),
        "length_mm": metadata.get("Geometry Parameters.Length", "").replace(" mm", ""),
        "width_mm": metadata.get("Geometry Parameters.Width", "").replace(" mm", ""),
        "thickness_mm": metadata.get("Geometry Parameters.Thickness", "").replace(" mm", ""),
        "n_steps": len(steps),
        "steps": ";".join(s["kind"] for s in steps),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=None,
                         help="DMA root directory containing one subfolder per material "
                              "(default: parent directory of this script)")
    parser.add_argument("--file", default=None,
                         help="Process a single .txt file instead of scanning a root directory")
    args = parser.parse_args()

    summary_rows = []

    if args.file:
        path = Path(args.file).resolve()
        material = path.parent.name
        print(f"Processing single file: {path}")
        process_file(path, material, summary_rows)
        root = path.parent.parent
    else:
        root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
        print(f"Scanning DMA root: {root}")
        for material_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "scripts"):
            txt_files = sorted(material_dir.glob("*.txt"))
            if not txt_files:
                continue
            print(f"Material: {material_dir.name}")
            for txt_file in txt_files:
                process_file(txt_file, material_dir.name, summary_rows)

    if summary_rows:
        summary_path = root / "processed_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to {summary_path} ({len(summary_rows)} file(s))")


if __name__ == "__main__":
    main()
