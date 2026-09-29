#!/usr/bin/env python3
"""
Generate standard DMA plots from the tidy CSVs produced by parse_dma.py.

For each <Sample>__temp_sweep.csv found under <Material>/processed/, produces
one figure per sample:
  - Top panel:    Storage modulus (E', solid) & Loss modulus (E'', dashed)
                  vs Temperature, log y-axis, one color per frequency.
  - Bottom panel: Tan(delta) vs Temperature, linear y-axis, one color per
                  frequency -- the peak marks Tg at that frequency.

Output: <Material>/processed/<Sample>__dma_temp_sweep.png

Usage:
  python3 plot_dma.py [DMA_ROOT]
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Validated categorical palette (dataviz skill, light mode), fixed order.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_sample(csv_path: Path, sample: str, material: str):
    df = pd.read_csv(csv_path)
    freq_col = "Angular frequency (rad/s)"
    df["freq_hz"] = df[freq_col] / (2 * math.pi)
    freqs = sorted(df["freq_hz"].unique())

    fig, (ax_mod, ax_tan) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
        facecolor=SURFACE,
    )

    for i, freq in enumerate(freqs):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        sub = df[df["freq_hz"] == freq].sort_values("Temperature (°C)")
        label = f"{freq:.2g} Hz"
        # Non-positive modulus values are single-point instrument-noise artifacts
        # near the resolution floor; mask them so the log-scale line breaks
        # cleanly instead of spiking to the axis floor. Raw CSV is left untouched.
        storage = sub["Storage modulus (kPa)"].where(sub["Storage modulus (kPa)"] > 0)
        loss = sub["Loss modulus (kPa)"].where(sub["Loss modulus (kPa)"] > 0)
        ax_mod.plot(sub["Temperature (°C)"], storage,
                    color=color, linewidth=1.6, marker="o", markersize=4,
                    linestyle="-", label=label, zorder=3)
        ax_mod.plot(sub["Temperature (°C)"], loss,
                    color=color, linewidth=1.4, marker="o", markersize=3,
                    linestyle="--", zorder=2)
        ax_tan.plot(sub["Temperature (°C)"], sub["Tan(delta)"],
                    color=color, linewidth=1.6, marker="o", markersize=4,
                    linestyle="-", label=label, zorder=3)

    ax_mod.set_yscale("log")
    ax_mod.set_ylabel("Modulus (kPa)")
    style_axis(ax_mod)
    freq_legend = ax_mod.legend(
        title="Frequency", loc="upper right", fontsize=8, title_fontsize=8,
        frameon=False, labelcolor=INK_SECONDARY,
    )
    ax_mod.add_artist(freq_legend)
    style_handles = [
        plt.Line2D([0], [0], color=INK_SECONDARY, linestyle="-", linewidth=1.6, label="Storage modulus (E')"),
        plt.Line2D([0], [0], color=INK_SECONDARY, linestyle="--", linewidth=1.4, label="Loss modulus (E'')"),
    ]
    ax_mod.legend(handles=style_handles, loc="lower left", fontsize=8,
                  frameon=False, labelcolor=INK_SECONDARY)

    ax_tan.set_ylabel("Tan(delta)")
    ax_tan.set_xlabel("Temperature (°C)")
    style_axis(ax_tan)
    ax_tan.legend(title="Frequency", loc="upper right", fontsize=8,
                  title_fontsize=8, frameon=False, labelcolor=INK_SECONDARY, ncol=2)

    fig.suptitle(f"{material} — {sample}: Oscillatory Temperature Sweep",
                 color=INK_PRIMARY, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = csv_path.parent / f"{sample}__dma_temp_sweep.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  [{material}/{sample}] -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=None,
                         help="DMA root directory (default: parent directory of this script)")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    print(f"Scanning DMA root: {root}")

    for material_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "scripts"):
        processed_dir = material_dir / "processed"
        if not processed_dir.is_dir():
            continue
        sweep_files = sorted(processed_dir.glob("*__temp_sweep.csv"))
        if not sweep_files:
            continue
        print(f"Material: {material_dir.name}")
        for sweep_csv in sweep_files:
            sample = sweep_csv.name[: -len("__temp_sweep.csv")]
            plot_sample(sweep_csv, sample, material_dir.name)


if __name__ == "__main__":
    main()
