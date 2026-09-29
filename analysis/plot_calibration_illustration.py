"""Generate single-panel relaxation calibration plot for presentation slide 'Approach for objective 1'.

Shows experimental stress-relaxation data and calibrated Bergström-Boyce (BB) model curve
for material A25V75 at the very slow rate group, clearly resolving both the 50s ramp-up
and the 600s relaxation decay.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import colors
from fit_bb_model import BASE, MATERIALS, validate_bb
from fit_qlv_model import load

RMMSS_IMG_DIR = Path(__file__).resolve().parents[2] / "rmmss" / "img"
OUT_DIR = BASE
MATERIAL = "A25V75"
RATE_GROUP = "very slow"


def generate_plot():
    cfg = MATERIALS[MATERIAL]
    params_bb = pd.read_csv(BASE / "bb_params.csv")
    r_bb = params_bb[
        (params_bb["material"] == MATERIAL) & (params_bb["hyperelastic_base"] == "ArrudaBoyce")
    ].iloc[0]

    _, bb_plots = validate_bb(
        MATERIAL,
        cfg,
        "ArrudaBoyce",
        (r_bb["A1"], r_bb["A2"], r_bb["A3"]),
        (r_bb["B1"], r_bb["B2"], r_bb["B3"]),
        r_bb["c1"],
        r_bb["c2"],
        r_bb["c3"],
    )

    t_full, measured, t_ds, pred = bb_plots[RATE_GROUP]

    fig, ax = plt.subplots(figsize=(3.0, 2.35))

    # Faint individual experimental replicate curves
    test_ids = cfg["rate_groups"][RATE_GROUP]
    for test_id in test_ids:
        df = load(cfg, MATERIAL, test_id)
        ax.plot(df["time_s"], df["stress"], color="0.75", linewidth=0.9, alpha=0.45, zorder=1)

    # Representative/measured experimental curve
    ax.plot(
        t_full,
        measured,
        color="0.15",
        linewidth=1.7,
        linestyle="-",
        label="Experiment",
        zorder=2,
    )

    # Calibrated BB model prediction
    ax.plot(
        t_ds,
        pred,
        color=colors[1],
        linestyle="--",
        linewidth=1.7,
        marker="^",
        markersize=4.0,
        markevery=3,
        label="Model (BB)",
        zorder=3,
    )

    ax.set_xlabel("Time in s", fontsize=10)
    ax.set_ylabel("Stress in MPa", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True, alpha=0.35, linestyle="-")
    ax.set_ylim(-0.5, max(measured.max(), pred.max()) * 1.08)

    ax.legend(frameon=False, fontsize=9.5, loc="upper right")

    fig.tight_layout(pad=0.2)

    out_name = f"calibration_relaxation_{MATERIAL}"
    # Save in local dir
    for ext in ("pdf", "png"):
        local_path = OUT_DIR / f"{out_name}.{ext}"
        fig.savefig(local_path, dpi=300, bbox_inches="tight")
        print(f"Saved {local_path}")

    # Also save directly into rmmss/img if available
    if RMMSS_IMG_DIR.exists():
        for ext in ("pdf", "png"):
            rmmss_path = RMMSS_IMG_DIR / f"{out_name}.{ext}"
            fig.savefig(rmmss_path, dpi=300, bbox_inches="tight")
            print(f"Saved {rmmss_path}")

    plt.close(fig)


if __name__ == "__main__":
    generate_plot()
