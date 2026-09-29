"""Axial-stress contour on the deformed shape of each lattice-dogbone
specimen, one panel per material, for the Bergstrom-Boyce predictive FE
runs. Companion to plot_lattice_stress_contours.py (QLV-Marmot) -- identical
logic, just reads from stress_contours_bb/ instead of stress_contours/ and
writes lattice_stress_contour_bb_{material} instead.

Shows only the meshed octant (not mirrored back to the full specimen
footprint) so the lattice strut/void detail renders larger and sharper.
Data is produced by
edelweissfe/extract_stress_contours_bb.py (needs
pyvista, run in the `dogbones` env). Panels are rotated 90 degrees (long
axis vertical). Color scales are not shared across materials.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import figsize_single  # noqa: E402

DATA_DIR = REPO_ROOT / "simulations" / "fe-tension" / "postprocess" / "stress_contours_bb"
OUT_BASE = Path(__file__).parent
MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]


def save(fig, name):
    for ext in ("pdf", "png"):
        out_path = OUT_BASE / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def plot_material(material):
    npz_path = DATA_DIR / f"{material}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    quads, stress = data["quads"], data["stress"]
    rotated = np.stack([-quads[:, :, 1], quads[:, :, 0]], axis=-1)

    fig, ax = plt.subplots(figsize=(figsize_single[0], figsize_single[1] * 1.6))
    norm = plt.Normalize(vmin=stress.min(), vmax=stress.max())
    facecolors = plt.get_cmap("viridis")(norm(stress))
    coll = PolyCollection(rotated, facecolors=facecolors, edgecolors=facecolors,
                           linewidths=0.3)
    ax.add_collection(coll)
    ax.set_xlim(rotated[:, :, 0].min(), rotated[:, :, 0].max())
    ax.set_ylim(rotated[:, :, 1].min(), rotated[:, :, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.09, pad=0.04)
    cbar.set_label("Axial stress in MPa")
    fig.tight_layout()
    save(fig, f"lattice_stress_contour_bb_{material}")
    plt.close(fig)
    return float(data["t_final"]), bool(data["complete"]), float(data["warp_scale"])


for material in MATERIALS:
    result = plot_material(material)
    if result is None:
        print(f"{material}: not yet run, skipping")
        continue
    t_final, complete, warp_scale = result
    status = "complete" if complete else "in progress"
    print(f"{material}: t_final={t_final:.2f}s warp_scale={warp_scale}x [{status}]")
