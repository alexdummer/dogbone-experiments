"""Axial-stress contour on the deformed shape of each lattice-dogbone
specimen, one panel per material, for the paper's "Validation and
predictive capabilities" subsection. Companion to plot_lattice_fe_results.py
(force-time/force-disp curves) -- same FE runs, but a spatial rather than a
time view.

Shows only the meshed octant (not mirrored back to the full specimen
footprint) so the lattice strut/void detail renders larger and sharper in
the paper figure.

Data is produced by edelweissfe/extract_stress_contours.py
(needs pyvista, run in the `dogbones` env) and dumped to per-material .npz
files of octant quad footprints + per-quad sigma_xx -- this script only
needs numpy/matplotlib and follows plotstyle.py conventions.

Each material gets its own self-contained PDF (own viridis-mapped colorbar,
no title/axes -- those are added by the paper's TikZ arrangement) so the
panels can be arranged as columns and annotated directly in main.tex.
Panels are rotated 90 degrees (long/grip-to-grip axis vertical) so five of
them sit side by side as narrow columns instead of one wide horizontal strip.
Not sharing a color scale across materials: peak stress spans nearly two
orders of magnitude across the five blends (same reasoning as
plot_lattice_fe_results.py's sharey=False).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import figsize_single  # noqa: E402

DATA_DIR = REPO_ROOT / "simulations" / "fe-tension" / "postprocess" / "stress_contours"
OUT_BASE = Path(__file__).parent
MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]


def save(fig, name):
    for ext in ("pdf", "png"):
        out_path = OUT_BASE / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"saved {out_path}")


def plot_material(material):
    data = np.load(DATA_DIR / f"{material}.npz")
    quads, stress = data["quads"], data["stress"]
    # rotate 90 deg: plotted-x = -original-y, plotted-y = original-x
    # (long grip-to-grip axis becomes vertical, reads bottom-to-top)
    rotated = np.stack([-quads[:, :, 1], quads[:, :, 0]], axis=-1)

    fig, ax = plt.subplots(figsize=(figsize_single[0], figsize_single[1] * 1.6))
    # In the PDF (vector) backend, matplotlib's antialiased=False has no
    # effect -- adjacent polygon edges are rasterized independently by the
    # PDF viewer/renderer and still show hairline seams at any zoom level
    # unless each polygon's stroke is given the exact same color as its
    # fill, which papers over the seam regardless of renderer.
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
    save(fig, f"lattice_stress_contour_{material}")
    plt.close(fig)
    return float(data["t_final"]), bool(data["complete"]), float(data["warp_scale"])


for material in MATERIALS:
    t_final, complete, warp_scale = plot_material(material)
    status = "complete" if complete else "in progress"
    print(f"{material}: t_final={t_final:.2f}s warp_scale={warp_scale}x [{status}]")
