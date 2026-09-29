"""Comparison figure combining the tensile lattice physical experiment screenshot
and the finite element simulation result (mirrored twice across x and y to reconstruct
the full 2D specimen).

Aligns the physical experiment and the FE simulation onto an identical physical coordinate
grid (in mm) so that all 15 columns of square lattice cells, struts, and fillets
vertically align one-to-one.

Follows plotstyle.py conventions and outputs high-resolution vector PDF and 300-DPI PNGs.
"""

import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors, figsize_double  # noqa: E402

HERE = Path(__file__).parent
PAPER_FIGS = Path(__file__).resolve().parents[2] / "paper" / "figures"
STRESS_DIR_QLV = REPO_ROOT / "simulations" / "fe-tension" / "postprocess" / "stress_contours"
STRESS_DIR_BB = REPO_ROOT / "simulations" / "fe-tension" / "postprocess" / "stress_contours_bb"
EXP_FRAME_PATH = REPO_ROOT / "data" / "tensile-lattice" / "raw" / "A25V75_lattice_frame.png"
VIDEO_PATH = REPO_ROOT / "data" / "tensile-lattice" / "raw" / "2026_09_14" / "D30A2009.MOV"


def ensure_experiment_frame():
    """Ensure the experiment reference frame exists, extracting from MOV if needed."""
    if not EXP_FRAME_PATH.exists():
        if not VIDEO_PATH.exists():
            raise FileNotFoundError(f"Neither {EXP_FRAME_PATH} nor {VIDEO_PATH} found.")
        print(f"Extracting frame from {VIDEO_PATH} at t=20s...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                "00:00:20",
                "-i",
                str(VIDEO_PATH),
                "-frames:v",
                "1",
                str(EXP_FRAME_PATH),
            ],
            check=True,
            capture_output=True,
        )
    return EXP_FRAME_PATH


def mirror_quads_and_stress(quads, stress):
    """Mirror the (x>=0, y>=0) octant mesh twice: across y=0 and across x=0.

    Reconstructs all four quadrants to form the full in-plane dogbone specimen.
    """
    quads_pp = quads.copy()  # (+x, +y)
    quads_pn = quads.copy()
    quads_pn[:, :, 1] = -quads_pn[:, :, 1]  # (+x, -y)
    quads_np = quads.copy()
    quads_np[:, :, 0] = -quads_np[:, :, 0]  # (-x, +y)
    quads_nn = quads.copy()
    quads_nn[:, :, 0] = -quads_nn[:, :, 0]
    quads_nn[:, :, 1] = -quads_nn[:, :, 1]  # (-x, -y)

    all_quads = np.concatenate([quads_pp, quads_pn, quads_np, quads_nn], axis=0)
    all_stress = np.concatenate([stress, stress, stress, stress], axis=0)
    return all_quads, all_stress


def get_aligned_experiment_crop(raw_img, xmin, xmax, ymin, ymax):
    """Crop the experiment photograph to exactly match the simulation domain in mm.

    Calibration:
      - 15 lattice hole columns (span 67.0 mm) span 733 pixels (1 mm = 10.9403 px).
      - Horizontal center x = 0 mm is at pixel x = 708.5.
      - 3 hole rows (span 10.0 mm) span 113 pixels (1 mm = 11.3000 px).
      - Vertical centerline y = 0 mm is at pixel y = 200.0.
    """
    px_xmin = int(round(708.5 + 10.9403 * xmin))
    px_xmax = int(round(708.5 + 10.9403 * xmax))
    px_ymin = int(round(200.0 - 11.3000 * ymax))
    px_ymax = int(round(200.0 - 11.3000 * ymin))

    cropped = raw_img.crop((px_xmin, px_ymin, px_xmax, px_ymax))
    return cropped


def plot_experiment_sim_comparison(material="A25V75", model="gm", panel_letters=False):
    """Generate 2-panel figure comparing experiment with specified FE model (GM or BB)."""
    ensure_experiment_frame()
    raw_img = Image.open(EXP_FRAME_PATH)

    data_dir = STRESS_DIR_QLV if model.lower() == "gm" else STRESS_DIR_BB

    npz_path = data_dir / f"{material}.npz"
    data = np.load(npz_path)
    all_quads, all_stress = mirror_quads_and_stress(data["quads"], data["stress"])

    xmin, xmax = all_quads[:, :, 0].min(), all_quads[:, :, 0].max()
    ymin, ymax = -13.0, 13.0

    cropped_exp = get_aligned_experiment_crop(raw_img, xmin, xmax, ymin, ymax)

    fig, (ax_exp, ax_sim) = plt.subplots(
        2, 1, figsize=(figsize_double[0], figsize_double[1] * 1.25), sharex=True, sharey=True
    )

    title_exp = r"\textbf{(a) Experiment}" if panel_letters else r"\textbf{Experiment}"
    title_sim = r"\textbf{(b) Simulation}" if panel_letters else r"\textbf{Simulation}"

    # Top: Experiment
    ax_exp.imshow(cropped_exp, extent=[xmin, xmax, ymin, ymax], aspect="equal")
    ax_exp.set_title(title_exp, fontsize=9, loc="left", pad=4)
    ax_exp.set_xlim(xmin, xmax)
    ax_exp.set_ylim(ymin, ymax)
    ax_exp.set_aspect("equal")
    ax_exp.axis("off")

    # Bottom: FE Simulation
    norm = plt.Normalize(vmin=all_stress.min(), vmax=all_stress.max())
    facecolors = plt.get_cmap("viridis")(norm(all_stress))
    coll = PolyCollection(all_quads, facecolors=facecolors, edgecolors=facecolors, linewidths=0.2)
    ax_sim.add_collection(coll)
    ax_sim.set_xlim(xmin, xmax)
    ax_sim.set_ylim(ymin, ymax)
    ax_sim.set_aspect("equal")
    ax_sim.set_title(title_sim, fontsize=9, loc="left", pad=4)
    ax_sim.axis("off")

    # Colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    cbar = fig.colorbar(
        mappable, ax=[ax_sim], orientation="horizontal", fraction=0.08, pad=0.12, shrink=0.45
    )
    cbar.set_label(r"Axial stress $\sigma_{xx}$ in MPa", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    fig.subplots_adjust(hspace=0.22)

    suffix = f"_{model.lower()}" if model.lower() != "gm" else ""
    out_stem = f"lattice_experiment_sim_comparison{suffix}_{material}"
    for ext in ("pdf", "png"):
        out_file = HERE / f"{out_stem}.{ext}"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        if PAPER_FIGS.exists():
            fig.savefig(PAPER_FIGS / f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved {out_file}")

    plt.close(fig)


def plot_experiment_sim_both_models(material="A25V75", panel_letters=False):
    """Generate 3-panel figure: Experiment, Simulation GM, Simulation BB."""
    ensure_experiment_frame()
    raw_img = Image.open(EXP_FRAME_PATH)

    data_qlv = np.load(STRESS_DIR_QLV / f"{material}.npz")
    quads_qlv, stress_qlv = mirror_quads_and_stress(data_qlv["quads"], data_qlv["stress"])

    data_bb = np.load(STRESS_DIR_BB / f"{material}.npz")
    quads_bb, stress_bb = mirror_quads_and_stress(data_bb["quads"], data_bb["stress"])

    xmin, xmax = quads_qlv[:, :, 0].min(), quads_qlv[:, :, 0].max()
    ymin, ymax = -13.0, 13.0

    cropped_exp = get_aligned_experiment_crop(raw_img, xmin, xmax, ymin, ymax)

    fig, (ax_exp, ax_gm, ax_bb) = plt.subplots(
        3, 1, figsize=(figsize_double[0], figsize_double[1] * 1.8), sharex=True, sharey=True
    )

    title_exp = r"\textbf{(a) Experiment}" if panel_letters else r"\textbf{Experiment}"
    title_gm = r"\textbf{(b) GM}" if panel_letters else r"\textbf{GM}"
    title_bb = r"\textbf{(c) BB}" if panel_letters else r"\textbf{BB}"

    # (a) Experiment
    ax_exp.imshow(cropped_exp, extent=[xmin, xmax, ymin, ymax], aspect="equal")
    ax_exp.set_title(title_exp, fontsize=9, loc="left", pad=4)
    ax_exp.axis("off")

    # (b) FE Simulation - GM
    norm_gm = plt.Normalize(vmin=stress_qlv.min(), vmax=stress_qlv.max())
    facecolors_gm = plt.get_cmap("viridis")(norm_gm(stress_qlv))
    coll_gm = PolyCollection(quads_qlv, facecolors=facecolors_gm, edgecolors=facecolors_gm, linewidths=0.2)
    ax_gm.add_collection(coll_gm)
    ax_gm.set_title(title_gm, fontsize=9, loc="left", pad=4)
    ax_gm.axis("off")

    # (c) FE Simulation - BB
    norm_bb = plt.Normalize(vmin=stress_bb.min(), vmax=stress_bb.max())
    facecolors_bb = plt.get_cmap("viridis")(norm_bb(stress_bb))
    coll_bb = PolyCollection(quads_bb, facecolors=facecolors_bb, edgecolors=facecolors_bb, linewidths=0.2)
    ax_bb.add_collection(coll_bb)
    ax_bb.set_title(title_bb, fontsize=9, loc="left", pad=4)
    ax_bb.axis("off")

    for ax in (ax_exp, ax_gm, ax_bb):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

    # Colorbar for GM
    mappable_gm = plt.cm.ScalarMappable(norm=norm_gm, cmap="viridis")
    cbar = fig.colorbar(
        mappable_gm, ax=[ax_bb], orientation="horizontal", fraction=0.08, pad=0.12, shrink=0.45
    )
    cbar.set_label(r"Axial stress $\sigma_{xx}$ in MPa", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    fig.subplots_adjust(hspace=0.25)

    out_stem = f"lattice_experiment_sim_comparison_both_{material}"
    for ext in ("pdf", "png"):
        out_file = HERE / f"{out_stem}.{ext}"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        if PAPER_FIGS.exists():
            fig.savefig(PAPER_FIGS / f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"saved {out_file}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment screenshot vs FE simulation comparison figure"
    )
    parser.add_argument(
        "--material", default="A25V75", help="Material to compare (default: A25V75)"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["gm", "bb", "both", "all"],
        help="FE model to compare: gm, bb, both (3-panel), or all (generates all variants)",
    )
    parser.add_argument(
        "--panel-letters",
        action="store_true",
        help="Include (a), (b), (c) subfigure prefixes in titles (default: False)",
    )
    args = parser.parse_args()

    if args.model in ("gm", "all"):
        print(f"Generating GM comparison for {args.material} (panel_letters={args.panel_letters})...")
        plot_experiment_sim_comparison(args.material, "gm", panel_letters=args.panel_letters)

    if args.model in ("bb", "all"):
        print(f"Generating BB comparison for {args.material} (panel_letters={args.panel_letters})...")
        plot_experiment_sim_comparison(args.material, "bb", panel_letters=args.panel_letters)

    if args.model in ("both", "all"):
        print(f"Generating 3-panel comparison for {args.material} (panel_letters={args.panel_letters})...")
        plot_experiment_sim_both_models(args.material, panel_letters=args.panel_letters)


if __name__ == "__main__":
    main()
