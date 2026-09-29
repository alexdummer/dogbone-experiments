"""Extract the front-face (z=0 free surface) axial-stress contour of each
material's lattice-dogbone FE relaxation run, at the last available
increment, and dump it to a compact .npz for plotting elsewhere (this script
needs pyvista, which lives in the `dogbones` conda env; the actual
publication plot is made in the `marmot` env via
plot_lattice_stress_contours.py, which only needs numpy/matplotlib).

For each material:
  - reads ensightExport.case (written by the *fieldOutput / *fieldOutput
    export mechanism is separate -- this is the volume-mesh ensight export
    configured in generate_jobs.py)
  - takes the last available timestep (may be mid-run for a still-running
    material)
  - warps nodal coordinates by nodal displacement x WARP_SCALE[material] for
    visualization (raw displacements are far too small to see against the
    ~95x25mm specimen footprint)
  - extracts the quads on the true free surface z=0 (the FE model only
    covers z in [0, T/2] thanks to the mid-thickness mirror symmetry, so z=0
    IS the physical outer face, not a cut)
  - saves just the meshed octant (no mirroring across x=0/y=0) so the lattice
    strut/void detail renders at higher effective resolution in the paper
    figure -- quads (N,4,2), per-quad sigma_xx (N,), and whether the run is
    complete, to stress_contours/{material}.npz
"""
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).parent
OUT_DIR = HERE / "stress_contours"
OUT_DIR.mkdir(exist_ok=True)

MATERIALS = ["A0V100", "A25V75", "A50V50", "A75V25", "A100V0"]
N_EXPECTED = 41  # 2 steps x 20 increments + 1 initial zero increment

# Chosen so the exaggerated raw (pre-symmetry-doubling) axial displacement
# lands around 2-7mm across all five materials -- large enough to see the
# fillet/lattice deform, without the softer materials' true (larger) motion
# being blown up into something absurd. Annotated per-panel in the paper.
WARP_SCALE = {
    "A0V100": 10.0,
    "A25V75": 5.0,
    "A50V50": 2.0,
    "A75V25": 1.0,
    "A100V0": 1.0,
}


def process(material):
    case_path = HERE / material / "ensightExport.case"
    if not case_path.exists():
        print(f"{material}: no ensightExport.case yet, skipping")
        return

    reader = pv.get_reader(str(case_path))
    reader.set_active_time_set(1)  # time set 2 in the .case file (variables)
    n_t = reader.number_time_points
    reader.set_active_time_point(n_t - 1)
    t_final = reader.active_time_value
    complete = len(np.unique(np.round(reader.time_values, 6))) >= N_EXPECTED

    grid = reader.read()[0]  # block 0 = the volume mesh
    disp = grid.point_data["displacement"]
    warp_scale = WARP_SCALE[material]
    warped_points = grid.points + warp_scale * disp

    surf = grid.extract_surface(pass_pointid=True, pass_cellid=True, algorithm="dataset_surface")
    surf_to_grid_ids = surf.point_data["vtkOriginalPointIds"]
    front_mask = np.abs(surf.cell_centers().points[:, 2]) < 1e-6
    front = surf.extract_cells(np.where(front_mask)[0])

    # extract_cells() re-adds its own vtkOriginalPointIds/vtkOriginalCellIds,
    # overwriting surf's (relative to grid) with ones relative to surf itself
    # -- compose the two mappings to get front-local -> grid ids.
    front_to_surf_ids = front.point_data["vtkOriginalPointIds"]
    org_point_ids = surf_to_grid_ids[front_to_surf_ids]
    front_warped_xy = warped_points[org_point_ids][:, :2]
    sigma_xx = front.cell_data["stress"][:, 0]

    quads = np.empty((front.n_cells, 4, 2))
    for i in range(front.n_cells):
        local_ids = front.get_cell(i).point_ids
        corners = front_warped_xy[local_ids]
        # The lattice-region (transfinite) and fillet-region (general-mesher)
        # hexes were stitched together with different quad winding
        # conventions (see mesh_lattice.py's winding-normalization fix for
        # the volume mesh); the FRONT-FACE quads inherited from them are not
        # consistently wound, so naively connecting point_ids in native
        # order draws self-intersecting bowties for roughly half the faces.
        # Sorting corners by angle around their own centroid always gives a
        # proper simple polygon, independent of the source ordering.
        centroid = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        quads[i] = corners[np.argsort(angles)]

    out_path = OUT_DIR / f"{material}.npz"
    np.savez(out_path, quads=quads, stress=sigma_xx, t_final=t_final,
              complete=complete, warp_scale=warp_scale)
    status = "complete" if complete else f"in progress (t={t_final:.2f}s, {n_t} increments)"
    print(f"{material}: {status} -> {out_path} ({quads.shape[0]} quads)")


for material in MATERIALS:
    process(material)
