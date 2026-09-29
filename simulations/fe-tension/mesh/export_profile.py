"""Export the two pieces mesh_lattice.py needs to build a structured-lattice
FE mesh of the quarter-symmetry (x>=0, y>=0) dogbone-lattice gauge specimen:

  - dogbone_profile.step: the fillet+grip envelope (lattice gauge box already
    excised), truncated GRIP_MESH_CUT mm short of the true grip end (the
    grip is rigid, so most of its length doesn't need meshing at all -- see
    mesh_lattice.py's module docstring). This piece needs no x=0/y=0
    symmetry-plane clipping: create_dogbone()'s 'fillet_grip_region' is
    already a clean quadrant boundary (x>=0, y>=0) by construction.
  - lattice_params.json: the lattice grid parameters (cell counts, pitch,
    strut width) mesh_lattice.py needs to build the lattice struts itself as
    separate, non-overlapping gmsh rectangles (for per-strut structured/
    transfinite meshing) and glue them to the fillet piece via a conformal
    boolean fragment -- create_dogbone()'s own lattice construction fuses
    every strut into a single face with overlap-padded (double-covering)
    junctions, which is right for robust printing/mirroring but unusable for
    per-strut structured FE meshing.
"""
import json
import sys
sys.path.insert(0, "..")
import cadquery as cq
from create_specimens import create_dogbone

# Must match create_dogbone()'s defaults.
L = 115.0
W = 25.0
T = 2.0
L_GRIP = 11.0
X_GRIP = L / 2.0 - L_GRIP  # 46.5, the fillet/grip transition (unchanged)

# The real grip clamp is rigid: no relevant deformation happens inside it, so
# most of its length doesn't need to be meshed at all -- truncate the mesh
# GRIP_MESH_CUT mm short of the true end (L/2=57.5), leaving only a thin
# buffer past the fillet-grip junction, and apply the "moving grip" BC
# directly on the new cut face (mesh_lattice.py's movingGrip nSet) instead of
# on the whole grip block.
GRIP_MESH_CUT = 10.0  # mm removed from the mesh's outer end
X_CUT = L / 2.0 - GRIP_MESH_CUT  # 47.5; leaves a 1mm buffer past X_GRIP=46.5

specimen, quadrant, fillet_grip_region, lattice_params = create_dogbone()

clip_box = cq.Workplane("XY").rect(X_CUT, W / 2.0, centered=False).extrude(T)
clipped_fillet_region = fillet_grip_region.intersect(clip_box)

bottom_face = clipped_fillet_region.faces("<Z")
cq.exporters.export(bottom_face, "dogbone_profile.step")

with open("lattice_params.json", "w") as f:
    json.dump({**lattice_params, "T": T, "X_CUT": X_CUT}, f, indent=2)

print("wrote dogbone_profile.step (fillet+grip envelope only, grip truncated at "
      f"x={X_CUT}) and lattice_params.json")
