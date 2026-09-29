"""Mesh the dogbone-lattice gauge specimen into an all-hexahedron mesh with a
genuinely structured (transfinite) lattice region.

Two pieces (from export_profile.py) are combined:
  - dogbone_profile.step: the fillet+grip envelope (lattice box excised,
    grip truncated 10mm short of its true end -- see export_profile.py),
    a single, geometrically irregular face meshed with the general
    recombination/subdivision algorithm.
  - the lattice struts, built HERE directly as separate, non-overlapping,
    axis-aligned gmsh rectangles from lattice_params.json (create_dogbone()'s
    own lattice sketch fuses every strut into one overlap-padded face, which
    is right for robust printing/mirroring but unusable for per-strut
    structured meshing: overlapping struts would double-mesh their shared
    corners if kept separate). Each strut is meshed as a clean transfinite
    (structured) quad grid.

The two pieces are glued into one conformal topology via gmsh's boolean
fragment (shared edges become common, not overlapping), then the whole
profile is swept through HALF the thickness -- the specimen is a uniform
extrusion, so this reproduces the exact geometry without a general 3D tet
mesh (Marmot's finite-strain elements only support hex8/hex20, not tets).

This models only the (x>=0, y>=0, z<=T/2) octant of the specimen, exploiting
three mirror symmetries of the tension-relaxation test:
  - x=0: the physical "fixed grip moves by 0, moving grip moves by u_final"
    test is kinematically equivalent (internal stress depends only on
    relative grip displacement) to both grips moving apart symmetrically by
    u_final/2 -- so only x in [0, L/2] needs meshing, with a u_x=0 mirror BC
    at x=0 (see generate_jobs.py, which halves u_final accordingly).
  - y=0: the geometry, material and loading are all symmetric about the
    specimen's width-wise centerline, so only y in [0, W/2] is needed, with a
    u_y=0 mirror BC at y=0.
  - z=T/2: the specimen is a uniform through-thickness extrusion with no
    z-dependent loading, so only z in [0, T/2] is needed, with a u_z=0 mirror
    BC at the new z=T/2 face (the true z=0 face stays a free surface).
Combined, this is 1/8 the elements of the full-specimen mesh.

The grip block itself is rigid, so the mesh is truncated 10mm short of its
true end; the "moving grip" BC is applied directly on that cut face.

Writes lattice_mesh.inc, an EdelweissFE/Abaqus-style include file with
*node/*element/*nSet blocks (element type C3D8UL), shared across all 5
material variants -- the geometry does not depend on the material blend.
"""
import json
import gmsh
import sys

T_HALF = 1.0  # half the specimen thickness (mm); full T=2.0 in create_specimens.py
N_LAYERS = 2  # elements through T_HALF
MESH_SIZE = 0.35  # mm, target in-plane element size (~5-6 elements across the
                   # current 2.0mm lattice strut width; unchanged from the
                   # original 0.8mm-strut geometry, just finer relatively now)

L = 115.0
W = 25.0
L_GRIP = 11.0
X_GRIP = L / 2.0 - L_GRIP  # 46.5, the true fillet/grip transition
GRIP_MESH_CUT = 10.0  # mm removed from the mesh's outer end, see export_profile.py
X_CUT = L / 2.0 - GRIP_MESH_CUT  # 47.5, the mesh's truncated outer boundary

with open("lattice_params.json") as f:
    P = json.load(f)
Lx, Ly = P["Lx"], P["Ly"]
n_x, n_half, row_offset = P["n_x"], P["n_half"], P["row_offset"]
p_x, p_y = P["p_x"], P["p_y"]
sw = P["strut_width"]
Wc_half = P["Wc_half"]  # true box height -- NOT Ly (just the row-pitch
                         # reference used to *position* h_locs)

# cadquery ground-truth cross-check (see the investigation that found the
# original version of this script's geometry bug: it used Ly instead of
# Wc_half for the vertical struts' height, and gave the k=n_x boundary strut
# full width instead of half, undershooting the true lattice area). Computed
# directly from create_dogbone()'s own construction:
#   quadrant.intersect(clip_box).faces("<Z") area, minus fillet_grip_region's.
FILLET_FACE_AREA = 150.17668810758002
TRUE_LATTICE_AREA = 197.00000000000003


def vertical_strut_x_range(k):
    # Every vertical strut (interior or boundary) is centered on its grid
    # line k*p_x with the full strut_width -- only the k=0/k=n_x boundary
    # struts get halved, since the other half is provided elsewhere (mirror
    # symmetry for k=0, the already-solid fillet envelope for k=n_x) and
    # would otherwise double-cover that material in our separate-pieces
    # reconstruction.
    if k == 0:
        return 0.0, sw / 2.0
    if k == n_x:
        return Lx - sw / 2.0, Lx
    return k * p_x - sw / 2.0, k * p_x + sw / 2.0


def horizontal_row_y_range(i):
    # Row i's full centered extent is [(i+row_offset)*p_y - sw/2,
    # (i+row_offset)*p_y + sw/2]. create_dogbone() phases rows by parity
    # (row_offset=0 for even n_cells, 0.5 for odd) so that mirroring about
    # y=0 always reunites two half-features into one whole one: for even
    # n_cells row 0 is centered AT y=0 (a strut straddles the centerline,
    # needing the same half-height clip as the old always-even code); for
    # odd n_cells row 0 is offset half a cell away from y=0 (a *void* cell
    # straddles the centerline instead, and row 0 is an ordinary, entirely
    # in-bounds row needing no clipping at all). Clamping at 0 handles both
    # cases uniformly instead of special-casing on parity directly. The
    # outermost row (i=n_half) is never affected: its extent always sits
    # entirely inside [0, Wc_half] by construction (Wc_half = Ly + sw/2).
    y_center = (i + row_offset) * p_y
    return max(0.0, y_center - sw / 2.0), y_center + sw / 2.0


gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("dogbone_lattice_octant")

gmsh.merge("dogbone_profile.step")
gmsh.model.occ.synchronize()
fillet_tags = [tag for dim, tag in gmsh.model.getEntities(2)]
print(f"Imported {len(fillet_tags)} fillet/grip surface(s) from STEP.")

# --- build the lattice struts as separate, non-overlapping rectangles ---
vertical_ranges = [vertical_strut_x_range(k) for k in range(n_x + 1)]
vertical_ranges.sort()

lattice_tags = []
lattice_corners = {}  # tag -> (xmin, ymin, xmax, ymax), known analytically
for xmin, xmax in vertical_ranges:
    tag = gmsh.model.occ.addRectangle(xmin, 0.0, 0.0, xmax - xmin, Wc_half)
    lattice_tags.append(tag)
    lattice_corners[tag] = (xmin, 0.0, xmax, Wc_half)

for i in range(n_half + 1):
    ymin, ymax = horizontal_row_y_range(i)
    # horizontal segments fill the gaps between consecutive vertical struts
    for (_, xprev_max), (xnext_min, _) in zip(vertical_ranges[:-1], vertical_ranges[1:]):
        if xnext_min - xprev_max < 1e-9:
            continue
        tag = gmsh.model.occ.addRectangle(xprev_max, ymin, 0.0, xnext_min - xprev_max, ymax - ymin)
        lattice_tags.append(tag)
        lattice_corners[tag] = (xprev_max, ymin, xnext_min, ymax)

gmsh.model.occ.synchronize()
print(f"Built {len(lattice_tags)} non-overlapping lattice strut rectangles.")

# --- glue the fillet piece and the lattice struts into one conformal model ---
allInput = [(2, t) for t in fillet_tags + lattice_tags]
outDimTags, outDimTagsMap = gmsh.model.occ.fragment(allInput, [])
gmsh.model.occ.synchronize()

# outDimTagsMap[i] holds the output (dim,tag) pairs produced from allInput[i];
# the first len(fillet_tags) entries are fillet-derived, the rest lattice-derived
# (1:1, since our rectangles don't overlap and so aren't split by the fragment
# -- only their boundary edges pick up extra T-junction vertices from
# neighboring struts touching them).
fillet_out = set()
lattice_out = {}  # output tag -> (xmin, ymin, xmax, ymax)
for i, dimTags in enumerate(outDimTagsMap):
    if i < len(fillet_tags):
        for dim, tag in dimTags:
            if dim == 2:
                fillet_out.add(tag)
    else:
        corners = lattice_corners[lattice_tags[i - len(fillet_tags)]]
        for dim, tag in dimTags:
            if dim == 2:
                lattice_out[tag] = corners
for tag in fillet_out:
    lattice_out.pop(tag, None)  # a surface touched by both keeps its fillet treatment

print(f"After fragment: {len(fillet_out)} fillet surface(s), {len(lattice_out)} lattice surface(s).")

gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE * 0.5)
gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
# Guarantee an all-quad mesh on the (general-algorithm) fillet surface: any
# leftover triangles are subdivided into quads as a post-process. Recombine
# is set explicitly per-surface below (transfinite lattice struts) and via
# the fillet surface's own setRecombine call, rather than the blanket
# RecombineAll/RecombinationAlgorithm options, which conflict with having
# some surfaces already transfinite-meshed.
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # simple full-quad
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # all quadrangles (2D)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
for tag in fillet_out:
    gmsh.model.mesh.setRecombine(2, tag)


def edge_length(curveTag):
    bbox = gmsh.model.getBoundingBox(1, curveTag)
    dx, dy, dz = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def edge_bbox(curveTag):
    return gmsh.model.getBoundingBox(1, curveTag)


def point_at(x, y, eps=1e-6):
    pts = gmsh.model.getEntitiesInBoundingBox(x - eps, y - eps, -eps, x + eps, y + eps, eps, 0)
    if len(pts) != 1:
        raise RuntimeError(f"expected exactly one point at ({x}, {y}), found {len(pts)}")
    return pts[0][1]


# Every strut rectangle picks up extra vertices along its sides wherever a
# neighboring strut's corner touches it (T-junctions from the conformal
# fragment), so a side is generally a *chain* of collinear edges, not a
# single one. A per-edge count of round(length/MESH_SIZE) does not
# necessarily sum to the same total on both sides of a chained dimension as
# on the (single-edge) opposite side, which the transfinite algorithm
# requires -- so counts are memoized per curve tag (a shared edge always
# gets the same count, wherever it's referenced from), and for any side that
# is a single, unfragmented edge, its count is instead forced to match
# whatever its (possibly multi-edge) opposite side naturally sums to.
curveCount = {}


def count_for(curveTag):
    if curveTag not in curveCount:
        curveCount[curveTag] = max(1, round(edge_length(curveTag) / MESH_SIZE))
    return curveCount[curveTag]


def classify_sides(edges, xmin, ymin, xmax, ymax, tol=1e-6):
    sides = {"left": [], "right": [], "bottom": [], "top": []}
    for e in edges:
        x0, y0, _, x1, y1, _ = edge_bbox(e)
        if abs(x1 - x0) < tol:  # vertical edge -> left/right side
            sides["left" if abs(x0 - xmin) < tol else "right"].append(e)
        elif abs(y1 - y0) < tol:  # horizontal edge -> bottom/top side
            sides["bottom" if abs(y0 - ymin) < tol else "top"].append(e)
        else:
            raise RuntimeError(f"edge {e} is neither purely vertical nor horizontal")
    return sides


nTransfinite = 0
for tag, (xmin, ymin, xmax, ymax) in lattice_out.items():
    boundary = gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False, recursive=False)
    edges = [b[1] for b in boundary if b[0] == 1]
    if any(gmsh.model.getType(1, e) != "Line" for e in edges):
        print(f"  WARNING: lattice surface {tag} has a non-straight boundary edge "
              "-- left on the general algorithm.")
        continue

    sides = classify_sides(edges, xmin, ymin, xmax, ymax)
    for a, b in (("left", "right"), ("bottom", "top")):
        sumA = sum(count_for(e) for e in sides[a])
        sumB = sum(count_for(e) for e in sides[b])
        if sumA == sumB:
            continue
        # force whichever side is a single, unfragmented edge to match the
        # other (possibly multi-edge) side's natural sum
        if len(sides[a]) == 1 and len(sides[b]) > 1:
            curveCount[sides[a][0]] = sumB
        elif len(sides[b]) == 1 and len(sides[a]) > 1:
            curveCount[sides[b][0]] = sumA
        else:
            raise RuntimeError(
                f"surface {tag}: sides '{a}'/'{b}' both multi-edge or both "
                f"single-edge but counts disagree ({sumA} != {sumB})"
            )

    for e in edges:
        gmsh.model.mesh.setTransfiniteCurve(e, count_for(e) + 1)
    corners = [
        point_at(xmin, ymin), point_at(xmax, ymin),
        point_at(xmax, ymax), point_at(xmin, ymax),
    ]
    gmsh.model.mesh.setTransfiniteSurface(tag, cornerTags=corners)
    gmsh.model.mesh.setRecombine(2, tag)
    nTransfinite += 1

print(f"Forced structured (transfinite) quads on {nTransfinite}/{len(lattice_out)} lattice surfaces.")

# Mesh the 2D profile only. gmsh's own occ.extrude(..., recombine=True)
# re-triangulates the source surfaces internally as part of the 3D sweep,
# which does not reliably reapply the subdivision-to-quads fallback (it
# works when generate(2) is called standalone, but the leftover triangles
# reappear once the same surface is re-meshed during an extrude), so the Z
# direction is instead extruded by hand below, directly reusing this
# (verified all-quad) 2D mesh.
gmsh.model.mesh.generate(2)

elTypes2, elTags2, elNodeTags2 = gmsh.model.mesh.getElements(2)
nTri = sum(len(tags) for t, tags in zip(elTypes2, elTags2) if t == 2)
quads2D = []
for t, tags, nodeTagsForType in zip(elTypes2, elTags2, elNodeTags2):
    if t != 3:  # 3 = 4-node quadrangle
        continue
    for i in range(len(tags)):
        quads2D.append(list(nodeTagsForType[4 * i:4 * i + 4]))
print(f"2D mesh: {sum(len(tags) for tags in elTags2)} elements, {nTri} triangles "
      f"remaining (should be 0), {len(quads2D)} quads.")
if nTri > 0:
    print("ERROR: non-quad elements present in the 2D profile mesh.", file=sys.stderr)
    sys.exit(1)

nodeTags2D, nodeCoords2D, _ = gmsh.model.mesh.getNodes()
coords2D = {tag: (nodeCoords2D[3 * i], nodeCoords2D[3 * i + 1]) for i, tag in enumerate(nodeTags2D)}


def signed_area(nodes):
    pts = [coords2D[n] for n in nodes]
    area = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    return area / 2.0


# The fillet surface's general-algorithm quads and the lattice's transfinite
# quads come out with OPPOSITE winding (confirmed: fillet is clockwise, all
# lattice struts are counter-clockwise) -- gmsh's 2D meshers don't guarantee a
# common winding across different algorithms/surfaces the way they do within
# a single surface. Since the manual Z-extrusion below maps each quad's node
# order directly from the bottom layer to the top layer, a mix of windings
# produces inverted (negative-Jacobian) hexahedra on whichever region has the
# "wrong" sign -- normalize every quad to counter-clockwise here so the
# extrusion is correct everywhere.
nFlipped = 0
for q in quads2D:
    if signed_area(q) < 0:
        q.reverse()
        nFlipped += 1
print(f"Normalized winding: flipped {nFlipped}/{len(quads2D)} quads to counter-clockwise.")


def quad_area(nodes):
    pts = [coords2D[n] for n in nodes]
    area = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


# --- area sanity checks ---
# (1) self-consistency: does the actual meshed footprint match what our own
#     construction formulas above say it should be? (catches meshing bugs,
#     but NOT errors in the formulas themselves -- see (2).)
latticeArea = sum((xmax - xmin) * Wc_half for xmin, xmax in vertical_ranges)
for i in range(n_half + 1):
    ymin, ymax = horizontal_row_y_range(i)
    for (_, xprev_max), (xnext_min, _) in zip(vertical_ranges[:-1], vertical_ranges[1:]):
        if xnext_min - xprev_max > 1e-9:
            latticeArea += (xnext_min - xprev_max) * (ymax - ymin)
expectedFootprint = FILLET_FACE_AREA + latticeArea
meshedFootprint = sum(quad_area(q) for q in quads2D)
print(f"meshed footprint area: {meshedFootprint:.6f} mm^2, expected (own formulas): "
      f"{expectedFootprint:.6f} mm^2, relative diff: "
      f"{abs(meshedFootprint - expectedFootprint) / expectedFootprint:.2e}")

# (2) ground truth: does the *lattice construction itself* (independent of
#     this script's formulas) match create_dogbone()'s own cadquery lattice
#     area? This is the check that actually caught the Ly-vs-Wc_half /
#     k=n_x-half-vs-full-width bug -- (1) alone cannot, since it was
#     comparing the mesh against the same wrong formulas that built it.
trueFootprint = FILLET_FACE_AREA + TRUE_LATTICE_AREA
print(f"meshed footprint area: {meshedFootprint:.6f} mm^2, TRUE ground truth "
      f"(cadquery): {trueFootprint:.6f} mm^2, relative diff: "
      f"{abs(meshedFootprint - trueFootprint) / trueFootprint:.2e}")
if abs(meshedFootprint - trueFootprint) / trueFootprint > 1e-4:
    print("ERROR: meshed geometry does not match the true cadquery-computed "
          "lattice geometry.", file=sys.stderr)
    sys.exit(1)

totalVolume = meshedFootprint * T_HALF
expectedVolume = trueFootprint * T_HALF
print(f"meshed volume: {totalVolume:.6f} mm^3, expected: {expectedVolume:.6f} mm^3")

gmsh.write("lattice_mesh.msh")
gmsh.write("lattice_mesh.vtk")

# ---------------------------------------------------------- manual Z sweep ---
# Duplicate the 2D mesh at N_LAYERS+1 z-levels and connect corresponding
# quads between consecutive levels into hex8 elements (bottom face node
# order == top face node order, matching Marmot/gmsh's hex8 convention).
maxTag2D = max(nodeTags2D)
OFFSET = maxTag2D + 1
zLevels = [T_HALF * k / N_LAYERS for k in range(N_LAYERS + 1)]

nodeTags = []
coords = {}
for k, z in enumerate(zLevels):
    for tag in nodeTags2D:
        x, y = coords2D[tag]
        newTag = tag + k * OFFSET
        nodeTags.append(newTag)
        coords[newTag] = (x, y, z)

allHexNodes = []
for quad in quads2D:
    for k in range(N_LAYERS):
        bottom = [n + k * OFFSET for n in quad]
        top = [n + (k + 1) * OFFSET for n in quad]
        allHexNodes.append(bottom + top)
allHexTags = list(range(1, len(allHexNodes) + 1))

print(f"Total hex elements to export: {len(allHexTags)}")
print(f"Total nodes to export: {len(nodeTags)}")

TOL = 1e-6
xSymmetryNodes = [tag for tag, c in coords.items() if c[0] < TOL]
ySymmetryNodes = [tag for tag, c in coords.items() if c[1] < TOL]
zSymmetryNodes = [tag for tag, c in coords.items() if c[2] > T_HALF - TOL]
movingGripNodes = [tag for tag, c in coords.items() if c[0] > X_CUT - TOL]
print(f"xSymmetry nodes: {len(xSymmetryNodes)}, ySymmetry nodes: {len(ySymmetryNodes)}, "
      f"zSymmetry nodes: {len(zSymmetryNodes)}, movingGrip nodes: {len(movingGripNodes)}")


def write_nset(f, name, tags):
    f.write(f"\n*nSet, nSet={name}\n")
    for i in range(0, len(tags), 16):
        f.write(", ".join(str(int(n)) for n in tags[i:i + 16]) + "\n")


with open("lattice_mesh.inc", "w") as f:
    f.write("** Auto-generated by mesh_lattice.py -- all-hex mesh of the\n")
    f.write("** quarter-symmetry, half-thickness dogbone-lattice gauge octant\n")
    f.write("** (x>=0, y>=0, z<=T/2; see create_specimens.py and the module\n")
    f.write("** docstring in mesh_lattice.py for the symmetry argument), with\n")
    f.write("** the grip block truncated to a thin buffer past the fillet\n")
    f.write("** junction (movingGrip is the cut face at x=X_CUT, not the\n")
    f.write("** grip's true outer end).\n")
    f.write(f"** Lattice struts ({nTransfinite} of them) are forced structured\n")
    f.write("** (transfinite) quads, built as separate non-overlapping rectangles\n")
    f.write("** and glued to the fillet region via a conformal boolean fragment;\n")
    f.write("** the fillet region itself uses general recombination.\n")
    f.write(f"** Mesh size {MESH_SIZE}mm in-plane, {N_LAYERS} layers through the "
            f"{T_HALF}mm half-thickness.\n")
    f.write("*node\n")
    for tag in nodeTags:
        c = coords[tag]
        f.write(f"{tag}, {c[0]:.8g}, {c[1]:.8g}, {c[2]:.8g}\n")

    f.write("\n*element, type=C3D8UL\n")
    for elTag, elNodes in zip(allHexTags, allHexNodes):
        f.write(f"{elTag}, " + ", ".join(str(int(n)) for n in elNodes) + "\n")

    write_nset(f, "xSymmetry", xSymmetryNodes)
    write_nset(f, "ySymmetry", ySymmetryNodes)
    write_nset(f, "zSymmetry", zSymmetryNodes)
    write_nset(f, "movingGrip", movingGripNodes)

gmsh.finalize()
print("wrote lattice_mesh.inc")
