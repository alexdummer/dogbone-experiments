import cadquery as cq
import math
import pyvcad as pv
import pyvcad_rendering as viz
from pyvcad_compilers import DirectMaterialCompiler
import os

def create_dogbone(L: float = 115,
                        W: float = 25,
                        n_cells: int = 3,
                        Lc: float = 67,
                        L_grip: float = 11.0,
                        T: float = 2,
                        R1: float = 4.5,
                        R2: float = 8,
                        export_filename: str = "dogbone.stl",
                        lattice_cell_size: float = 5.0,
                        lattice_strut_width: float = 2.0):

    # ======================================================================
    #              Parametric Dogbone Specimen (Top-Down Profile)
    # ======================================================================
    #
    #    |<-- L_grip -->|                                   |<-- L_grip -->|
    #    +--------------+                                   +--------------+  ---
    #    |               \ R2                           R2 /               |   |
    #    |                \                               /                |   |
    #    |                 \ <-- Tangent                 /                 |   | W
    #    |               R1 \                           / R1               |   |
    #    |                   ---------------------------                   |   |
    #    |                                |                                |   |
    # CL + - - - - - - - - - - - - - - - -|- Wc- - - - - - - - - - - - - - +   |
    #    |                                |                                |   |
    #    |                   ---------------------------                   |   |
    #    |               R1 /                           \ R1               |   |
    #    |                 / <-- Tangent                 \                 |   |
    #    |                /                               \                |   |
    #    |               / R2                           R2 \               |   |
    #    +--------------+                                   +--------------+  ---
    #
    #                        |<---------- Lc --------->|
    #
    #    |<----------------------------- L ------------------------------->|
    #
    #    The gauge section (Wc x Lc, above) is filled with a square lattice
    #    instead of solid material: an axis-aligned grid of struts
    #    (lattice_cell_size, lattice_strut_width), no frame border. Wc is
    #    not an independent input - it is derived from the lattice itself
    #    (n_cells, lattice_cell_size, lattice_strut_width) so the gauge
    #    boundary always matches exactly where the struts naturally end,
    #    with no snapping/clipping needed and no target width to hit.
    #
    # --- 0. Derive Wc from the lattice cell layout ---
    # n_cells square cells span the full gauge width, plus one strut
    # thickness (the two boundary struts are each centered on the outermost
    # grid line, so their two halves sum to one full strut width). The
    # lattice is built as one mirrored half, so the row phasing depends on
    # parity: for even n_cells a strut sits centered on the y=0 centerline
    # (mirroring reunites its two halves into one strut); for odd n_cells a
    # cell (hole) straddles the centerline instead (mirroring reunites its
    # two halves into one cell), and the innermost strut sits half a cell
    # off-center. Both cases reduce to n_half = n_cells // 2 struts spaced
    # one cell apart in the half, phased by `offset`.
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1.")
    n_half = n_cells // 2
    row_offset = 0.0 if n_cells % 2 == 0 else 0.5
    Ly = (n_half + row_offset) * lattice_cell_size  # == n_cells * lattice_cell_size / 2
    Wc = n_cells * lattice_cell_size + lattice_strut_width

    # --- 1. Math & Geometry Calculations ---
    # Center of the first radius (C1)
    X1 = Lc / 2.0
    Y1 = (Wc / 2.0) + R1
    
    # Center of the second radius (C2)
    X2 = (L / 2.0) - L_grip
    Y2 = (W / 2.0) - R2
    
    # Distance between the two circle centers
    dx = X2 - X1
    dy = Y2 - Y1
    D = math.sqrt(dx**2 + dy**2)
    
    # Guard against impossible geometry (circles overlapping too much)
    if D < (R1 + R2):
        raise ValueError(
            "Transition section is too short. Increase L, decrease L_grip, or decrease radii."
        )
    
    # Calculate the angle of the tangent line connecting the two circles
    theta = math.atan2(dy, dx)
    gamma = math.acos(-(R1 + R2) / D)
    phi = theta + gamma
    
    # Normal vector perpendicular to the tangent line
    nx = math.cos(phi)
    ny = math.sin(phi)
    
    # Calculate the exact tangent points on the arcs
    P1 = (X1 - R1 * nx, Y1 - R1 * ny)
    P2 = (X2 + R2 * nx, Y2 + R2 * ny)
    
    
    # Helper function to find the midpoint of an arc (required by CadQuery's threePointArc)
    def get_arc_midpoint(C, P_start, P_end, radius):
        vx_mid = (P_start[0] - C[0]) + (P_end[0] - C[0])
        vy_mid = (P_start[1] - C[1]) + (P_end[1] - C[1])
        length = math.sqrt(vx_mid**2 + vy_mid**2)
        return (C[0] + (vx_mid / length) * radius, C[1] + (vy_mid / length) * radius)
    
    
    # Points where the arcs hit the horizontal flat sections
    A1 = (X1, Wc / 2.0)
    A2 = (X2, W / 2.0)
    
    # Calculate arc midpoints
    M1 = get_arc_midpoint((X1, Y1), A1, P1, R1)
    M2 = get_arc_midpoint((X2, Y2), P2, A2, R2)
    
    
    # --- 2. Build 3D Model ---
    # Trace the outer boundary of the top-right quadrant
    quadrant = (
        cq.Workplane("XY")
        .moveTo(0, 0)
        .lineTo(0, Wc / 2.0)  # Center to gauge top
        .lineTo(A1[0], A1[1])  # Flat gauge section
        .threePointArc(M1, P1)  # Arc 1 (R1)
        .lineTo(P2[0], P2[1])  # Straight tangent line
        .threePointArc(M2, A2)  # Arc 2 (R2)
        .lineTo(L / 2.0, W / 2.0)  # Flat grip section
        .lineTo(L / 2.0, 0)  # Outer edge down to centerline
        .close()  # Straight line back to origin
        .extrude(T)
    )

    # --- 2b. Replace the solid gauge block (x in [0, Lc/2], y in [0, Wc/2])
    #          with a square lattice (no frame border) ---
    Lc_half = Lc / 2.0
    Wc_half = Wc / 2.0

    Lx = Lc_half
    # Ly, n_half and row_offset were already fixed by the Wc-derivation step above

    # Cell count along Lc per quadrant-half; strut pitch recomputed to fit
    # exactly (Lc is specified directly, unlike Wc which is derived)
    n_x = max(1, round(Lx / lattice_cell_size))
    p_x = Lx / n_x
    p_y = lattice_cell_size

    # Struts are phased so one is centered on each centerline (x=0, y=0):
    # mirroring then recombines each half-strut into one continuous strut
    # with no coincident/duplicate faces. The outermost strut in each
    # direction is likewise centered on its grid line (y=Ly, x=Lx); Wc/Lx
    # were derived/chosen so that this is exactly where the gauge boundary
    # (Wc/2) and the fillet transition (Lc/2) sit, so there's no separate
    # clipping step needed.
    overlap = lattice_strut_width  # let struts fully penetrate each other at every junction

    gauge_sketch = cq.Sketch()
    v_locs = [(k * p_x, Ly / 2.0) for k in range(n_x + 1)]
    gauge_sketch = gauge_sketch.push(v_locs).rect(lattice_strut_width, Ly + overlap, mode="a")
    h_locs = [(Lx / 2.0, (i + row_offset) * p_y) for i in range(n_half + 1)]
    gauge_sketch = gauge_sketch.push(h_locs).rect(Lx + overlap, lattice_strut_width, mode="a")

    gauge_fill_quadrant = cq.Workplane("XY").placeSketch(gauge_sketch).extrude(T)
    gauge_box = cq.Workplane("XY").rect(Lc_half, Wc_half, centered=False).extrude(T)

    # fillet_grip_region: the envelope (fillet + grip) with the lattice gauge
    # box excised, but *before* the (overlap-padded, fully-fused) lattice is
    # unioned back in. Exposed separately so the FE mesh generator can build
    # its own non-overlapping, per-strut lattice geometry (for structured/
    # transfinite meshing) and glue it to this piece via a conformal
    # boolean fragment, instead of inheriting the overlap-padded, single-
    # fused-face lattice below (needed here for robust mirroring/printing,
    # but unsuitable for per-strut structured FE meshing).
    fillet_grip_region = quadrant.cut(gauge_box)
    quadrant = fillet_grip_region.union(gauge_fill_quadrant)

    # Mirror the quadrant to create the full symmetrical dogbone
    # Mirroring across XZ and YZ planes duplicates it into all 4 quadrants
    specimen = quadrant.mirror("XZ", union=True).mirror("YZ", union=True)

    # Export
    cq.exporters.export(
        specimen,
        "dogbone.stl",
        tolerance=1e-5,
        angularTolerance= 1e-1,
    )

    lattice_params = dict(
        Lx=Lx, Ly=Ly, n_x=n_x, n_half=n_half, row_offset=row_offset, p_x=p_x, p_y=p_y,
        strut_width=lattice_strut_width, Lc_half=Lc_half, Wc_half=Wc_half,
    )

    return specimen, quadrant, fillet_grip_region, lattice_params

def create_png_stack_from_mesh(
        name: str = "A50V50",
        mesh_filename: str = "dogbone.stl",
        x_cut: float = 46.5,
        agilus_frac_center: float = 0,
        ):
    materials = pv.default_materials
    red = materials.id("red")
    blue = materials.id("blue")

    # load the dogbone geometry as a mesh
    dogbone = pv.Mesh(
            "dogbone.stl", 
            pv.Vec3(0.0423, 0.0846, 0.027), 
            red)
    
    # Hard material cut, no grading: only the flat grip sections
    # (|x| >= x_cut, matching X2 = L/2 - L_grip in create_dogbone) are pure
    # rigid material (blue/vero); the transition (fillet) AND the lattice
    # gauge section (|x| < x_cut) are both the composite mix set by
    # agilus_frac_center, with an abrupt boundary at x_cut instead of a
    # graded ramp.
    # ----- vero -----|      composite (transition + gauge)     |----- vero -----  1
    #                                (agilus_frac_center)
    # ----- 0    -----|--------------------------------------- |----- 0    -----   0
    # ------------------------------------------------------------------------> x
    #            -x_cut                                       x_cut
    in_gauge = f"(x > {-x_cut}) * (x < {x_cut})"
    grading_str = [
        f"{agilus_frac_center} * {in_gauge}",
        f"1 - {agilus_frac_center} * {in_gauge}",
    ]

    root = pv.FGrade(grading_str, [red, blue], True)
    root.set_child(dogbone)

    font = "Consolas"
    font_aspect = pv.FontAspect.Bold
    horizontal_alignment = pv.HorizontalAlignment.Left # Left, Center, Right
    vertical_alignment = pv.VerticalAlignment.Top # Bottom, Center, Top
    text = pv.Text(name,
               2, 
               0.3, 
               materials.id('red'), 
               font_aspect, 
               font, 
               horizontal_alignment, 
               vertical_alignment)
    text = pv.Rotate(0, 0, 90, pv.Vec3(0,0,0), text)
    text = pv.Translate(-55, -12, 2, text)
    # text.set_child(fgrade)
    union = pv.Difference( root, text)
    # viz.Export(union, materials)
    # viz.Render(union, materials)
    DirectMaterialCompiler(union , pv.Vec3(42.3e-3,85.6e-3,27e-3), materials, name, "layer_", False, 0).compile()

def create_dma_specimen_stack(
        name: str = "A50V50",
        agilus_frac: float = 0.5,
        size: tuple = (25.0, 3.0, 1.0),
        output_dir: str = "../DMA_specimens",
        ):
    """Directly generate (no CadQuery/STL step) a uniform-composition
    rectangular bar for DMA time-temperature sweeps, and compile its
    material PNG stack. Unlike the dogbone, there is no spatial grading:
    the whole bar is one uniform red(agilus)/blue(vero) dither at
    `agilus_frac`, matching the same material convention used for the
    dogbone composite stacks. DMA specimens live in the sibling
    DMA_specimens/ directory, not alongside this script's own tension-
    dogbone output -- this default assumes create_specimens.py is run with
    cwd=dogbone-lattices/, same as the rest of this file's __main__ block.
    """
    materials = pv.default_materials
    red = materials.id("red")
    blue = materials.id("blue")

    Lx, Ly, Lz = size
    bar = pv.RectPrism(pv.Vec3(0, 0, 0), pv.Vec3(Lx, Ly, Lz), red)

    grading_str = [f"{agilus_frac}", f"1 - {agilus_frac}"]
    root = pv.FGrade(grading_str, [red, blue], True)
    root.set_child(bar)

    out_path = os.path.join(output_dir, name)
    os.makedirs(out_path, exist_ok=True)
    os.system(f"rm -rf {out_path}/*")
    DirectMaterialCompiler(
        root, pv.Vec3(42.3e-3, 85.6e-3, 27e-3), materials, out_path, "layer_", False, 0
    ).compile()

def create_cubic_lattice_bending_bar(a: float, cell_size: float = 5.0, hole_size: float = 2.5, material: int = None):
    """Builds the simple-cubic lattice bending specimen from Dummer &
    Regueiro (2026), Section 6.2.1 / Fig. 16: a cuboid with length-to-
    height ratio 4, exactly one unit cell thick. `a` sets the length
    (must be a whole multiple of cell_size); height = 4*a, thickness =
    cell_size (1 unit cell).

    Each 5mm unit cell (per the paper's text and Fig. 16) is a SOLID cube
    with a square through-channel cut in each of the 3 axis directions
    (hole_size = 2.5mm, centered), the three channels intersecting in a
    cross-shaped void at the cell center - not a 12-edge wireframe. Wall
    thickness = (cell_size - hole_size) / 2 = 1.25mm on the specimen's
    outer surface; since each unit cell is tiled as an independent solid
    with its own full wall on every face, the wall BETWEEN two adjacent
    cells is the sum of both cells' walls = 2.5mm (2x the outer wall).

    Built directly in openvcad via boolean Difference: one solid RectPrism
    spanning the whole specimen, minus a union of one capped (per-cell,
    not full-span) channel prism per cell per axis direction - each
    channel is capped by the neighboring cell's own wall rather than
    running continuously, which is what produces the doubled internal
    wall thickness.
    Axis convention (print orientation, rotated 90 degrees in the xy
    plane from the paper's own length/height labels): height (the long,
    4*a axis) in X -- the printer/build direction -- length (a) in Y,
    thickness in Z.
    """
    materials = pv.default_materials
    if material is None:
        material = materials.id("red")

    n = round(a / cell_size)
    if abs(n * cell_size - a) > 1e-6:
        raise ValueError(f"a ({a}) must be a whole multiple of cell_size ({cell_size}).")
    n_x = 4 * n  # height (long axis, length-to-height ratio of 4) -- aligned with X (printer direction)
    n_y = n      # length (short axis)
    n_z = 1      # one unit cell thick

    Lx, Ly, Lz = n_x * cell_size, n_y * cell_size, n_z * cell_size

    base = pv.RectPrism(pv.Vec3(Lx / 2.0, Ly / 2.0, Lz / 2.0), pv.Vec3(Lx, Ly, Lz), material)

    # Each channel is cut exactly cell_size long, so its end faces land
    # exactly flush with the base's outer faces (outermost cells) and with
    # the neighboring cell's channel end (interior cells) -- an exact
    # coincident-face boolean, which is floating-point-degenerate and
    # produced boundary-layer artifacts. Overshoot each channel's length by
    # boundary_eps (negligible next to cell_size/hole_size) so cut faces
    # always land strictly past the true boundary instead of exactly on it;
    # only the along-axis length is padded, not the hole_size cross-section,
    # since only that dimension ever touches a cell/specimen boundary face.
    boundary_eps = 1e-3
    channel_len = cell_size + boundary_eps

    channels = pv.Union()
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                cx = (i + 0.5) * cell_size
                cy = (j + 0.5) * cell_size
                cz = (k + 0.5) * cell_size
                channels.add_child(pv.RectPrism(pv.Vec3(cx, cy, cz), pv.Vec3(channel_len, hole_size, hole_size), material))
                channels.add_child(pv.RectPrism(pv.Vec3(cx, cy, cz), pv.Vec3(hole_size, channel_len, hole_size), material))
                channels.add_child(pv.RectPrism(pv.Vec3(cx, cy, cz), pv.Vec3(hole_size, hole_size, channel_len), material))

    lattice = pv.Difference(base, channels)
    lattice = pv.Translate(-Lx / 2.0, -Ly / 2.0, -Lz / 2.0, lattice)
    return lattice, (Lx, Ly, Lz)

def create_bending_specimen_stack(
        a: float,
        cell_size: float = 5.0,
        hole_size: float = 2.5,
        output_dir: str = "../bending-lattices/bending_lattice_specimens",
        agilus_frac: float = None,
        ):
    """Compiles the PNG material stack for one cubic-lattice bending
    specimen (see create_cubic_lattice_bending_bar) into
    `output_dir/a{a}mm/`. Bending specimens live in the sibling
    bending-lattices/ directory, not alongside this script's own tension-
    dogbone output -- this default assumes create_specimens.py is run with
    cwd=dogbone-lattices/, same as the rest of this file's __main__ block.

    `agilus_frac`: if given, the whole bar (strut material, not the void)
    is a uniform red(agilus)/blue(vero) composite dither at this fraction,
    same convention/no-spatial-grading approach as create_dma_specimen_stack.
    If None (default), the bar stays single-material (red), matching the
    original paper-replication stacks.
    """
    materials = pv.default_materials
    lattice, (Lx, Ly, Lz) = create_cubic_lattice_bending_bar(a, cell_size, hole_size)

    if agilus_frac is not None:
        red = materials.id("red")
        blue = materials.id("blue")
        grading_str = [f"{agilus_frac}", f"1 - {agilus_frac}"]
        root = pv.FGrade(grading_str, [red, blue], True)
        root.set_child(lattice)
        lattice = root

    name = f"a{a:g}mm"
    out_path = os.path.join(output_dir, name)
    os.makedirs(out_path, exist_ok=True)
    os.system(f"rm -rf {out_path}/*")
    DirectMaterialCompiler(
        lattice, pv.Vec3(42.3e-3, 85.6e-3, 27e-3), materials, out_path, "layer_", False, 0
    ).compile()
    return Lx, Ly, Lz

if __name__ == "__main__":

    composites = {
            "A0V100": 0,
            "A25V75": 0.25,
            "A50V50": 0.5,
            "A75V25": 0.75,
            "A100V0": 1,
    }
    
    create_dogbone()
    for name, agilus_frac in composites.items():
        print(f"Creating PNG stack for {name} with blue fraction {agilus_frac}")
        os.makedirs(f"{name}", exist_ok=True)
        os.system(f"rm -rf {name}/*")
        create_png_stack_from_mesh(name=name, agilus_frac_center=agilus_frac)

    # DMA time-temperature sweep specimens: uniform 25x3x1mm bars, one per
    # composite, generated directly in openvcad (no STL step).
    for name, agilus_frac in composites.items():
        print(f"Creating DMA specimen stack for {name} with agilus fraction {agilus_frac}")
        create_dma_specimen_stack(name=name, agilus_frac=agilus_frac)

    # Cubic-lattice bending specimens (Dummer & Regueiro 2026, Fig. 16):
    # a x 4a x 5mm wireframe lattice bars, generated directly in openvcad.
    for a in (5, 10, 20, 40):
        print(f"Creating bending lattice specimen stack for a={a}mm")
        dims = create_bending_specimen_stack(a=a)
        print(f"  -> dimensions (L x H x T): {dims} mm")

    # 2x-scaled unit cell (10mm cell / 5mm hole, same wall-thickness ratio
    # as the 5mm/2.5mm original) bending specimens, one per composite
    # material (uniform-composite dither, no spatial grading -- same
    # convention as the DMA bars above). a=5mm is skipped: it can't fit
    # even one 10mm cell. Written to a separate bending_lattice_specimens_2x_cell/
    # directory so the original single-material a5/a10/a20/a40mm stacks
    # above are untouched.
    for name, agilus_frac in composites.items():
        for a in (10, 20, 40):
            print(f"Creating 2x-cell bending lattice specimen stack for {name}, a={a}mm")
            dims = create_bending_specimen_stack(
                a=a, cell_size=10.0, hole_size=5.0,
                output_dir=f"../bending-lattices/bending_lattice_specimens_2x_cell/{name}",
                agilus_frac=agilus_frac,
            )
            print(f"  -> dimensions (L x H x T): {dims} mm")
