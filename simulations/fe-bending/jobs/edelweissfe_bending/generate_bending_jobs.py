"""Generate EdelweissFE job decks predicting 3-point bending tests on the
cubic-lattice bending specimens from Dummer & Regueiro (2026), Fig. 16
(see ../bending_lattice_specimens/ for the matching openvcad/PNG-stack
geometry of the same specimens).

Unlike that folder, no custom mesh script is used here: the mesh is built
natively by EdelweissFE's own CuboidLatticeGenerator (*modelGenerator,
generator=cuboidlatticegenerator) directly inside each .inp file, from the
same lattice parameters (5mm unit cell, 1.25mm strut / 2.5mm hole, one
cell thick, length-to-height ratio 4).

Material: the paper's own calibrated classical (non-micropolar) finite
strain viscoelastic model for the solid FLX-YK-S70 digital composite
(Table 3: K=10MPa, G=0.47MPa, 3-term Prony series) -- i.e. exactly the
material the paper itself used for its DNS on this same lattice
(Section 6.1/6.2), assigned directly to the strut geometry. This is a
genuine DNS of the lattice, not the paper's separate homogenized
micropolar continuum model (Table 4), which is calibrated FROM such DNS
results rather than used to produce them.

Loading: the paper's own bending case (Section 6.2.1) is a cantilever
(bottom fixed, top rotated). Ours is a true 3-point bend instead.

IMPORTANT axis note: the specimen's beam SPAN is Y (height = 4a) -- X
(=a) and Z (=5mm) are just the cross-section. So the supports (at the
two Y-ends) and the point load (at Y-midspan) all act in the TRANSVERSE
X-direction, not Y -- pushing in Y would be an axial test along the
beam's own length, not bending (an earlier version of this script made
exactly that mistake and was caught by inspecting the deformed shape:
a monotonic axial-type gradient end-to-end, not a bending shape peaked
at midspan).

Distributing the load/support singularities (rigid body constraints):
a single-node point load/support (an earlier version of this script)
concentrates the whole reaction at one mesh node -- a stress
singularity. Following the paper's own technique ("all nodes on the
[given] surface are constrained to a reference point" via a rigid body
constraint), each of the three BCs here instead ties a whole cross-
section's worth of nodes to one reference point (RP) with
*constraint, type=rigidbody (geometrically-exact, 3D, ties 3
translations + 3 rotations), and the Dirichlet BC is applied at the RP:
  - pin RP:    ties gen_bottom (the whole y=0 end face) -- displacement
    1=0,2=0,3=0 at the RP; rotation 1=0,2=0 (rotation component 3, the
    in-plane Z-axis bending rotation, stays free so that end can still
    pivot as the beam bends -- a true simple support, not a clamp).
  - roller RP: ties gen_top (the whole y=Ly end face) -- displacement
    1=0,2=0,3=0 at the RP too (see the DOF-readout caveat below for why
    this is fully pinned rather than a true roller free in Y/Z);
    rotation 1=0,2=0, component 3 free, same as the pin.
  - load RP:   ties `midspanCrossSection`, a custom node set built by
    *modelGenerator, generator=executePythonCode (there is no built-in
    generator set for an interior y=Ly/2 cross-section) selecting every
    node at y=Ly/2 -- displacement-controlled push in +X at the RP
    (2=0, 3=0 also fixed there), ramped to 0.1*a over 180s then held
    for 420s (paper's own bending ramp/hold shape, now applied as a
    midspan deflection); rotation 1=0,2=0 there too, component 3 free.

Rotation components 1 and 2 (about the global X and Y axes -- out-of-
plane twist and torsion, respectively) are fixed at all three RPs;
only component 3 (about Z, the in-plane bending rotation this problem
actually needs free) is left unconstrained anywhere. An earlier version
of this script left all 3 rotation components free at all three RPs --
under-constrained, per a correct call from the user: with 3 independent
RPs each free to rotate about all 3 axes, the out-of-plane modes (1, 2)
have no other source of resistance/removal in this X-Y bending problem,
and this materialized not as an outright solver error but as a subtle,
size-dependent near-mechanism -- a=5mm happened to still produce a
sensible result, but a=10mm converged to a state where the load RP
reached its full prescribed displacement while the rest of the mesh
barely followed (max structural displacement ~0.14mm vs. 1.0mm
prescribed; peak stress ~200x lower than a=5mm's), with the reaction-
force balance identity still holding perfectly throughout (a necessary
but, as this demonstrated, not remotely sufficient check). Fixing
rotations 1,2 at every RP resolved it.

CAVEAT -- all 3 translational DOFs are fixed at every RP, even where
the intended support is a "roller" (should be free in Y/Z) or the load
point (arguably free in Y/Z too): empirically, leaving any RP
translational DOF un-prescribed while a *constraint, type=rigidbody
ties a real node set to it corrupts this EdelweissFE version's `P`
(reaction force) readout to ~0 (machine noise) for EVERY DOF at that
RP and at its tied slave set -- not just the free one -- even though
the displacement solution itself is unaffected (converges normally,
correct prescribed deflection). Verified in isolation on both a plain
boxGen box and this lattice mesh, with 2 or 3 simultaneous constraints,
with/without a disk round-trip through *output, type=meshdatatofile:
explicitly setting all 3 components (even to 0) at every RP is what
fixes the readout. So the roller is modeled fully pinned (like the
pin end), sacrificing free axial sliding as the beam bends -- a minor
stiffening approximation for these modest deflections (~2.5% of span).
The load RP's 2=0,3=0 is not a real compromise: by the problem's own
Y/Z symmetry, the midspan cross-section wouldn't displace in Y or Z
anyway, so fixing them there just states the expected symmetric result
explicitly rather than constraining anything.

Each RP is an ORDINARY mesh node, found via *modelGenerator,
generator=findClosestNode at the geometric center of the tied cross-
section -- it is fine (and expected) for the RP to also be a member of
the node set it ties, matching the pattern in EdelweissFE's own
RigidBodyConstraintLargeDeformations3D testfile (its `rBottom`
reference point is mesh node 6, which is also a member of the `right`
node set that constraint ties to it). An EARLIER version of this script
instead added a brand-new, mesh-independent `*node` (following a
different testfile's, LinearizedRigidBodyConstraint3D's, pattern) at
each RP location -- that produced "Obtained NaN in linear solve" at the
very first (zero) increment: an isolated node with no element
connectivity apparently does not work as a rigidbody constraint's
reference point in this version of EdelweissFE, even though it is a
documented/working pattern for linearizedrigidbody. Using an ordinary,
element-connected mesh node as the RP fixed it.

Reaction force output: following the pattern used throughout
EdelweissFE's own rigid-body testfiles (e.g. GosfordSandstone,
CosseratHoekBrown), the resultant reaction force is read out by summing
`result=P` over the TIED SLAVE node set (gen_bottom / gen_top /
midspanCrossSection), not over the RP -- each slave node's P equals the
share of the constraint force transmitted from the RP, so the sum over
the whole face gives the correct total resultant.
"""
from pathlib import Path

HERE = Path(__file__).parent

# Paper's Table 3: calibrated classical viscoelastic parameters for the
# solid FLX-YK-S70 digital composite (K, G in MPa; 3-term Prony series).
K = 10.0
G = 0.47
PRONY = [(0.3, 5.0), (0.25, 50.0), (0.08, 250.0)]
RHO = 1.1e-9  # tonne/mm^3 (~1100 kg/m^3); unused by this quasi-static analysis

CELL_SIZE = 5.0
HOLE_SIZE = 2.5
STRUT_WIDTH = (CELL_SIZE - HOLE_SIZE) / 2.0  # 1.25mm, matches the paper
N_ELE_PER_CELL = 8         # matches the paper's 0.625mm DNS element size
N_ELE_STRUT = round(N_ELE_PER_CELL * STRUT_WIDTH / CELL_SIZE)  # 2

RAMP_TIME = 180.0
HOLD_TIME = 420.0
DEFLECTION_FRACTION = 0.1  # midspan deflection = this fraction of `a`

SIZES = (5, 10, 20, 40)


def render_job(a: float) -> str:
    n_x = round(a / CELL_SIZE)
    if abs(n_x * CELL_SIZE - a) > 1e-6:
        raise ValueError(f"a ({a}) must be a whole multiple of the {CELL_SIZE}mm unit cell.")
    n_y = 4 * n_x  # length-to-height ratio of 4, per the paper
    n_z = 1        # one unit cell thick, per the paper

    Lx = a
    Ly = 4 * a
    Lz = CELL_SIZE

    deflection = DEFLECTION_FRACTION * a
    name = f"a{a:g}mm"

    maxwell_lines = "\n".join(f"{g:.10g}, {t:.10g}," for g, t in PRONY)

    return f"""\
** 3-point-bending prediction for the cubic-lattice bending specimen
** (Dummer & Regueiro 2026, Fig. 16), size a={a:g}mm.
** Auto-generated by generate_bending_jobs.py -- do not hand-edit; regenerate instead.
**
** Mesh: generated natively by EdelweissFE's CuboidLatticeGenerator (no
** external mesh file/script) -- {n_x}x{n_y}x{n_z} unit cells of {CELL_SIZE}mm
** edge, {HOLE_SIZE}mm through-hole ({STRUT_WIDTH:g}mm outer wall, {2 * STRUT_WIDTH:g}mm
** internal wall between tiled cells -- see ../bending_lattice_specimens/
** for the matching openvcad geometry and its derivation from the paper).
** Axis convention (matches the openvcad specimens): length a in X,
** height 4a in Y, thickness (1 unit cell) in Z.
**
** Material: Marmot CompressibleFiniteStrainLinearViscoelasticity,
** NeoHookean base (baseModel=0: K, G only), the paper's own Table 3
** calibration for the solid FLX-YK-S70 composite (K={K:g}MPa, G={G:g}MPa,
** 3-term Prony series) -- this is the actual DNS base material, distinct
** from the paper's separate homogenized micropolar continuum model
** (Table 4), which is calibrated FROM such DNS rather than used here.
**
** Boundary conditions -- true 3-point bend, beam axis Y, pushed in X
** (paper's own bending case is a different load case: a cantilever,
** bottom fixed, top rotated), each distributed via a rigid body
** constraint tying a whole cross-section to a reference point (RP) --
** see the module docstring for the full reasoning:
**   - pinSupport RP    (found at Lx/2, 0, Lz/2): ties gen_bottom;
**     1=0, 2=0, 3=0 at the RP.
**   - rollerSupport RP (found at Lx/2, {Ly:g}, Lz/2): ties gen_top;
**     1=0 at the RP.
**   - loadNode RP       (found at Lx/2, {Ly / 2:g}, strut_width/2): ties
**     midspanCrossSection; displacement-controlled push in +X at the RP,
**     ramped to {deflection:g}mm ({DEFLECTION_FRACTION:g}*a) over {RAMP_TIME:g}s then held for
**     {HOLD_TIME:g}s.
**
** Output: loadForce/loadDisp (loadNode RP displacement in X; reaction
** force summed over the tied midspanCrossSection, for a load-deflection
** and load-time curve) plus the full displacement/stress fields
** (ensight).

*modelGenerator, generator=cuboidlatticegenerator, name=gen
lX={CELL_SIZE:g}
lY={CELL_SIZE:g}
lZ={CELL_SIZE:g}
nEleX={N_ELE_PER_CELL}
nEleY={N_ELE_PER_CELL}
nEleZ={N_ELE_PER_CELL}
nEleStrutX={N_ELE_STRUT}
nEleStrutY={N_ELE_STRUT}
nEleStrutZ={N_ELE_STRUT}
nX={n_x}
nY={n_y}
nZ={n_z}
elType=C3D8UL

*modelGenerator, generator=executePythonCode, name=gen2, executeAfterManualGeneration=True
** midspanCrossSection: every node at y=Ly/2 (no built-in generator set
** covers an interior cross-section; gen_bottom/gen_top only give the
** two y-end faces). This is the lattice's own hollow-frame cross
** section at that y (a through-channel runs the full height at the
** cell's own x,z center), not a filled square.
from edelweissfe.sets.nodeset import NodeSet
model.nodeSets['midspanCrossSection'] = NodeSet('midspanCrossSection', [n for n in model.nodes.values()
                                                 if abs(n.coordinates[1] - {Ly / 2:g}) <= 1e-6])

*modelGenerator, generator=findClosestNode, name=gen3
location='{Lx / 2:g}, 0, {Lz / 2:g}'
storeIn=pinSupport

*modelGenerator, generator=findClosestNode, name=gen4
location='{Lx / 2:g}, {Ly:g}, {Lz / 2:g}'
storeIn=rollerSupport

*modelGenerator, generator=findClosestNode, name=gen5
location='{Lx / 2:g}, {Ly / 2:g}, {STRUT_WIDTH / 2:g}'
storeIn=loadNode

*constraint, type=rigidbody, name=pinRB
nSet=gen_bottom
referencePoint=pinSupport

*constraint, type=rigidbody, name=rollerRB
nSet=gen_top
referencePoint=rollerSupport

*constraint, type=rigidbody, name=loadRB
nSet=midspanCrossSection
referencePoint=loadNode

*material, name=CompressibleFiniteStrainLinearViscoelasticity, id=flxykS70
** baseModel(0=NeoHooke), onlyShearCreep, K, G,
0, 1,
{K:g}, {G:g},
** nMaxwell, (gamma_i, tau_i) x 3
3,
{maxwell_lines}
** rho (unused by this quasi-static analysis; physically-reasonable placeholder)
{RHO:.3g}

*section, name=section1, material=flxykS70, type=solid
all

*job, name={name}_3pb, domain=3d
*solver, solver=NISTParallel, name=theSolver

*fieldOutput
>>perNode, name=displacement, elSet=all, field=displacement, result=U
>>perElement, name=stress, elSet=all, result=stress, quadraturePoint=0:8, f(x)='np.mean(x,axis=1)'
>>perNode, name=loadDisp, nSet=loadNode, field=displacement, result=U, f(x)='np.mean(x[:,0])', saveHistory=True, export=loadDisp
>>perNode, name=loadForce, nSet=midspanCrossSection, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=loadForce
>>perNode, name=supportForcePin, nSet=gen_bottom, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=supportForcePin
>>perNode, name=supportForceRoller, nSet=gen_top, field=displacement, result=P, f(x)='np.sum(x[:,0])', saveHistory=True, export=supportForceRoller

*output, type=ensight, name=ensightExport
>>perNode, fieldOutput=displacement
>>perElement, fieldOutput=stress
>>configuration, overwrite=yes

*output, type=monitor, name=mon
fieldOutput=loadForce

** NOTE: maxInc/minInc/startInc are FRACTIONS of stepLength (0..1), matching
** this project's existing edelweissfe/generate_jobs.py convention.

*step, solver=theSolver
startInc=0.05, maxInc=0.05, minInc=1e-4, maxNumInc=100000, maxIter=25, stepLength={RAMP_TIME:g}
>>options, category=NISTSolver, extrapolation=linear, linsolver=pardiso
>>dirichlet, name=pin, nSet=pinSupport, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=pinRot, nSet=pinSupport, field=rotation, 1=0, 2=0
>>dirichlet, name=roller, nSet=rollerSupport, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=rollerRot, nSet=rollerSupport, field=rotation, 1=0, 2=0
>>dirichlet, name=load, nSet=loadNode, field=displacement, 1={deflection:g}, 2=0, 3=0
>>dirichlet, name=loadRot, nSet=loadNode, field=rotation, 1=0, 2=0

*step, solver=theSolver
startInc=0.05, maxInc=0.05, minInc=1e-4, maxNumInc=100000, maxIter=25, stepLength={HOLD_TIME:g}
>>options, category=NISTSolver, extrapolation=linear, linsolver=pardiso
"""


def main():
    for a in SIZES:
        text = render_job(a)
        out_dir = HERE / f"a{a:g}mm"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"a{a:g}mm_3pb.inp"
        out_path.write_text(text)
        print(f"wrote {out_path.relative_to(HERE)}")


if __name__ == "__main__":
    main()
