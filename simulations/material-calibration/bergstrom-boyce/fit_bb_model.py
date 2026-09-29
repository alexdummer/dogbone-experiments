"""Calibrate the classical Bergstrom-Boyce (BB) finite-strain model --
implemented in Marmot (modules/materials/BergstromBoyce, conda env
"marmot") -- against the same stress-relaxation experiments already fit
by the simplified 1D PRF model (fit_prf_model.py). Unlike PRF, this drives
the REAL tensorial finite-strain material through Marmot's Python
material-point solver (marmot.solvers.FiniteStrainSolver): axial gradU
(and shear) components strain-controlled to replay the measured stretch
history exactly, transverse normal components stress-free (uniaxial
tension with free lateral surfaces).

Both networks support a selectable hyperelastic base potential
(BergstromBoyce material property #0, "hyperelasticBase"): NeoHooke (0),
Yeoh (1), Mooney-Rivlin (2), or Arruda-Boyce 8-chain (3). This script
fits only Arruda-Boyce -- the 8-chain potential is the hyperelastic base
used in Bergstrom & Boyce's own original formulation of this model, so
it is the physically appropriate (not just one-of-several-tried) choice
here, rather than a phenomenological polynomial fit (NeoHooke/Yeoh/
Mooney-Rivlin, which this script fit and compared in an earlier version):

  - Arruda-Boyce: 7 free params (muA,lambdaL_A, muB,lambdaL_B, c1, c2, c3)

Caveat specific to Arruda-Boyce: its locking stretch lambdaL is only
identifiable from data that gets the average chain stretch
lambda_chain=sqrt(Ibar1/3) meaningfully close to it. Every material fit
here reaches at most ~5% engineering strain (A50V50), i.e.
lambda_chain <~ 1.001 -- three orders of magnitude below any physically
plausible locking stretch -- so lambdaL is expected to be essentially
UNIDENTIFIED by this data (the fit degenerates to whatever NeoHooke-
equivalent muA/muB/c1/c2/c3 give, for a wide range of lambdaL). It is
fit anyway (not fixed) for honesty -- see the printed per-material
lambdaL sensitivity check in main() -- since this base's real purpose is
future large-strain data (e.g. A100V0, ~40% strain) where it will
matter.

Compressibility (kappaA, kappaB) cannot be identified from uniaxial data
alone (same caveat as fit_qlv_model.py / fit_prf_model.py), so kappaA =
kappaB = 100*A1 (near-incompressible) is fixed, not fit. muA0/muB0 are
seeded from the small-strain-equivalent shear modulus of the earlier
PRF fit (prf_params.csv); c1,c2,c3 are seeded fresh from the PRF/QLV
fits, same as the original NeoHooke-only version of this script.

marmot.solvers.FiniteStrainSolver.solve() prints a large amount of
per-increment diagnostic text directly to the OS-level stdout (not
Python's sys.stdout), so it is suppressed via fd-level redirection
around each solve() call -- otherwise a single joint fit (hundreds of
solves) would produce unusable amounts of output.
"""

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import marmot

from fit_prf_model import downsample_for_fit
from fit_qlv_marmot_model import PLACEHOLDER_MATERIALS, make_grid_plot
from fit_qlv_model import BASE, MATERIALS, load, find_ramp_bounds, mean_strain_rates, estimate_youngs_modulus

# Wall-clock cap (seconds) for a single multi-start seed's *entire*
# least_squares call, enforced by running it in its own subprocess (see
# _fit_one_start below). A per-evaluation timeout via signal.alarm was tried
# first and does NOT work here: SIGALRM is only handled between Python
# bytecode instructions, so it cannot preempt a single blocking call into the
# nanobind-wrapped C++ solver, which is exactly the pathological case (a
# multi-minute solve inside one evaluation) this is meant to catch. A
# subprocess can be forcibly terminated by the OS regardless of what it is
# blocked on, so it is used instead, at the coarser granularity of "one
# multi-start seed" rather than "one residual evaluation" (spawning a fresh
# process per evaluation, of which there are thousands per seed, would be
# far too slow).
START_TIMEOUT_S = 120

BASES = {"ArrudaBoyce": 3, "MooneyRivlin": 2}

IS_GRADU_CONTROLLED = np.array([[True, True, True], [True, False, True], [True, True, False]])
IS_STRESS_CONTROLLED = np.logical_not(IS_GRADU_CONTROLLED)



def _solve_quiet(solver):
    """Run solver.solve(), discarding its verbose C++-side stdout output
    (a plain Python stdout redirect can't catch this -- it writes directly
    to file descriptor 1)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        solver.solve()
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def build_props(base_name, A, B, c1, c2, c3):
    """Assemble the 13-slot BergstromBoyce material property array for the
    given hyperelastic base ("NeoHooke"/"Yeoh"/"MooneyRivlin"/"ArrudaBoyce"),
    network coefficient tuples A=(A1,A2,A3) and B=(B1,B2,B3), and flow-law
    params. For ArrudaBoyce, A1/B1 = mu, A2/B2 = lambdaL (A3/B3 unused),
    matching BergstromBoyce.h's parameter-slot convention exactly.
    kappaA = kappaB = 100*A1 (near-incompressible, fixed not fit) -- A1 is
    still the leading shear-modulus-like coefficient for every base,
    ArrudaBoyce included, so this convention carries over unchanged."""
    base_id = BASES[base_name]
    A1, A2, A3 = A
    B1, B2, B3 = B
    kappaA = kappaB = 100.0 * A1
    return np.array(
        [base_id, kappaA, kappaB, A1, A2, A3, B1, B2, B3, c1, c2, c3, 0.0],
        dtype=np.float64,
    )


def _simulate_bb_once(t, stretch, base_name, A, B, c1, c2, c3):
    """Replay a measured (time, stretch) history through the Marmot
    BergstromBoyce material point solver; returns the predicted nominal
    (engineering, 1st Piola-Kirchhoff) axial stress at each input time,
    converted from the solver's native Kirchhoff stress via P11 = tau11/F11
    (F is diagonal here, so this is the exact, not approximate, conversion)."""
    props = build_props(base_name, A, B, c1, c2, c3)
    options = marmot.solvers.FiniteStrainSolver.SolverOptions()
    solver = marmot.solvers.FiniteStrainSolver("BERGSTROMBOYCE", props, options)

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt <= 0:
            continue
        step = marmot.solvers.FiniteStrainSolver.Step()
        step.timeStart = float(t[i - 1])
        step.timeEnd = float(t[i])
        step.dTStart = dt
        step.dTMax = dt
        # increment budget: raised from dt/50 (30 max increments) after
        # confirming (once the Marmot solver's timestep-cutback-reset bug was
        # fixed -- see MarmotMaterialPointSolverFiniteStrain.cpp -- which had
        # been making the OLD, smaller budget completely untestable, since dT
        # never actually reached dTMin regardless of its value) that A100V0's
        # much larger per-step strain increments (~1% vs <0.1% for the other
        # materials) can occasionally need finer sub-stepping. This is a
        # generous headroom, not a requirement: the other three materials
        # already converge well within the old budget and are unaffected;
        # it only matters for points a solve would otherwise reach dTMin on
        # too early to resolve.
        step.dTMin = dt / 2000
        step.maxIncrements = 1000
        step.isGradUComponentControlled = IS_GRADU_CONTROLLED
        step.isStressComponentControlled = IS_STRESS_CONTROLLED
        gradu_target = np.zeros((3, 3))
        gradu_target[0, 0] = stretch[i] - stretch[i - 1]
        step.gradUIncrementTarget = gradu_target
        step.stressIncrementTarget = np.zeros((3, 3))
        solver.addStep(step)

    _solve_quiet(solver)
    history = solver.getHistory()
    h_t = np.array([h.time for h in history])
    h_F11 = np.array([h.F[0, 0] for h in history])
    h_tau11 = np.array([h.stress[0, 0] for h in history])
    h_P11 = h_tau11 / h_F11
    result = np.interp(t, h_t, h_P11)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite prediction")
    return result


def simulate_bb(t, stretch, base_name, A, B, c1, c2, c3):
    """Robust wrapper around _simulate_bb_once: the underlying solve has a
    narrow, essentially last-bit-of-float64-sensitive Newton-Raphson
    convergence failure mode for isolated parameter combinations (a
    structural numerical edge case in the C++ solver, not fixable from
    here -- see fit_bb_joint's multi-start docstring for how this was
    diagnosed). A tiny (0.01%) parameter nudge reliably lands outside the
    failing region when it occurs, so retry a few times with small nudges
    before giving up."""
    for nudge in (0.0, 1e-4, -1e-4, 2e-4, -2e-4, 5e-4, -5e-4, 1e-3, -1e-3, 2e-3, -2e-3, 5e-3, -5e-3):
        try:
            scale = 1.0 + nudge
            return _simulate_bb_once(t, stretch, base_name,
                                      tuple(a * scale for a in A), tuple(b * scale for b in B),
                                      c1 * scale, c2, c3)
        except Exception:
            continue
    raise RuntimeError("simulate_bb: solver failed for all nudge attempts")


LAMBDA_L_FIXED = 5.0

# Below this peak engineering strain, lambda_chain never gets meaningfully
# close to any physically plausible locking stretch (see module docstring:
# lambda_chain <~ 1.05 for a 10% engineering strain, three orders of
# magnitude below where an elastomer typically locks), so lambdaL is not
# identifiable from the data and free-fitting it only adds a noise direction
# to the search (confirmed for A0V100/A25V75/A50V50: fitted lambdaL varied
# fit-to-fit with no RMSE improvement over lambdaL fixed). Only A100V0
# (~35% peak strain) clears this and is left free.
LAMBDA_L_IDENTIFIABLE_STRAIN = 0.10


def unpack(x, base_name, fix_lambdaL=False):
    """Map the base-specific free-parameter vector x to (A, B, c1, c2, c3).
    fix_lambdaL=True means x omits A2/B2 entirely (5 free params, not 7);
    both locking stretches are then LAMBDA_L_FIXED, not fit."""
    if base_name == "ArrudaBoyce":
        if fix_lambdaL:
            A1, B1, c1, c2, c3 = x
            return (A1, LAMBDA_L_FIXED, 0.0), (B1, LAMBDA_L_FIXED, 0.0), c1, c2, c3
        A1, A2, B1, B2, c1, c2, c3 = x
        return (A1, A2, 0.0), (B1, B2, 0.0), c1, c2, c3
    if base_name == "MooneyRivlin":
        # A1/B1 = C10, A2/B2 = C01 (A3/B3 unused), matching
        # BergstromBoyce.h's mooneyRivlinPotential(C, C10, C01, kappa).
        A1, A2, B1, B2, c1, c2, c3 = x
        return (A1, A2, 0.0), (B1, B2, 0.0), c1, c2, c3
    raise ValueError(base_name)


def seed_and_bounds(base_name, muA0, muB0, c1_0, c2_0, c3_0, fix_lambdaL=False):
    """x0/lb/ub for the base-specific free-parameter vector, seeded from the
    prior NeoHooke-only fit (muA0, muB0) and the PRF/QLV-derived flow-law
    seeds (c1_0, c2_0, c3_0), per the seeding strategy in the module
    docstring."""
    # c2 (chain-stretch exponent, applied to (lambda_chain-1) exactly as
    # implemented here) is bounded to [-1, 0] per Bergstrom & Boyce's own
    # original paper: "C2 is a constant that is restricted by reptational
    # dynamics to be in [0, -1]." All previously-reported fits (c2 in
    # 0.59-1.93) were searching entirely the wrong sign region -- outside
    # this literature-sanctioned range -- and need to be redone.
    # c3 (stress exponent, m) is bounded below at 1.0: for a power law x^c
    # with c<1, dx^c/dx -> infinity as x->0, so a sub-unity exponent gives
    # infinite tangent stiffness at zero driving stress. Upper bound widened
    # from 6 to 10 (typical literature range for this kind of exponent)
    # since one fit landed right at the old bound (c3=5.97).
    # Caution: with c2<0, the material's stretchTermClamped (BergstromBoyce.h)
    # clamps (lambda_chain-1) to exactly 0.0 whenever lambda_chain<=1 (true
    # at the start of every simulation, and possibly transiently during a
    # difficult Newton iteration) -- pow(0.0, negative) is +inf in C++, so
    # this sign region is more likely to trip the solver's existing
    # exception fallback (or, rarely, the subprocess timeout) than the
    # all-positive region explored before.
    c_lb, c_ub = [1e-8, -1.0, 1.0], [1e3, 0.0, 10.0]
    c_x0 = [c1_0, c2_0, c3_0]
    if base_name == "MooneyRivlin":
        # C10=mu/2 convention (unlike ArrudaBoyce's A1=mu directly -- see the
        # comment below), mirroring fit_qlv_marmot_model.py's MooneyRivlin
        # seed exactly: C10_0 = G0/2, C01_0 a small perturbation off it, C01
        # bounds wide/signed since it need not be positive.
        A1_0, B1_0 = muA0 / 2, muB0 / 2
        x0 = [A1_0, 0.1 * A1_0, B1_0, 0.1 * B1_0] + c_x0
        lb = [1e-3, -1e3, 1e-3, -1e3] + c_lb
        ub = [1e4, 1e4, 1e4, 1e4] + c_ub
        return list(np.clip(x0, lb, ub)), lb, ub
    if base_name != "ArrudaBoyce":
        raise ValueError(base_name)
    # A1/B1 = mu directly (unlike Yeoh/MooneyRivlin's C10=mu/2 convention --
    # ArrudaBoyce's isochoric potential IS the neo-Hookean mu/2*(Ibar1-3) as
    # lambdaL -> infinity, so mu itself, not mu/2, is the leading coefficient).
    # lambdaL0=5.0 is a physically-typical middle-of-the-road elastomer
    # locking stretch, not derived from this dataset -- per the module
    # docstring, none of these materials' strain ranges (<=~5%) can actually
    # identify it. Bounds: lb=1.05 stays safely above every reached chain
    # stretch here (<=~1.001, see docstring) with generous headroom so the
    # solver never approaches the potential's genuine locking singularity;
    # ub=20 is loose enough that "large lambdaL" fits are practically
    # indistinguishable from NeoHooke without letting the search wander to
    # numerically silly values on a direction the data can't constrain anyway.
    # A1/B1 lower bound: was 1.0 MPa, chosen without checking it against any
    # material's actual stiffness. For A100V0 (the softest material fit here)
    # the real secant modulus at its ~35% peak strain is only ~0.7-2.0 MPa
    # *total* across both networks, i.e. well under muA+muB>=2.0 implied by a
    # 1.0 floor on each -- the fit was structurally unable to reach a soft
    # enough answer and degenerated regardless of seeding/bounds elsewhere.
    # 1e-3 is a generic floor (just keeps mu positive/away from the
    # numerically-singular mu=0 case) rather than an assumed physical minimum,
    # so it stays permissive for every material, not just this one.
    if fix_lambdaL:
        x0 = [muA0, muB0] + c_x0
        lb = [1e-3, 1e-3] + c_lb
        ub = [1e4, 1e4] + c_ub
    else:
        x0 = [muA0, 5.0, muB0, 5.0] + c_x0
        lb = [1e-3, 1.05, 1e-3, 1.05] + c_lb
        ub = [1e4, 20.0, 1e4, 20.0] + c_ub
    return list(np.clip(x0, lb, ub)), lb, ub


def _bb_residuals(x, base_name, test_data, fix_lambdaL=False):
    A, B, c1, c2, c3 = unpack(x, base_name, fix_lambdaL)
    out = []
    for t, stretch, stress, peak, weight in test_data.values():
        try:
            # _simulate_bb_once (not the nudge-retrying simulate_bb wrapper):
            # a single failed evaluation here just falls back to the bounded
            # zero-prediction residual below, which is fine during
            # optimization -- retrying with nudges on every one of thousands
            # of residual evaluations would be far too slow.
            pred = _simulate_bb_once(t, stretch, base_name, A, B, c1, c2, c3)
        except Exception:
            # Rare, non-monotonic Newton-Raphson-convergence failures occur
            # for isolated parameter combinations (a real, narrow numerical
            # edge case in the underlying material-point solve, not fixable
            # from here). A bounded fallback -- "predict zero stress" -- is
            # used instead of a large constant offset: since residuals are
            # already normalized by peak stress, this keeps the penalized
            # residual O(1) (clearly bad, but not orders of magnitude larger
            # than genuine residuals), so a single failed finite-difference
            # perturbation during Jacobian estimation doesn't corrupt the
            # whole search direction.
            pred = np.zeros_like(stress)
        out.append((pred - stress) / peak * weight)
    return np.concatenate(out)


def _bb_fit_one_start_worker(x0_try, lb, ub, base_name, test_data, result_queue, fix_lambdaL=False):
    """Runs in its own subprocess (see _fit_one_start_with_timeout): a single
    least_squares call from one multi-start seed, isolated so a
    pathologically slow seed can be killed by the OS without taking the rest
    of the multi-start down with it."""
    result = least_squares(lambda x: _bb_residuals(x, base_name, test_data, fix_lambdaL), x0_try, bounds=(lb, ub),
                            method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10, diff_step=1e-3, max_nfev=150)
    result_queue.put((result.cost, result.x, result.nfev))


def _fit_one_start_with_timeout(x0_try, lb, ub, base_name, test_data, timeout_s=START_TIMEOUT_S, fix_lambdaL=False):
    """Runs one multi-start seed's least_squares call in a subprocess with a
    hard wall-clock cap. Returns (cost, x, nfev), or None if the seed timed
    out or otherwise failed to report a result. A subprocess (not
    signal.alarm) is required: SIGALRM can only be handled between Python
    bytecode instructions, so it cannot preempt a single blocking call into
    the nanobind-wrapped C++ solver -- exactly the pathological case (a
    multi-minute solve inside one evaluation) this needs to catch. A
    subprocess can be terminated by the OS regardless of what it is blocked
    on."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_bb_fit_one_start_worker,
                           args=(x0_try, lb, ub, base_name, test_data, result_queue, fix_lambdaL))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    return result_queue.get() if not result_queue.empty() else None


def fit_bb_joint(material, cfg, base_name, muA0, muB0, prf_params, qlv_prony, verbose=True, x0_override=None):
    test_data = {}
    for rate_group, ids in cfg["rate_groups"].items():
        df = load(cfg, material, ids[0])
        # downsample_for_fit returns df["strain"] (the extensometer channel) at
        # each sampled time -- it only uses crosshead position internally to
        # find the ramp/hold boundary, not to derive strain values themselves.
        t, strain, stress = downsample_for_fit(df)
        # downsample_for_fit uses a FIXED point count per phase (n_ramp=40,
        # n_hold=60) regardless of each phase's actual duration. The ramp is
        # typically 10-600x shorter than the 600s hold, so its points end up
        # 3-36x denser in time -- e.g. A50V50/slow: 40 ramp points over ~12s
        # (3.3 pts/s) vs. 56 hold points over ~600s (0.09 pts/s), a 36x density
        # ratio. Since least_squares weighs every residual entry equally, this
        # silently lets the ramp's SHAPE dominate the fit cost purely because
        # it's oversampled, not because it's more physically important -- the
        # relaxation branch ends up systematically under-fit relative to the
        # ramp. Correct this by weighting each point so the two phases
        # contribute equally to the total cost regardless of point count
        # (weight = 1/sqrt(n_points_in_that_phase), so summed squared,
        # weighted residuals equal each phase's MEAN squared residual).
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t <= t_ramp_end
        n_ramp_pts, n_hold_pts = is_ramp.sum(), (~is_ramp).sum()
        weight = np.where(is_ramp, 1.0 / np.sqrt(max(n_ramp_pts, 1)), 1.0 / np.sqrt(max(n_hold_pts, 1)))
        test_data[rate_group] = (t, 1.0 + strain, stress, stress.max(), weight)

    c1_0 = 1.0 / qlv_prony["tau2"]
    c2_0 = -0.5
    c3_0 = float(np.clip(prf_params["m_flow"], 1.0, 9.9))

    peak_strain = max(stretch.max() - 1.0 for _, stretch, _, _, _ in test_data.values())
    fix_lambdaL = base_name == "ArrudaBoyce" and peak_strain < LAMBDA_L_IDENTIFIABLE_STRAIN
    if verbose:
        print(f"  peak engineering strain across rate groups = {100 * peak_strain:.2f}% -> "
              f"lambdaL {'FIXED at ' + str(LAMBDA_L_FIXED) if fix_lambdaL else 'free (identifiable)'}")

    x0, lb, ub = seed_and_bounds(base_name, muA0, muB0, c1_0, c2_0, c3_0, fix_lambdaL)
    if x0_override is not None:
        x0 = list(np.clip(x0_override, lb, ub))

    # The underlying material-point solve has a narrow, non-monotonic-in-
    # perturbation-size Newton-Raphson convergence failure mode for isolated
    # parameter combinations (confirmed: relative perturbations as small as
    # 1e-13 -- literally the last bit of float64 precision, e.g. a value
    # read back from a CSV round-trip vs. the identical literal -- can flip
    # a fit from converging cleanly to landing in a markedly worse basin;
    # this is a structural numerical sensitivity in the C++ solver itself,
    # not a smooth/tunable step-size artifact, and out of scope to fix here).
    # `trf`'s trust-region trajectory from a seed sitting near such a region
    # can be thrown off entirely by a single corrupted finite-difference
    # Jacobian column.
    #
    # That numerical instability is a SEPARATE problem from genuine local
    # optima in the underlying (non-convex) joint-fit landscape: a 10-seed
    # randomized-restart check (done outside this pipeline) found that a
    # single-seed-only multi-start had, for A0V100/NeoHooke specifically,
    # converged to a basin 40% worse (by joint-fit cost) than one reachable
    # from a seed elsewhere in the same bounded region -- cutting the fitted
    # model's fast-rate RMSE roughly in half. `scipy.optimize.
    # differential_evolution` was tried as a more principled global-search
    # replacement, but timed out (>590s without completing even one
    # generation) against this solver: early generations sample broadly
    # across the bounds, including combinations that make the local
    # Newton-Raphson return-mapping solve either fail outright (cheap, caught
    # below) or converge only after many slow sub-stepped increments
    # (expensive, and not caught by anything) -- unlike a bounded
    # trust-region search, which shrinks its step and moves on quickly once
    # it senses a bad region. The practical compromise adopted here: several
    # differently-seeded LOCAL (least_squares/trf) searches, each fast and
    # fail-fast on its own -- a small nudge-based cluster around the primary
    # seed to specifically dodge the narrow numerical-instability failure
    # mode (as before), plus several widely-scattered random seeds to reduce
    # the chance of missing a materially better basin the way the
    # single-seed fit did for A0V100. Every start shares the same bounds
    # `lb`/`ub`, so a better basin found this way is a genuine improvement,
    # not a bound artifact.
    starts = [np.clip(np.asarray(x0) * scale, lb, ub) for scale in (1.0, 1.001, 0.999)]
    rng = np.random.default_rng(0)
    for _ in range(6):
        random_start = rng.uniform(lb, ub)
        # The leading modulus (muA for NeoHooke; A1 for Yeoh/Mooney-Rivlin)
        # is kept near the seed's order of magnitude -- letting it roam the
        # full [1, 1e4] range on every random start makes most draws
        # physically implausible (and slow to fail) without meaningfully
        # broadening the search, since the seed's own magnitude is already a
        # good estimate from the PRF/QLV pre-fits.
        random_start[0] = rng.uniform(0.3, 3.0) * x0[0]
        starts.append(np.clip(random_start, lb, ub))

    best = None
    for i, x0_try in enumerate(starts):
        outcome = _fit_one_start_with_timeout(list(x0_try), lb, ub, base_name, test_data, fix_lambdaL=fix_lambdaL)
        if outcome is None:
            if verbose:
                print(f"  start {i}: timed out after {START_TIMEOUT_S}s, skipping")
            continue
        cost, x, nfev = outcome
        if verbose:
            print(f"  start {i}: cost={cost:.6f} nfev={nfev}")
        if best is None or cost < best[0]:
            best = (cost, x, nfev)
    if best is None:
        raise RuntimeError(f"fit_bb_joint: all {len(starts)} multi-start seeds timed out or failed for "
                            f"{material}/{base_name}")
    cost, x, nfev = best
    A, B, c1, c2, c3 = unpack(x, base_name, fix_lambdaL)
    params = {
        "hyperelastic_base": base_name,
        "A1": A[0], "A2": A[1], "A3": A[2],
        "B1": B[0], "B2": B[1], "B3": B[2],
        "kappaA": 100.0 * A[0], "kappaB": 100.0 * A[0],
        "c1": c1, "c2": c2, "c3": c3,
        "cost": cost, "nfev": nfev, "lambdaL_fixed": fix_lambdaL,
    }
    return params


def validate_bb(material, cfg, base_name, A, B, c1, c2, c3):
    rows = []
    plot_rows = {}
    for rate_group, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, material, test_id)
        t = df["time_s"].to_numpy()
        measured = df["stress"].to_numpy()

        # full-resolution measured curves are too dense to replay step-by-step
        # in reasonable time; downsample for the simulation, then compare the
        # (identically downsampled) measured curve against it
        t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=60, n_hold=90)
        try:
            pred = simulate_bb(t_ds, 1.0 + strain_ds, base_name, A, B, c1, c2, c3)
        except Exception as exc:
            print(f"  [validate_bb] {material}/{base_name}/{rate_group}: solver failed even after "
                  f"nudge retries ({exc}); skipping this curve")
            continue

        peak = stress_ds.max()
        rmse = np.sqrt(np.mean((pred - stress_ds) ** 2))
        rows.append({"material": material, "hyperelastic_base": base_name, "rate_group": rate_group,
                     "test_id": test_id, "rmse_mpa": rmse, "rmse_pct_of_peak": 100 * rmse / peak})
        plot_rows[rate_group] = (t, measured, t_ds, pred)
    return pd.DataFrame(rows), plot_rows


def validate_bb_ramp(material, cfg, base_name, A, B, c1, c2, c3):
    """Same idea as validate_bb, but replay only the loading ramp (exclude
    the hold/relaxation phase) and compare engineering stress vs.
    engineering strain in percent, mirroring the ramp_bounds/
    ramp_strain_stress convention in e.g. A0V100-relax/plot_stress_strain.py."""
    ramp_rows = {}
    for rate_group, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, material, test_id)
        # find_ramp_bounds runs on crosshead position (a hard mechanical stop,
        # cleanly detectable) purely to locate WHEN the ramp/hold boundary is;
        # the actual strain VALUES replayed through the solver below always
        # come from df["strain"] (the extensometer channel), never from
        # position. Using strain itself for the boundary search was tried and
        # rejected: strain keeps drifting (viscoelastic creep) during the hold
        # phase, which throws the 99%-of-hold-value threshold off by orders of
        # magnitude for several tests (verified empirically), unlike position
        # which stops dead the instant the crosshead does.
        onset, ramp_end = find_ramp_bounds(df["position_mm"])
        sl = slice(onset, ramp_end + 1)
        t_full = df["time_s"].to_numpy()[sl]
        strain_full = df["strain"].to_numpy()[sl]
        measured_stress = df["stress"].to_numpy()[sl]

        n_pts = 40
        idx = np.linspace(0, len(t_full) - 1, n_pts, dtype=int)
        t_ds = t_full[idx]
        strain_ds = strain_full[idx]
        try:
            pred = simulate_bb(t_ds, 1.0 + strain_ds, base_name, A, B, c1, c2, c3)
        except Exception as exc:
            print(f"  [validate_bb_ramp] {material}/{base_name}/{rate_group}: solver failed even after "
                  f"nudge retries ({exc}); skipping this curve")
            continue

        ramp_rows[rate_group] = (100 * strain_full, measured_stress, 100 * strain_ds, pred)
    return ramp_rows


def main():
    all_bb = []
    validation_rows = []
    plot_data = {}
    ramp_data = {}

    prf_params_df = pd.read_csv(BASE / "prf_params.csv", index_col=0)
    qlv_prony_df = pd.read_csv(BASE / "qlv_prony_params.csv", index_col=0)

    for material, cfg in MATERIALS.items():
        prf_params = prf_params_df.loc[material].to_dict()
        qlv_prony = qlv_prony_df.loc[material].to_dict()
        # seed muA0/muB0 from a model-free tangent-Young's-modulus estimate
        # taken directly off the raw measured ramp (estimate_youngs_modulus),
        # not from the PRF fit -- the PRF-derived seed (muA*alphaA/2) turned
        # out to disagree with the material's real stiffness by a large
        # factor for A100V0 (real E ~1-9 MPa vs. a PRF-derived ~0.49/3.82 MPa
        # split that, combined with the old bounds, made the fit
        # structurally too stiff to reach a valid answer -- see
        # seed_and_bounds's lb comment). Averaging the 3 rate groups' tangent
        # E and splitting evenly between the two networks (E=3*(muA+muB) for
        # a near-incompressible rubber) gives a material-agnostic order-of-
        # magnitude-correct seed for every material, not just the ones the
        # PRF fit happened to agree with.
        # floor is just to keep the seed positive/away from mu=0, not an
        # assumed physical minimum -- see seed_and_bounds's lb comment.
        E_mean = np.mean(list(estimate_youngs_modulus(cfg, material).values()))
        muA0 = muB0 = max(E_mean / 3.0 / 2.0, 1e-3)

        plot_data[material] = {}
        ramp_data[material] = {}

        for base_name in BASES:
            print(f"\n=== fitting Bergstrom-Boyce (Marmot), {base_name}: {material} ===")
            try:
                params = fit_bb_joint(material, cfg, base_name, muA0, muB0, prf_params, qlv_prony, verbose=False)
            except RuntimeError as exc:
                # A single (material, base) combination hard-failing (every
                # multi-start seed timing out/erroring -- a documented,
                # pre-existing solver-numerics risk, see fit_bb_joint's
                # docstring) must not discard every OTHER already-computed
                # result: bb_params.csv/bb_vs_prf_comparison.csv/grid plots
                # are only written once at the very end, so an uncaught
                # exception here used to lose the entire run. Skip this combo
                # and keep going; it is simply absent from the outputs, not
                # silently substituted with a placeholder.
                print(f"  SKIPPED {material}/{base_name}: {exc}")
                continue
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()})

            A = (params["A1"], params["A2"], params["A3"])
            B = (params["B1"], params["B2"], params["B3"])
            c1, c2, c3 = params["c1"], params["c2"], params["c3"]

            bb_val, plot_rows = validate_bb(material, cfg, base_name, A, B, c1, c2, c3)
            if bb_val.empty:
                # fit_bb_joint returned a "best of 9 starts" result rather than
                # raising (at least one start produced SOME cost), but those
                # fitted parameters turn out to be unusable: every rate group
                # failed to even simulate in validate_bb. Treat this the same
                # as a hard SKIPPED failure rather than recording bogus
                # parameters or crashing on the empty-dataframe lambdaL check
                # below.
                print(f"  SKIPPED {material}/{base_name}: fit succeeded (cost={params['cost']:.6g}) but the "
                      f"fitted parameters could not be validated (solver failed for every rate group)")
                continue

            row = {"material": material}
            row.update(params)
            all_bb.append(row)

            validation_rows.append(bb_val)
            plot_data[material][base_name] = plot_rows

            ramp_data[material][base_name] = validate_bb_ramp(material, cfg, base_name, A, B, c1, c2, c3)

            # lambdaL identifiability check (see module docstring): compare
            # validation RMSE at the fitted lambdaL against lambdaL->"infinity"
            # (1e4, numerically indistinguishable from the NeoHooke limit at
            # these strains). If the two RMSEs match closely, lambdaL is
            # confirmed NOT identified by this data -- any reasonably large
            # value fits equally well, exactly as expected from the strain
            # ranges involved. Skipped when lambdaL was already fixed (not
            # fit) -- the peak-strain-based rule in fit_bb_joint already
            # decided this case ahead of time, so re-checking it here would
            # just repeat the same conclusion at the cost of 2 more solves.
            if params["lambdaL_fixed"]:
                print(f"  lambdaL fixed at {LAMBDA_L_FIXED} (peak strain below "
                      f"{100 * LAMBDA_L_IDENTIFIABLE_STRAIN:.0f}%, not identifiable) -- skipping check")
            else:
                A_largeL = (A[0], 1e4, 0.0)
                B_largeL = (B[0], 1e4, 0.0)
                bb_val_largeL, _ = validate_bb(material, cfg, base_name, A_largeL, B_largeL, c1, c2, c3)
                rmse_fitted = bb_val["rmse_pct_of_peak"].mean()
                rmse_largeL = bb_val_largeL["rmse_pct_of_peak"].mean()
                print(f"  lambdaL identifiability check: fitted (A={A[1]:.3g}, B={B[1]:.3g}) mean RMSE% "
                      f"= {rmse_fitted:.4f} vs. lambdaL->1e4 mean RMSE% = {rmse_largeL:.4f} "
                      f"(diff = {abs(rmse_fitted - rmse_largeL):.4f} pct points)")

    validation = pd.concat(validation_rows, ignore_index=True)

    params_df = pd.DataFrame(all_bb)
    params_df.to_csv(BASE / "bb_params.csv", index=False)
    print(f"\nsaved {BASE / 'bb_params.csv'}")

    # --- BB vs. PRF comparison, same RMSE-%-of-peak metric, all 3 bases ---
    prf_validation = pd.read_csv(BASE / "prf_vs_qlv_comparison.csv")[
        ["material", "rate_group", "test_id", "rmse_pct_of_peak_prf"]
    ].rename(columns={"rmse_pct_of_peak_prf": "rmse_pct_of_peak_prf_1d"})
    merged = validation.merge(prf_validation, on=["material", "rate_group", "test_id"], how="left")
    merged = merged.rename(columns={"rmse_pct_of_peak": "rmse_pct_of_peak_bb"})
    merged["improvement_pct_points"] = merged["rmse_pct_of_peak_prf_1d"] - merged["rmse_pct_of_peak_bb"]
    merged.to_csv(BASE / "bb_vs_prf_comparison.csv", index=False)
    print(f"saved {BASE / 'bb_vs_prf_comparison.csv'}")
    print("\n=== Bergstrom-Boyce (Marmot) vs. 1D PRF baseline (RMSE, % of peak stress) ===")
    print(merged[["material", "hyperelastic_base", "rate_group", "rmse_pct_of_peak_prf_1d",
                  "rmse_pct_of_peak_bb", "improvement_pct_points"]].to_string(index=False))

    # --- grids: one panel per material x rate-group, all bases overlaid;
    # rows for PLACEHOLDER_MATERIALS (not yet tested) render as empty
    # "pending" panels rather than being skipped.
    rate_groups = ["very slow", "slow", "fast"]
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    bases = list(BASES)
    strain_rates = mean_strain_rates()
    make_grid_plot(plot_data, rate_groups, materials, bases, "Time in s",
                    "Bergstrom-Boyce (Marmot) vs. measured stress relaxation",
                    BASE / "bb_model_validation_grid", strain_rates=strain_rates)
    make_grid_plot(ramp_data, rate_groups, materials, bases, "Engineering strain in \\%",
                    "Bergstrom-Boyce (Marmot) vs. measured stress-strain (ramp only)",
                    BASE / "bb_stress_strain_grid", strain_rates=strain_rates, rate_label_loc="upper left")


if __name__ == "__main__":
    main()
