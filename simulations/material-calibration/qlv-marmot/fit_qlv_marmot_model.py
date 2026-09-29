"""Calibrate Marmot's CompressibleFiniteStrainLinearViscoelasticity material --
a finite-strain generalized-Maxwell (Prony) hyper-viscoelastic model with a
selectable hyperelastic base potential -- against the same stress-relaxation
experiments used by fit_bb_model.py, driving the real tensorial material
through Marmot's Python material-point solver exactly as that script does
(axial gradU strain-controlled to replay the measured stretch history,
transverse normal components stress-free).

This is a genuinely different model from the existing fit_qlv_model.py: that
script hand-rolls a 2-term-Ogden + 3-term-Prony fit with plain scipy, with no
FE/material-point solver involved. Here we calibrate Marmot's own built-in
material model (already implemented, no C++ work needed), which combines:

  - one of four hyperelastic base potentials (BergstromBoyce-style property
    #0, "hyperelasticBase"): NeoHooke (0), Yeoh (1), MooneyRivlin (2) -- see
    modules/materials/CompressibleFiniteStrainLinearViscoelasticity/include/
    Marmot/CompressibleFiniteStrainLinearViscoelasticity.h for the exact
    property layout;
  - 3 Maxwell ("dissipative") processes, i.e. a 3-term Prony series in the
    2nd Piola-Kirchhoff stress space (property "n_Maxwell" = 3, followed by
    3 (gamma_i, tau_i) pairs) -- gamma_i/tau_i have EXACTLY the same meaning
    as fit_qlv_model.py's g_i/tau_i (g_inf = 1 - sum(gamma_i) is implicit),
    so the existing qlv_prony_params.csv fit is a natural, physically
    meaningful seed.

"onlyShearCreep" is fixed at 1 (viscoelastic evolution acts on the deviatoric
part of the initial PK2 stress only, volumetric response instantaneous) --
the same "compressibility can't be identified from uniaxial data, so keep it
simple and near-incompressible" stance as fit_bb_model.py, here expressed as
"only the shape-changing part creeps" rather than "kappa is fixed".

Every base now composes Psi = Psi_iso(shape params) + kappa/8*(ln detC)^2,
with kappa always the LAST elasticProperties entry, fixed at kappa=100*C1
near-incompressible (mirroring fit_bb_model.py's kappa=100*mu convention):
  - NeoHooke:     elasticProperties = [mu, kappa]        (2 params, mu free)
  - MooneyRivlin: elasticProperties = [C1, C2, kappa]     (3 params, C1,C2 free)
  - Yeoh:         elasticProperties = [C1, C2, C3, kappa] (4 params, C1,C2,C3 free)

Learned from fit_bb_model.py's runaway: this script deliberately uses a
SINGLE-start least_squares (no multi-start) -- there is no known narrow
solver-instability here to work around, and a 7-point multi-start was what
made that script's 3x3 fit run for 90+ minutes. Timed on one combination
before committing to a full run.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import marmot

from fit_prf_model import downsample_for_fit
from fit_qlv_model import BASE, MATERIALS, load, find_ramp_bounds, mean_strain_rates, estimate_youngs_modulus
from plotstyle import colors, figsize_double

# Compositions not yet tested (see specimens/); rendered as empty "pending"
# rows in the comparison grids so the layout is already in place for them.
# A75V25 has real (partial) data now -- moved into MATERIALS.
PLACEHOLDER_MATERIALS = []

BASES = {"NeoHooke": 0, "Yeoh": 1, "MooneyRivlin": 2, "Ogden": 5}
N_ELASTIC = {"NeoHooke": 2, "Yeoh": 4, "MooneyRivlin": 3, "Ogden": 3}
BASE_LINESTYLE = {"NeoHooke": "-", "Yeoh": "--", "MooneyRivlin": ":",
                  "ArrudaBoyce": (0, (5, 1, 1, 1)), "Ogden": (0, (1, 1))}
BASE_MARKER = {"NeoHooke": "o", "Yeoh": "s", "MooneyRivlin": "^", "ArrudaBoyce": "v",
              "Ogden": "*"}
N_MAXWELL = 3
ONLY_SHEAR_CREEP = 1.0

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


def build_props(base_name, elastic, gammas, taus):
    """Assemble the CompressibleFiniteStrainLinearViscoelasticity material
    property array: hyperelasticBase, onlyShearCreep, elasticProperties (base-
    dependent length), n_Maxwell, then (gamma_i, tau_i) pairs."""
    base_id = BASES[base_name]
    assert len(elastic) == N_ELASTIC[base_name]
    assert len(gammas) == len(taus) == N_MAXWELL
    props = [base_id, ONLY_SHEAR_CREEP, *elastic, N_MAXWELL]
    for g, t in zip(gammas, taus):
        props += [g, t]
    return np.array(props, dtype=np.float64)


def elastic_from_C1(base_name, C1, C2, C3):
    """Map the free elastic coefficient(s) to the base's elasticProperties
    array, with kappa = 100*C1 fixed near-incompressible and always the LAST
    slot (every base now composes Psi = Psi_iso(shape params) +
    kappa/8*(ln detC)^2 -- see the module docstring). For "Ogden", C1 plays
    the role of the single-term Ogden modulus mu and C2 the exponent alpha."""
    kappa = 100.0 * C1
    if base_name == "NeoHooke":
        return [C1, kappa]  # here "C1" plays the role of mu (shear modulus)
    if base_name == "MooneyRivlin":
        return [C1, C2, kappa]
    if base_name == "Yeoh":
        return [C1, C2, C3, kappa]
    if base_name == "Ogden":
        return [C1, C2, kappa]  # C1=mu, C2=alpha
    raise ValueError(base_name)


def _simulate_once(t, stretch, base_name, C1, C2, C3, gammas, taus):
    """Replay a measured (time, stretch) history through the Marmot
    CompressibleFiniteStrainLinearViscoelasticity material-point solver;
    returns predicted nominal (engineering, 1st Piola-Kirchhoff) axial stress,
    converted from the solver's native Kirchhoff stress via P11 = tau11/F11
    (F is diagonal here, so this is exact, not approximate)."""
    elastic = elastic_from_C1(base_name, C1, C2, C3)
    props = build_props(base_name, elastic, gammas, taus)
    options = marmot.solvers.FiniteStrainSolver.SolverOptions()
    solver = marmot.solvers.FiniteStrainSolver("COMPRESSIBLEFINITESTRAINLINEARVISCOELASTICITY", props, options)

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt <= 0:
            continue
        step = marmot.solvers.FiniteStrainSolver.Step()
        step.timeStart = float(t[i - 1])
        step.timeEnd = float(t[i])
        step.dTStart = dt
        step.dTMax = dt
        step.dTMin = dt / 50
        step.maxIncrements = 30
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


def _gammas_from_raw(r):
    """Map 3 unconstrained reals to (gamma1, gamma2, gamma3) with
    sum(gammas) < 1 GUARANTEED BY CONSTRUCTION, via a smooth stick-breaking
    (sequential-fraction) transform: g_i = remaining_i * sigmoid(r_i),
    remaining_{i+1} = remaining_i - g_i, remaining_0 = 1. Each sigmoid(r_i) in
    (0,1) strictly, so remaining stays positive and g_inf = 1 - sum(gammas)
    (the implicit equilibrium/long-term modulus fraction of a generalized-
    Maxwell model) is always > 0 -- a negative g_inf is thermodynamically
    invalid (a negative equilibrium stiffness contribution).

    Bounding each gamma_i independently to [0, 0.95] (the previous approach)
    does NOT enforce this: three independent 0.95 caps allow sum(gammas) up
    to 2.85, and the unconstrained optimizer found sum > 1 solutions (g_inf <
    0) for 3 of 5 materials in practice. Reparametrizing so the constraint
    holds for every possible raw input, rather than bounding gammas directly
    and hoping the optimizer stays inside the (non-box-shaped) feasible
    simplex, is robust to this regardless of where the optimizer wanders."""
    remaining = 1.0
    gammas = []
    for ri in r:
        frac = 1.0 / (1.0 + np.exp(-ri))
        g = remaining * frac
        gammas.append(g)
        remaining -= g
    return gammas


def _raw_from_gammas(gammas):
    """Inverse of _gammas_from_raw, for seeding the raw parametrization from
    a target (gamma1, gamma2, gamma3) (e.g. qlv_prony_params.csv's g_i)."""
    remaining = 1.0
    raw = []
    for g in gammas:
        frac = np.clip(g / remaining, 1e-9, 1 - 1e-9)
        raw.append(np.log(frac / (1 - frac)))
        remaining -= g
    return raw


def unpack(x, base_name):
    if base_name == "NeoHooke":
        C1, r1, r2, r3, t1, t2, t3 = x
        return C1, 0.0, 0.0, _gammas_from_raw([r1, r2, r3]), [t1, t2, t3]
    if base_name in ("MooneyRivlin", "Ogden"):
        C1, C2, r1, r2, r3, t1, t2, t3 = x
        return C1, C2, 0.0, _gammas_from_raw([r1, r2, r3]), [t1, t2, t3]
    if base_name == "Yeoh":
        C1, C2, C3, r1, r2, r3, t1, t2, t3 = x
        return C1, C2, C3, _gammas_from_raw([r1, r2, r3]), [t1, t2, t3]
    raise ValueError(base_name)


def seed_and_bounds(base_name, G0, g0, tau0, tau_lb_value=1e-3):
    # Leading-modulus lower bound: was 1.0 MPa across all three bases,
    # unconditionally, the same bounds-vs-real-stiffness mismatch found (and
    # fixed the same way) in fit_bb_model.py's seed_and_bounds -- A100V0's
    # real modulus is only ~1-9 MPa (rate-dependent tangent) / ~0.7-2.0 MPa
    # (secant at peak strain), so a 1.0 floor left almost no room below the
    # seed and could clip a Yeoh/MooneyRivlin C1_0=G0/2 seed upward for it.
    # 1e-3 is a generic positivity floor, not an assumed physical minimum.
    #
    # g_lb/g_ub/g_x0 are the raw stick-breaking parameters (see
    # _gammas_from_raw), NOT the gammas themselves -- bounding gammas
    # directly to e.g. [0,0.95] each does not prevent sum(gammas) > 1 (found
    # in practice for 3 of 5 materials: an unphysical negative equilibrium
    # modulus). +/-12 saturates the sigmoid to within 1e-5 of 0/1, wide
    # enough not to constrain the fit in practice.
    # tau_lb_value: raised above the generic 1e-3 floor for A100V0 (see
    # main()) -- an unconstrained fit drove one Maxwell branch there down to
    # tau=0.0165s (79% of the total weight), which combined with A100V0's
    # already tiny base stiffness (C10~0.006 MPa, kappa~0.6 MPa) produced a
    # numerically ill-conditioned FE tangent stiffness (NaN in the linear
    # solve a few seconds into the relaxation hold, independent of time-step
    # size -- not a step-size/convergence issue, a genuine conditioning
    # problem). A higher floor forces the fit away from pathologically fast
    # branches even if that costs a slightly worse fit to the fastest part
    # of the transient.
    g_lb, g_ub = [-12.0] * 3, [12.0] * 3
    tau_lb, tau_ub = [tau_lb_value] * 3, [1e5] * 3
    g_x0, tau_x0 = list(_raw_from_gammas(g0)), list(tau0)
    if base_name == "NeoHooke":
        x0 = [G0] + g_x0 + tau_x0
        lb = [1e-3] + g_lb + tau_lb
        ub = [1e4] + g_ub + tau_ub
    elif base_name == "MooneyRivlin":
        C1_0 = G0 / 2
        x0 = [C1_0, 0.1 * C1_0] + g_x0 + tau_x0
        lb = [1e-3, -1e3] + g_lb + tau_lb
        ub = [1e4, 1e4] + g_ub + tau_ub
    elif base_name == "Yeoh":
        C1_0 = G0 / 2
        x0 = [C1_0, 0.01 * C1_0, 0.0] + g_x0 + tau_x0
        lb = [1e-3, -1e3, -1e3] + g_lb + tau_lb
        ub = [1e4, 1e4, 1e4] + g_ub + tau_ub
    elif base_name == "Ogden":
        # single-term Ogden: G0 = mu*alpha/2, a genuine mu/alpha degeneracy at
        # the small-strain limit -- seed at alpha=2 (the exact neo-Hookean
        # value), which pins mu_0=G0 unambiguously and gives a seed
        # equivalent to a NeoHooke fit, letting the joint fit move away from
        # alpha=2 only if the data actually supports it.
        alpha_0 = 2.0
        mu_0 = G0
        x0 = [mu_0, alpha_0] + g_x0 + tau_x0
        lb = [1e-3, 0.5] + g_lb + tau_lb
        ub = [1e4, 20.0] + g_ub + tau_ub
    else:
        raise ValueError(base_name)
    return list(np.clip(x0, lb, ub)), lb, ub


def fit_joint(material, cfg, base_name, G0, g0, tau0, max_nfev=100, verbose=False, x0_override=None,
              tau_lb_value=1e-3):
    test_data = {}
    for rate_group, ids in cfg["rate_groups"].items():
        df = load(cfg, material, ids[0])
        # downsample_for_fit returns df["strain"] (the extensometer channel) at
        # each sampled time -- it only uses crosshead position internally to
        # find the ramp/hold boundary, not to derive strain values themselves.
        t, strain, stress = downsample_for_fit(df)
        # downsample_for_fit uses a FIXED point count per phase (n_ramp=40,
        # n_hold=60) regardless of each phase's actual duration -- the ramp is
        # typically 10-600x shorter than the 600s hold, so its points end up
        # far denser in time. Since least_squares weighs every residual entry
        # equally, this silently lets the ramp's shape dominate the fit cost
        # purely from being oversampled, under-fitting the relaxation branch
        # relative to the ramp. Weight each point so the two phases
        # contribute equally to the total cost regardless of point count
        # (weight = 1/sqrt(n_points_in_that_phase); see fit_bb_model.py's
        # identical fix for the full derivation/measurements).
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t <= t_ramp_end
        n_ramp_pts, n_hold_pts = is_ramp.sum(), (~is_ramp).sum()
        weight = np.where(is_ramp, 1.0 / np.sqrt(max(n_ramp_pts, 1)), 1.0 / np.sqrt(max(n_hold_pts, 1)))
        test_data[rate_group] = (t, 1.0 + strain, stress, stress.max(), weight)

    x0, lb, ub = seed_and_bounds(base_name, G0, g0, tau0, tau_lb_value=tau_lb_value)
    if x0_override is not None:
        x0 = list(np.clip(x0_override, lb, ub))

    def residuals(x):
        C1, C2, C3, gammas, taus = unpack(x, base_name)
        out = []
        for t, stretch, stress, peak, weight in test_data.values():
            try:
                pred = _simulate_once(t, stretch, base_name, C1, C2, C3, gammas, taus)
            except Exception:
                pred = np.zeros_like(stress)
            out.append((pred - stress) / peak * weight)
        return np.concatenate(out)

    result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", verbose=2 if verbose else 0,
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, diff_step=1e-3, max_nfev=max_nfev)
    C1, C2, C3, gammas, taus = unpack(result.x, base_name)
    return {
        "hyperelastic_base": base_name, "C1": C1, "C2": C2, "C3": C3,
        "K": 100.0 * C1,
        "gamma1": gammas[0], "gamma2": gammas[1], "gamma3": gammas[2],
        "tau1": taus[0], "tau2": taus[1], "tau3": taus[2],
        "cost": result.cost, "nfev": result.nfev,
    }


def validate(material, cfg, base_name, C1, C2, C3, gammas, taus):
    rows = []
    plot_rows = {}
    for rate_group, ids in cfg["rate_groups"].items():
        test_id = ids[0]
        df = load(cfg, material, test_id)
        t = df["time_s"].to_numpy()
        measured = df["stress"].to_numpy()
        t_ds, strain_ds, stress_ds = downsample_for_fit(df, n_ramp=60, n_hold=90)
        try:
            pred = _simulate_once(t_ds, 1.0 + strain_ds, base_name, C1, C2, C3, gammas, taus)
        except Exception as exc:
            print(f"  [validate] {material}/{base_name}/{rate_group}: solver failed ({exc}); skipping")
            continue
        peak = stress_ds.max()
        rmse = np.sqrt(np.mean((pred - stress_ds) ** 2))
        rows.append({"material": material, "hyperelastic_base": base_name, "rate_group": rate_group,
                     "test_id": test_id, "rmse_mpa": rmse, "rmse_pct_of_peak": 100 * rmse / peak})
        plot_rows[rate_group] = (t, measured, t_ds, pred)
    return pd.DataFrame(rows), plot_rows


def validate_ramp(material, cfg, base_name, C1, C2, C3, gammas, taus):
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
            pred = _simulate_once(t_ds, 1.0 + strain_ds, base_name, C1, C2, C3, gammas, taus)
        except Exception as exc:
            print(f"  [validate_ramp] {material}/{base_name}/{rate_group}: solver failed ({exc}); skipping")
            continue
        ramp_rows[rate_group] = (100 * strain_full, measured_stress, 100 * strain_ds, pred)
    return ramp_rows


def make_grid_plot(data_by_material_base, rate_groups, materials, bases, xlabel, suptitle, out_path,
                    strain_rates=None, sharex=True, rate_label_loc="upper right"):
    """materials may include compositions with no entry in data_by_material_base
    (e.g. PLACEHOLDER_MATERIALS) -- those rows are rendered as empty "pending"
    panels, compressed to a fraction of a normal row's height, rather than
    skipped, so the grid's layout is already in place for when that data
    exists. strain_rates, if given, is strain_rates[material][rate_group] ->
    mean engineering strain rate (1/s) over the representative test's ramp,
    annotated in the corner of each real panel."""
    n_real = sum(1 for m in materials if m in data_by_material_base)
    n_placeholder = len(materials) - n_real
    height_ratios = [1.0] * n_real + [0.2] * n_placeholder
    # Total figure height is capped in absolute terms (MAX_TOTAL_HEIGHT_IN),
    # not just scaled per row: a fixed per_row_scale=0.8 was tuned for the
    # 3-real-row case (height_ratios summing to 3.2-3.4 depending on
    # placeholders) and rendered a PDF safely under matplotlib's/LaTeX's
    # ~25cm page-height danger zone. Adding a 4th real row (A100V0, sum=4.2)
    # silently pushed the rendered page height to ~25.4cm -- just over that
    # threshold -- which triggered a LaTeX float-placement pathology
    # (270k+ pages shipped) the moment this figure was included in the
    # paper. Capping the total height directly, rather than re-tuning a
    # per-row constant by hand every time a material is added, keeps this
    # safe regardless of how many real/placeholder rows exist in the future.
    MAX_TOTAL_HEIGHT_IN = 7.5  # ~19cm rendered height, well clear of the danger zone
    per_row_scale = min(0.8, MAX_TOTAL_HEIGHT_IN / (figsize_double[1] * sum(height_ratios)))
    fig, axes = plt.subplots(len(materials), 3,
                              figsize=(figsize_double[0], figsize_double[1] * per_row_scale * sum(height_ratios)),
                              sharey="row", sharex=sharex, gridspec_kw={"height_ratios": height_ratios})
    for row, material in enumerate(materials):
        is_placeholder = material not in data_by_material_base
        for col, rate_group in enumerate(rate_groups):
            ax = axes[row, col]
            if is_placeholder:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=7,
                        color="grey", style="italic", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                if row == 0:
                    ax.set_title(rate_group, fontsize=9)
                if col == 0:
                    ax.set_ylabel(material, fontsize=8)
                continue
            color = colors[col]
            x_full, measured = None, None
            for base_name in bases:
                if base_name not in data_by_material_base[material]:
                    continue
                if rate_group not in data_by_material_base[material][base_name]:
                    continue
                x_full, measured, x_ds, pred = data_by_material_base[material][base_name][rate_group]
                ax.plot(x_ds, pred, color=color, linewidth=1.1, linestyle=BASE_LINESTYLE[base_name],
                        marker=BASE_MARKER[base_name], markersize=2.2, markevery=4)
            if measured is not None:
                ax.plot(x_full, measured, color=color, linewidth=2.2, alpha=0.35, zorder=0)
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(rate_group, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{material}\nEngineering stress in MPa", fontsize=8)
            if row == n_real - 1:
                ax.set_xlabel(xlabel)
            if strain_rates is not None:
                rate = strain_rates.get(material, {}).get(rate_group)
                if rate is not None:
                    label_x, ha = (0.04, "left") if rate_label_loc == "upper left" else (0.96, "right")
                    ax.text(label_x, 0.96, rf"$\dot\varepsilon\approx${100 * rate:.3g}\%/s", fontsize=6,
                            ha=ha, va="top", transform=ax.transAxes, color="black",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))

    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color="grey", alpha=0.35, linewidth=2.2, label="measured")]
    legend_handles += [
        Line2D([0], [0], color="grey", linestyle=BASE_LINESTYLE[b], marker=BASE_MARKER[b],
               markersize=4, label=b) for b in bases
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(bases) + 1, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(suptitle, fontsize=11, y=1.07)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path.with_suffix('.pdf')} and .png")


def main():
    prony = pd.read_csv(BASE / "qlv_prony_params.csv", index_col=0)

    all_rows = []
    validation_rows = []
    plot_data = {mat: {} for mat in MATERIALS}
    ramp_data = {mat: {} for mat in MATERIALS}

    for material, cfg in MATERIALS.items():
        # G0 from a model-free tangent-Young's-modulus estimate on the raw
        # measured ramp (estimate_youngs_modulus), not the 2-term-Ogden fit's
        # small_strain_modulus_3mu -- that Ogden fit is itself poor for some
        # materials (e.g. A100V0: R^2=0.60, see fit_qlv_model.py) and can
        # disagree with the material's real stiffness by a large factor, the
        # same seeding mismatch found in fit_bb_model.py.
        E_mean = np.mean(list(estimate_youngs_modulus(cfg, material).values()))
        G0 = E_mean / 3.0
        g0 = [float(prony.loc[material, f"g{i}"]) for i in (1, 2, 3)]
        tau0 = [float(prony.loc[material, f"tau{i}"]) for i in (1, 2, 3)]

        plot_data[material] = {}
        ramp_data[material] = {}

        # A100V0's unconstrained fit drove a Maxwell branch to tau=0.0165s,
        # which (combined with its already-tiny base stiffness) produced a
        # numerically ill-conditioned FE tangent stiffness (NaN in the
        # linear solve during the relaxation hold -- see seed_and_bounds's
        # tau_lb_value docstring). Raised only for this material; the other
        # 4 already fit and simulate fine with the generic 1e-3 floor.
        tau_lb_value = 0.1 if material == "A100V0" else 1e-3

        for base_name in BASES:
            print(f"\n=== fitting CompressibleFiniteStrainLinearViscoelasticity, {base_name}: {material} ===", flush=True)
            params = fit_joint(material, cfg, base_name, G0, g0, tau0, verbose=False, tau_lb_value=tau_lb_value)
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}, flush=True)
            row = {"material": material}
            row.update(params)
            all_rows.append(row)

            C1, C2, C3 = params["C1"], params["C2"], params["C3"]
            gammas = [params["gamma1"], params["gamma2"], params["gamma3"]]
            taus = [params["tau1"], params["tau2"], params["tau3"]]

            val, plot_rows = validate(material, cfg, base_name, C1, C2, C3, gammas, taus)
            validation_rows.append(val)
            plot_data[material][base_name] = plot_rows
            ramp_data[material][base_name] = validate_ramp(material, cfg, base_name, C1, C2, C3, gammas, taus)

    validation = pd.concat(validation_rows, ignore_index=True)
    params_df = pd.DataFrame(all_rows)
    params_df.to_csv(BASE / "qlv_marmot_params.csv", index=False)
    print(f"\nsaved {BASE / 'qlv_marmot_params.csv'}")

    prf_validation = pd.read_csv(BASE / "prf_vs_qlv_comparison.csv")[
        ["material", "rate_group", "test_id", "rmse_pct_of_peak_qlv"]
    ]
    merged = validation.merge(prf_validation, on=["material", "rate_group", "test_id"], how="left")
    merged = merged.rename(columns={"rmse_pct_of_peak": "rmse_pct_of_peak_marmot"})
    merged["improvement_pct_points"] = merged["rmse_pct_of_peak_qlv"] - merged["rmse_pct_of_peak_marmot"]
    merged.to_csv(BASE / "qlv_marmot_vs_qlv1d_comparison.csv", index=False)
    print(f"saved {BASE / 'qlv_marmot_vs_qlv1d_comparison.csv'}")
    print("\n=== Marmot generalized-Maxwell model vs. 1D QLV baseline (RMSE, % of peak stress) ===")
    print(merged[["material", "hyperelastic_base", "rate_group", "rmse_pct_of_peak_qlv",
                  "rmse_pct_of_peak_marmot", "improvement_pct_points"]].to_string(index=False))

    rate_groups = ["very slow", "slow", "fast"]
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    bases = list(BASES)
    strain_rates = mean_strain_rates()
    make_grid_plot(plot_data, rate_groups, materials, bases, "Time in s",
                    "CompressibleFiniteStrainLinearViscoelasticity (Marmot) vs. measured stress relaxation",
                    BASE / "qlv_marmot_validation_grid", strain_rates=strain_rates)
    make_grid_plot(ramp_data, rate_groups, materials, bases, "Engineering strain in \\%",
                    "CompressibleFiniteStrainLinearViscoelasticity (Marmot) vs. measured stress-strain (ramp only)",
                    BASE / "qlv_marmot_stress_strain_grid", strain_rates=strain_rates, sharex=False,
                    rate_label_loc="upper left")


if __name__ == "__main__":
    main()
