"""Re-fit ALL QLV-Marmot (generalized-Maxwell) material/base combinations
with scipy's differential_evolution (a global optimizer) instead of the
local trust-region least_squares used so far, per the user's request to
verify the currently-reported parameters aren't stuck in a local optimum.

No known solver-hang/pathologically-slow-evaluation issue exists for this
material (unlike BergstromBoyce -- see refit_bb_de.py's docstring), so a
plain differential_evolution call is safe here, no subprocess/timeout
wrapper needed. Each full residual evaluation (~3 rate-group simulations)
measured at ~0.5s; with popsize=15, maxiter=200 and workers=-1 (all cores),
each material/base combination is expected to take on the order of
20-30 minutes wall-clock, ~5-7.5 hours total for all 15 combinations
(5 materials x {NeoHooke, Yeoh, MooneyRivlin}; Ogden excluded, stale
property wiring, see CLAUDE.md).

x0 (the current qlv_marmot_params.csv optimum) is included in DE's initial
population (scipy's `x0` argument) so the known-good basin is represented
from generation 0, without narrowing the bounds DE is allowed to explore.
"""
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from fit_qlv_marmot_model import (
    BASE, MATERIALS, seed_and_bounds, _simulate_once, unpack, _raw_from_gammas, validate,
)
from fit_qlv_model import estimate_youngs_modulus, load, find_ramp_bounds
from fit_prf_model import downsample_for_fit

BASES = ["NeoHooke", "Yeoh", "MooneyRivlin"]
OUT_CSV = BASE / "qlv_marmot_params_de.csv"
LOG = BASE / "qlv_marmot_de.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build_test_data(material, cfg):
    test_data = {}
    for rate_group, ids in cfg["rate_groups"].items():
        df = load(cfg, material, ids[0])
        t, strain, stress = downsample_for_fit(df)
        _, ramp_end = find_ramp_bounds(df["position_mm"])
        t_ramp_end = df["time_s"].iloc[ramp_end]
        is_ramp = t <= t_ramp_end
        n_ramp_pts, n_hold_pts = is_ramp.sum(), (~is_ramp).sum()
        weight = np.where(is_ramp, 1.0 / np.sqrt(max(n_ramp_pts, 1)), 1.0 / np.sqrt(max(n_hold_pts, 1)))
        test_data[rate_group] = (t, 1.0 + strain, stress, stress.max(), weight)
    return test_data


def _cost(x, base_name, test_data):
    # Must be a plain module-level function (not a closure) so
    # functools.partial(...) of it can be pickled for scipy DE's
    # multiprocessing worker pool (workers=-1).
    C1, C2, C3, gammas, taus = unpack(x, base_name)
    total = 0.0
    for t, stretch, stress, peak, weight in test_data.values():
        try:
            pred = _simulate_once(t, stretch, base_name, C1, C2, C3, gammas, taus)
            if not np.all(np.isfinite(pred)):
                raise ValueError("non-finite prediction")
        except Exception:
            pred = np.zeros_like(stress)
        r = (pred - stress) / peak * weight
        total += float(np.sum(r ** 2))
    return total


def main():
    old_params = pd.read_csv(BASE / "qlv_marmot_params.csv")
    rows = []

    for material, cfg in MATERIALS.items():
        test_data = build_test_data(material, cfg)
        tau_lb_value = 0.1 if material == "A100V0" else 1e-3
        for base_name in BASES:
            t0 = time.time()
            log(f"=== DE fitting {material}/{base_name} ===")
            old_row = old_params[(old_params["material"] == material) &
                                  (old_params["hyperelastic_base"] == base_name)].iloc[0]
            gammas_old = [old_row["gamma1"], old_row["gamma2"], old_row["gamma3"]]
            taus_old = [old_row["tau1"], old_row["tau2"], old_row["tau3"]]
            r1, r2, r3 = _raw_from_gammas(gammas_old)
            if base_name == "NeoHooke":
                x0 = [old_row["C1"], r1, r2, r3] + taus_old
            elif base_name == "MooneyRivlin":
                x0 = [old_row["C1"], old_row["C2"], r1, r2, r3] + taus_old
            elif base_name == "Yeoh":
                x0 = [old_row["C1"], old_row["C2"], old_row["C3"], r1, r2, r3] + taus_old
            else:
                raise ValueError(base_name)

            _, lb, ub = seed_and_bounds(base_name, 100.0, [0.2, 0.2, 0.2], [1.0, 10.0, 100.0],
                                        tau_lb_value=tau_lb_value)
            x0 = list(np.clip(x0, lb, ub))
            bounds = list(zip(lb, ub))
            cost = functools.partial(_cost, base_name=base_name, test_data=test_data)

            result = differential_evolution(
                cost, bounds, x0=x0, strategy="best1bin", maxiter=200, popsize=15,
                tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, seed=42,
                workers=-1, updating="deferred", polish=True, init="latinhypercube",
            )
            dt = time.time() - t0
            C1, C2, C3, gammas, taus = unpack(result.x, base_name)
            row = {
                "material": material, "hyperelastic_base": base_name,
                "C1": C1, "C2": C2, "C3": C3, "K": 100.0 * C1,
                "gamma1": gammas[0], "gamma2": gammas[1], "gamma3": gammas[2],
                "tau1": taus[0], "tau2": taus[1], "tau3": taus[2],
                "cost": result.fun, "nfev": result.nfev, "de_success": result.success,
                "old_cost": old_row["cost"], "wall_time_s": dt,
            }
            rows.append(row)
            log(f"  DE done in {dt:.1f}s ({result.nfev} evals): cost {old_row['cost']:.6g} -> {result.fun:.6g} "
                f"(success={result.success})")

            # Save incrementally so a later failure doesn't lose earlier results.
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    log(f"\nALL DONE, saved {OUT_CSV}")


if __name__ == "__main__":
    main()
