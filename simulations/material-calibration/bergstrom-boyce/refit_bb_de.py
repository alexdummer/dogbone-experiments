"""Attempt to re-fit Bergstrom-Boyce (Arruda-Boyce base, the paper's
reported one) with scipy's differential_evolution, per the user's explicit
request to try it despite this project's own prior documented finding (see
fit_bb_model.py's fit_bb_joint docstring) that DE timed out on this model
without completing even one generation: early generations sample broadly
across the bounds, including parameter combinations where the material's
Newton-Raphson return-mapping solve doesn't fail cleanly (cheap, caught) but
converges only after many slow sub-stepped increments (expensive, and nothing
in the residual function can catch or preempt that -- signal-based timeouts
can't interrupt a blocking call into the compiled solver).

Given that, THIS script wraps each material's entire differential_evolution()
call in its own subprocess with a wall-clock timeout (PER_MATERIAL_TIMEOUT_S):
if it hangs the way the prior attempt did, this bounds the damage to one
timeout window per material rather than an indefinite hang, and the batch
still moves on to attempt the rest. A material that times out or errors is
logged and SKIPPED (bb_params.csv keeps its existing, already-validated
multi-start least_squares row for that material -- never silently replaced
with a worse or non-existent result).
"""
import functools
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from fit_bb_model import (
    BASE, MATERIALS, seed_and_bounds, unpack, _simulate_bb_once,
    LAMBDA_L_IDENTIFIABLE_STRAIN,
)
from fit_qlv_model import load, find_ramp_bounds, estimate_youngs_modulus
from fit_prf_model import downsample_for_fit

BASE_NAME = "ArrudaBoyce"
PER_MATERIAL_TIMEOUT_S = 1800  # 30 min bounded attempt per material
OUT_CSV = BASE / "bb_params_de.csv"
LOG = BASE / "bb_de.log"


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


def _cost(x, fix_lambdaL, test_data):
    # Must be a plain module-level function (not a closure) so
    # functools.partial(...) of it can be pickled for scipy DE's
    # multiprocessing worker pool (workers=-1).
    A, B, c1, c2, c3 = unpack(x, BASE_NAME, fix_lambdaL)
    total = 0.0
    for t, stretch, stress, peak, weight in test_data.values():
        try:
            pred = _simulate_bb_once(t, stretch, BASE_NAME, A, B, c1, c2, c3)
            if not np.all(np.isfinite(pred)):
                raise ValueError("non-finite prediction")
        except Exception:
            pred = np.zeros_like(stress)
        r = (pred - stress) / peak * weight
        total += float(np.sum(r ** 2))
    return total


def _de_worker(material, x0, lb, ub, fix_lambdaL, test_data, result_queue):
    cost = functools.partial(_cost, fix_lambdaL=fix_lambdaL, test_data=test_data)
    bounds = list(zip(lb, ub))
    result = differential_evolution(
        cost, bounds, x0=x0, strategy="best1bin", maxiter=200, popsize=15,
        tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, seed=42,
        workers=-1, updating="deferred", polish=True, init="latinhypercube",
    )
    result_queue.put((result.x, result.fun, result.nfev, result.success))


def main():
    old_params = pd.read_csv(BASE / "bb_params.csv")
    old_params = old_params[old_params["hyperelastic_base"] == BASE_NAME]
    rows = []

    for material, cfg in MATERIALS.items():
        t0 = time.time()
        log(f"=== DE fitting {material}/{BASE_NAME} (timeout={PER_MATERIAL_TIMEOUT_S}s) ===")
        old_row = old_params[old_params["material"] == material]
        if old_row.empty:
            log(f"  SKIPPED {material}: no existing bb_params.csv row to seed/compare against")
            continue
        old_row = old_row.iloc[0]
        fix_lambdaL = bool(old_row["lambdaL_fixed"])

        test_data = build_test_data(material, cfg)
        E_mean = np.mean(list(estimate_youngs_modulus(cfg, material).values()))
        muA0 = muB0 = max(E_mean / 3.0 / 2.0, 1e-3)
        c1_0, c2_0, c3_0 = old_row["c1"], old_row["c2"], old_row["c3"]
        _, lb, ub = seed_and_bounds(BASE_NAME, muA0, muB0, c1_0, c2_0, c3_0, fix_lambdaL)

        if fix_lambdaL:
            x0 = [old_row["A1"], old_row["B1"], old_row["c1"], old_row["c2"], old_row["c3"]]
        else:
            x0 = [old_row["A1"], old_row["A2"], old_row["B1"], old_row["B2"],
                  old_row["c1"], old_row["c2"], old_row["c3"]]
        x0 = list(np.clip(x0, lb, ub))

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(target=_de_worker, args=(material, x0, lb, ub, fix_lambdaL, test_data, result_queue))
        process.start()
        process.join(PER_MATERIAL_TIMEOUT_S)
        dt = time.time() - t0
        if process.is_alive():
            process.terminate()
            process.join()
            log(f"  TIMED OUT after {dt:.1f}s -- SKIPPED (keeping existing bb_params.csv row for {material})")
            continue
        if result_queue.empty():
            log(f"  FAILED (no result, exit code {process.exitcode}) after {dt:.1f}s -- SKIPPED")
            continue

        x_opt, fun, nfev, success = result_queue.get()
        A, B, c1, c2, c3 = unpack(x_opt, BASE_NAME, fix_lambdaL)
        row = {
            "material": material, "hyperelastic_base": BASE_NAME,
            "A1": A[0], "A2": A[1], "A3": A[2], "B1": B[0], "B2": B[1], "B3": B[2],
            "kappaA": 100.0 * A[0], "kappaB": 100.0 * A[0],
            "c1": c1, "c2": c2, "c3": c3, "lambdaL_fixed": fix_lambdaL,
            "cost": fun, "nfev": nfev, "de_success": success,
            "old_cost": old_row["cost"], "wall_time_s": dt,
        }
        rows.append(row)
        log(f"  DE done in {dt:.1f}s ({nfev} evals): cost {old_row['cost']:.6g} -> {fun:.6g} (success={success})")
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    log(f"\nALL DONE, saved {OUT_CSV} ({len(rows)}/{len(MATERIALS)} materials succeeded)")


if __name__ == "__main__":
    main()
