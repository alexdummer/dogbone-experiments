"""Regenerate the QLV-Marmot validation/stress-strain grid figures and the
vs-1D-QLV RMSE comparison CSV from the CURRENT qlv_marmot_params.csv, without
re-running the (expensive, and already-correct) fit_joint optimization.

Why this exists: qlv_marmot_params.csv was already re-fit after the
thermodynamic (sum(gamma_i)<=1) constraint fix, but the paper's figures/table
still reflected the pre-fix parameters (stale by several days) -- confirmed by
the old table's gammas summing to >1 for A0V100. Rather than risk a fresh
optimization landing on different local optima (and shifting the narrative's
numbers yet again), this just re-runs validate()/validate_ramp() (pure
forward simulation, no fitting) on the already-fitted parameters and rebuilds
the plots/comparison CSV from that.

Restricted to bases=[NeoHooke, Yeoh, MooneyRivlin]: Ogden is excluded per
CLAUDE.md ("fit_qlv_marmot_model.py's Ogden wiring (N_ELASTIC["Ogden"]=3) is
stale: the installed C++ material needs 7 elastic-property slots") -- its
rows in qlv_marmot_params.csv are from a mis-wired property vector and not
meaningful (visibly: cost ~1.7-1.75, an order of magnitude worse than the
other three bases' ~0.0008-0.06).
"""
from pathlib import Path

import pandas as pd

from fit_qlv_marmot_model import (
    BASE, MATERIALS, PLACEHOLDER_MATERIALS, validate, validate_ramp, make_grid_plot,
)
from fit_qlv_model import mean_strain_rates

BASES = ["NeoHooke", "Yeoh", "MooneyRivlin"]  # Ogden excluded, see module docstring


def main():
    params = pd.read_csv(BASE / "qlv_marmot_params.csv")
    params = params[params["hyperelastic_base"].isin(BASES)]

    validation_rows = []
    plot_data = {mat: {} for mat in MATERIALS}
    ramp_data = {mat: {} for mat in MATERIALS}

    for material, cfg in MATERIALS.items():
        for _, row in params[params["material"] == material].iterrows():
            base_name = row["hyperelastic_base"]
            C1, C2, C3 = row["C1"], row["C2"], row["C3"]
            gammas = [row["gamma1"], row["gamma2"], row["gamma3"]]
            taus = [row["tau1"], row["tau2"], row["tau3"]]

            print(f"=== validating {material}/{base_name} ===", flush=True)
            val, plot_rows = validate(material, cfg, base_name, C1, C2, C3, gammas, taus)
            validation_rows.append(val)
            plot_data[material][base_name] = plot_rows
            ramp_data[material][base_name] = validate_ramp(material, cfg, base_name, C1, C2, C3, gammas, taus)

    validation = pd.concat(validation_rows, ignore_index=True)

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

    print("\n=== Full RMSE table (% of peak stress), all bases x materials x rates ===")
    print(validation[["material", "hyperelastic_base", "rate_group", "rmse_pct_of_peak"]]
          .pivot_table(index=["material", "rate_group"], columns="hyperelastic_base", values="rmse_pct_of_peak")
          .to_string())

    rate_groups = ["very slow", "slow", "fast"]
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    strain_rates = mean_strain_rates()
    make_grid_plot(plot_data, rate_groups, materials, BASES, "Time in s",
                    "CompressibleFiniteStrainLinearViscoelasticity (Marmot) vs. measured stress relaxation",
                    BASE / "qlv_marmot_validation_grid", strain_rates=strain_rates)
    make_grid_plot(ramp_data, rate_groups, materials, BASES, "Engineering strain in \\%",
                    "CompressibleFiniteStrainLinearViscoelasticity (Marmot) vs. measured stress-strain (ramp only)",
                    BASE / "qlv_marmot_stress_strain_grid", strain_rates=strain_rates, sharex=False,
                    rate_label_loc="upper left")


if __name__ == "__main__":
    main()
