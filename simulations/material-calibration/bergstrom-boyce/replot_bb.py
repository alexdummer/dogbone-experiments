"""Regenerate the Bergstrom-Boyce (Arruda-Boyce) validation/stress-strain
grid figures and the vs-1D-PRF RMSE comparison CSV from the CURRENT
bb_params.csv, without re-running the expensive multi-start fit for
materials whose parameters didn't change.

Why this exists: differential_evolution found a substantially better global
optimum for A0V100 specifically (RMSE 19.9-24.1% -> 2.5-3.1%, confirmed via
validate_bb) -- the original multi-start least_squares fit was stuck in a
real local optimum for that one material/base combination. The other four
materials' DE attempts either reproduced the same optimum (A25V75, exact
parameter match) or were worse/incomplete (A75V25 worse; A50V50/A100V0 timed
out per the documented DE-vs-BB solver instability), so only A0V100's row in
bb_params.csv was replaced.
"""
from pathlib import Path

import pandas as pd

from fit_bb_model import BASE, MATERIALS, PLACEHOLDER_MATERIALS, validate_bb, validate_bb_ramp, make_grid_plot
from fit_qlv_model import mean_strain_rates

BASES = ["ArrudaBoyce"]


def main():
    params = pd.read_csv(BASE / "bb_params.csv")
    params = params[params["hyperelastic_base"].isin(BASES)]

    validation_rows = []
    plot_data = {mat: {} for mat in MATERIALS}
    ramp_data = {mat: {} for mat in MATERIALS}

    for material, cfg in MATERIALS.items():
        for _, row in params[params["material"] == material].iterrows():
            base_name = row["hyperelastic_base"]
            A = (row["A1"], row["A2"], row["A3"])
            B = (row["B1"], row["B2"], row["B3"])
            c1, c2, c3 = row["c1"], row["c2"], row["c3"]

            print(f"=== validating {material}/{base_name} ===", flush=True)
            val, plot_rows = validate_bb(material, cfg, base_name, A, B, c1, c2, c3)
            validation_rows.append(val)
            plot_data[material][base_name] = plot_rows
            ramp_data[material][base_name] = validate_bb_ramp(material, cfg, base_name, A, B, c1, c2, c3)

    validation = pd.concat(validation_rows, ignore_index=True)

    prf_validation = pd.read_csv(BASE / "prf_vs_qlv_comparison.csv")[
        ["material", "rate_group", "test_id", "rmse_pct_of_peak_prf"]
    ].rename(columns={"rmse_pct_of_peak_prf": "rmse_pct_of_peak_prf_1d"})
    merged = validation.merge(prf_validation, on=["material", "rate_group", "test_id"], how="left")
    merged = merged.rename(columns={"rmse_pct_of_peak": "rmse_pct_of_peak_bb"})
    merged["improvement_pct_points"] = merged["rmse_pct_of_peak_prf_1d"] - merged["rmse_pct_of_peak_bb"]
    merged.to_csv(BASE / "bb_vs_prf_comparison.csv", index=False)
    print(f"saved {BASE / 'bb_vs_prf_comparison.csv'}")
    print("\n=== Bergstrom-Boyce (Marmot) vs. 1D PRF baseline (RMSE, % of peak stress) ===")
    print(merged[["material", "rate_group", "rmse_pct_of_peak_prf_1d",
                  "rmse_pct_of_peak_bb", "improvement_pct_points"]].to_string(index=False))

    rate_groups = ["very slow", "slow", "fast"]
    materials = list(MATERIALS) + PLACEHOLDER_MATERIALS
    strain_rates = mean_strain_rates()
    make_grid_plot(plot_data, rate_groups, materials, BASES, "Time in s",
                    "Bergstrom-Boyce (Marmot) vs. measured stress relaxation",
                    BASE / "bb_model_validation_grid", strain_rates=strain_rates)
    make_grid_plot(ramp_data, rate_groups, materials, BASES, "Engineering strain in \\%",
                    "Bergstrom-Boyce (Marmot) vs. measured stress-strain (ramp only)",
                    BASE / "bb_stress_strain_grid", strain_rates=strain_rates, rate_label_loc="upper left")


if __name__ == "__main__":
    main()
