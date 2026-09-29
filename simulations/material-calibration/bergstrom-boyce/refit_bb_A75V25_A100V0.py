"""Re-fit BB (Arruda-Boyce) for A75V25 and A100V0 ONLY against the corrected
videoextensometer data (shoulder-drift fix + auto-tracking, merged
2026-09-11 -- after this project's bb_params.csv fit on 2026-09-09).
Companion to refit_qlv_marmot_A75V25_A100V0.py, same underlying data issue.

Uses fit_bb_joint's own existing robust multi-start (nudge cluster + random
restarts, each in an isolated subprocess with a timeout) -- NOT
differential_evolution, which just demonstrated (both in this project's
prior documented attempt and in this session's own refit_bb_de.py run) that
it does not work reliably for this model (A75V25: worse than the local
fit; A50V50/A100V0: timed out without a result).
"""
import pandas as pd

from fit_bb_model import BASE, MATERIALS, fit_bb_joint
from fit_qlv_model import estimate_youngs_modulus

REFIT_MATERIALS = ["A75V25", "A100V0"]
BASE_NAME = "ArrudaBoyce"


def main():
    prf_params_df = pd.read_csv(BASE / "prf_params.csv", index_col=0)
    qlv_prony_df = pd.read_csv(BASE / "qlv_prony_params.csv", index_col=0)
    old_params = pd.read_csv(BASE / "bb_params.csv")

    new_rows = []
    for material in REFIT_MATERIALS:
        cfg = MATERIALS[material]
        prf_params = prf_params_df.loc[material].to_dict()
        qlv_prony = qlv_prony_df.loc[material].to_dict()
        E_mean = sum(estimate_youngs_modulus(cfg, material).values()) / len(
            estimate_youngs_modulus(cfg, material))
        muA0 = muB0 = max(E_mean / 3.0 / 2.0, 1e-3)

        old_row = old_params[(old_params["material"] == material) &
                              (old_params["hyperelastic_base"] == BASE_NAME)].iloc[0]

        print(f"=== re-fitting {material}/{BASE_NAME} against corrected data ===", flush=True)
        params = fit_bb_joint(material, cfg, BASE_NAME, muA0, muB0, prf_params, qlv_prony, verbose=True)
        print(f"  old cost={old_row['cost']:.6g} -> new cost={params['cost']:.6g}")
        row = {"material": material}
        row.update(params)
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    mask = old_params["material"].isin(REFIT_MATERIALS) & (old_params["hyperelastic_base"] == BASE_NAME)
    merged = pd.concat([old_params[~mask], new_df], ignore_index=True)
    merged = merged.sort_values(["material", "hyperelastic_base"]).reset_index(drop=True)
    merged.to_csv(BASE / "bb_params.csv", index=False)
    print(f"\nsaved {BASE / 'bb_params.csv'}")


if __name__ == "__main__":
    main()
