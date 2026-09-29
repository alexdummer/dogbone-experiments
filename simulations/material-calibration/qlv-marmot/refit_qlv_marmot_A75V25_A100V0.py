"""Re-fit ONLY A75V25 and A100V0 (all 3 valid bases: NeoHooke, Yeoh,
MooneyRivlin) against the corrected videoextensometer data (shoulder-drift
fix + per-test auto-detected tracking points, merged into main.tex on
2026-09-11 17:07/18:44 -- after this project's last qlv_marmot_params.csv
fit on 2026-09-10 22:21). The other three materials' underlying data is
unaffected, so re-fitting them too would just waste time re-deriving the
same optimum.

Seeds from each material/base's PREVIOUS optimum (qlv_marmot_params.csv),
per this project's convention for re-fits after a data/formula change: much
faster than a generic seed, and the size of the parameter movement is itself
informative.
"""
import pandas as pd

from fit_qlv_marmot_model import (
    BASE, MATERIALS, fit_joint, _raw_from_gammas, unpack,
)
from fit_qlv_model import estimate_youngs_modulus

REFIT_MATERIALS = ["A75V25", "A100V0"]
BASES = ["NeoHooke", "Yeoh", "MooneyRivlin"]


def main():
    prony = pd.read_csv(BASE / "qlv_prony_params.csv", index_col=0)
    old_params = pd.read_csv(BASE / "qlv_marmot_params.csv")

    new_rows = []
    for material in REFIT_MATERIALS:
        cfg = MATERIALS[material]
        E_mean = sum(estimate_youngs_modulus(cfg, material).values()) / len(
            estimate_youngs_modulus(cfg, material))
        G0 = E_mean / 3.0
        g0 = [float(prony.loc[material, f"g{i}"]) for i in (1, 2, 3)]
        tau0 = [float(prony.loc[material, f"tau{i}"]) for i in (1, 2, 3)]
        tau_lb_value = 0.1 if material == "A100V0" else 1e-3

        for base_name in BASES:
            old_row = old_params[(old_params["material"] == material) &
                                  (old_params["hyperelastic_base"] == base_name)].iloc[0]
            gammas_old = [old_row["gamma1"], old_row["gamma2"], old_row["gamma3"]]
            taus_old = [old_row["tau1"], old_row["tau2"], old_row["tau3"]]
            r1, r2, r3 = _raw_from_gammas(gammas_old)
            if base_name == "NeoHooke":
                x0_override = [old_row["C1"], r1, r2, r3] + taus_old
            elif base_name == "MooneyRivlin":
                x0_override = [old_row["C1"], old_row["C2"], r1, r2, r3] + taus_old
            elif base_name == "Yeoh":
                x0_override = [old_row["C1"], old_row["C2"], old_row["C3"], r1, r2, r3] + taus_old
            else:
                raise ValueError(base_name)

            print(f"=== re-fitting {material}/{base_name} (seeded from previous optimum) ===", flush=True)
            params = fit_joint(material, cfg, base_name, G0, g0, tau0, verbose=False,
                                x0_override=x0_override, tau_lb_value=tau_lb_value)
            moved = {k: (round(params[k] - old_row[k], 6) if isinstance(params[k], float) else None)
                     for k in ("C1", "C2", "C3", "gamma1", "gamma2", "gamma3", "tau1", "tau2", "tau3")}
            print(f"  old cost={old_row['cost']:.6g} -> new cost={params['cost']:.6g}")
            print(f"  parameter movement: {moved}")
            row = {"material": material}
            row.update(params)
            new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    # replace just these rows in the full params table, keep everything else untouched
    mask = old_params["material"].isin(REFIT_MATERIALS) & old_params["hyperelastic_base"].isin(BASES)
    merged = pd.concat([old_params[~mask], new_df], ignore_index=True)
    merged = merged.sort_values(["material", "hyperelastic_base"]).reset_index(drop=True)
    merged.to_csv(BASE / "qlv_marmot_params.csv", index=False)
    print(f"\nsaved {BASE / 'qlv_marmot_params.csv'}")


if __name__ == "__main__":
    main()
