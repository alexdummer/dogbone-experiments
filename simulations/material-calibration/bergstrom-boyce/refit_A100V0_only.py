"""Refit just A100V0 (all 4 hyperelastic bases) with the raised tau floor,
and merge the result into qlv_marmot_params.csv (replacing only its rows) --
avoids re-running the other 4 materials' fits, which are deterministic and
unaffected by this change."""
import pandas as pd
from pathlib import Path

from fit_qlv_marmot_model import BASES, fit_joint
from fit_qlv_model import MATERIALS, estimate_youngs_modulus
import numpy as np

BASE = Path(__file__).parent
material = "A100V0"
cfg = MATERIALS[material]

prony = pd.read_csv(BASE / "qlv_prony_params.csv", index_col=0)
E_mean = np.mean(list(estimate_youngs_modulus(cfg, material).values()))
G0 = E_mean / 3.0
g0 = [float(prony.loc[material, f"g{i}"]) for i in (1, 2, 3)]
tau0 = [float(prony.loc[material, f"tau{i}"]) for i in (1, 2, 3)]

rows = []
for base_name in BASES:
    print(f"=== refitting {material}, {base_name} (tau_lb=0.1) ===", flush=True)
    params = fit_joint(material, cfg, base_name, G0, g0, tau0, verbose=False, tau_lb_value=0.1)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}, flush=True)
    row = {"material": material}
    row.update(params)
    rows.append(row)

new_rows = pd.DataFrame(rows)

full = pd.read_csv(BASE / "qlv_marmot_params.csv")
full = full[full["material"] != material]
full = pd.concat([full, new_rows], ignore_index=True)
full.to_csv(BASE / "qlv_marmot_params.csv", index=False)
print(f"\nupdated {BASE / 'qlv_marmot_params.csv'} ({material} rows replaced)")
