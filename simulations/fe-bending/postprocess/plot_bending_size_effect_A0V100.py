"""Size-effect check for the A0V100 (pure Vero) 3-point-bending FE
predictions: load-time curves with the load scaled by crack length `a`
(F/a, N/mm), one line per crack-length variant (a5/a10/a20mm). Companion
to plot_bending_load_time_A0V100.py (same data, unscaled).

Why F/a is the CONTINUUM-BEAM prediction (not F/a^2 -- see
plot_bending_size_effect_a2_A0V100.py's docstring for that script's own
role): only ONE cross-section dimension scales with a here -- the beam's
depth in the bending (X) direction, h=a -- while its width (Z) is the
fixed 5mm unit-cell thickness, b=const, and its span is Ly=4a. The applied
midspan deflection is also proportional to a (delta=0.1*a). For an
equivalent SOLID beam in 3-point bending (delta = P*L^3/(48EI)), with
I = b*h^3/12 = b*a^3/12 (a^3, not a^4, since b is fixed):
    P = 48*E*I*delta / L^3 = 48*E*(b*a^3/12)*(k*a) / (4a)^3
      = (48*b*k/768) * E * a   ~  a^1
So a solid-beam-of-this-shape prediction is F ~ a, i.e. F/a should collapse
onto a single curve -- NOT F/a^2 (that would be the prediction only if BOTH
cross-section dimensions scaled with a, e.g. a square h=b=a cross-section,
which this specimen does not have). This lattice is not actually solid
though (strut slenderness and unit-cell count both change with a in a way
a homogeneous-beam model doesn't capture), so F/a collapsing only
approximately -- and converging toward the collapse as a grows (i.e. as
the lattice contains more unit cells across its height and better
approximates a homogenized continuum) -- is itself the interesting result,
not a bug in the F/a hypothesis.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from plotstyle import colors  # noqa: E402

HERE = Path(__file__).parent
JOB_BASE = HERE.parent / "jobs" / "edelweissfe_bending_A0V100"
SIZES = [5, 10, 20]
TOTAL_TIME = 600.0  # RAMP_TIME + HOLD_TIME, generate_bending_jobs_A0V100.py
# Row-count thresholds are NOT reliable here: EdelweissFE writes duplicate
# t=0 rows at each step boundary (2 steps -> 3 zero rows, not 1), so a
# completed run has 44 rows, not "2*20+1=41" -- checking the final TIME
# value instead of a row count avoids re-learning this the hard way.


def load_size(a):
    path = JOB_BASE / f"a{a}mm" / "loadForce.csv"
    if not path.exists():
        return None
    data = np.loadtxt(path)
    t, idx = np.unique(data[:, 0], return_index=True)
    force = data[idx, 1]
    complete = len(t) > 0 and t[-1] >= TOTAL_TIME - 1.0
    return t, force, complete


fig, ax = plt.subplots(figsize=(7, 5))
for i, a in enumerate(SIZES):
    result = load_size(a)
    if result is None:
        print(f"a{a}mm: no loadForce.csv yet, skipping")
        continue
    t, force, complete = result
    ls = "-" if complete else "--"
    label = f"a={a}mm" if complete else f"a={a}mm (in progress, t={t[-1]:.0f}/{TOTAL_TIME:.0f}s)"
    ax.plot(t, force / a, color=colors[i], linestyle=ls, linewidth=1.3, marker="o",
            markersize=2.5, markevery=2, label=label)

ax.set_xlabel("Time in s")
ax.set_ylabel("Load / a in N/mm")
ax.set_title("A0V100 (pure Vero) cubic-lattice 3-point-bending -- size effect (F/a)")
ax.legend(fontsize=9)
fig.tight_layout()
out_path = HERE / "bending_size_effect_A0V100.png"
fig.savefig(out_path, dpi=200)
print(f"saved {out_path}")
