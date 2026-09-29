"""Extract storage (E') and loss (E'') moduli from the A0V100 cyclic tests
(7, 8, 9), correcting for the underlying stress relaxation rather than
treating the cyclic signal as a clean oscillation about a fixed mean.

Specimen 9 required two retests before a clean run was captured (see
clean_data.py) -- treat its results with a bit more caution than 7/8.

Method:
1. Fit a normalized 3-term Prony relaxation shape to the fast-rate
   relaxation tests (4, 5, 6), which share the cyclic tests' fast
   (~1 s) initial ramp.
2. For each cyclic test, scale that shape (one amplitude parameter) to
   the test's own cycle-mean stress trend -- this is the "relaxation
   baseline" the cyclic oscillation rides on.
3. Subtract the baseline from the raw stress to get the residual
   (detrended) oscillatory stress.
4. Per cycle, fit both strain and residual stress to a sine + local
   linear drift at the cycle's own angular frequency, extract amplitude
   and phase, and compute E* = stress_amp / strain_amp, E' = |E*| cos(delta),
   E'' = |E*| sin(delta), where delta is the stress-strain phase lag.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent))
from plotstyle import colors, figsize_single  # noqa: E402

CLEANED_DIR = Path(__file__).parent / "cleaned"

WIDTH_MM = 6.0
THICKNESS_MM = 2.0
AREA_MM2 = WIDTH_MM * THICKNESS_MM

RELAXATION_TESTS = [4, 5, 6]
CYCLIC_TESTS = {7: colors[0], 8: colors[1], 9: colors[2]}
CAUTION_TESTS = {9}
PROMINENCE_MM = 0.02
TIME_GRID = np.logspace(np.log10(0.05), np.log10(590), 300)

# the cyclic protocol is a frequency sweep: ~20 cycles at each nominal
# frequency in turn (this material's sweep starts lower, at 0.05 Hz); the
# first few cycles after a frequency change are a settling transient and
# excluded from the steady-state summary
NOMINAL_FREQS_HZ = [0.05, 0.5, 1, 2, 5]
TRANSIENT_CYCLES = 5


def label_for(test_id):
    return f"test {test_id}" + (" (caution)" if test_id in CAUTION_TESTS else "")


def assign_nominal_frequency(freq_hz):
    log_f = np.log(freq_hz)
    log_nominal = np.log(NOMINAL_FREQS_HZ)
    return NOMINAL_FREQS_HZ[np.argmin(np.abs(log_f - log_nominal))]


def find_ramp_end_index(position):
    n = len(position)
    hold_value = position.iloc[-max(1, n // 10):].median()
    return (position >= 0.99 * hold_value).to_numpy().nonzero()[0][0]


def prony3(t, g_inf, g1, tau1, g2, tau2, g3, tau3):
    return g_inf + g1 * np.exp(-t / tau1) + g2 * np.exp(-t / tau2) + g3 * np.exp(-t / tau3)


def fit_relaxation_shape():
    curves = []
    for test_id in RELAXATION_TESTS:
        df = pd.read_csv(CLEANED_DIR / f"A0V100-{test_id}_cleaned.csv")
        stress = -df["force_N"] / AREA_MM2
        idx_end = find_ramp_end_index(df["position_mm"])
        t_rel = df["time_s"].to_numpy() - df["time_s"].iloc[idx_end]
        mask = t_rel > 0
        curves.append(np.interp(TIME_GRID, t_rel[mask], stress[mask] / stress.iloc[idx_end]))
    avg_curve = np.mean(curves, axis=0)

    p0 = [0.4, 0.2, 1.0, 0.2, 10.0, 0.2, 100.0]
    bounds = (0, [1, 1, 1e4, 1, 1e4, 1, 1e4])
    popt, _ = curve_fit(prony3, TIME_GRID, avg_curve, p0=p0, bounds=bounds, maxfev=20000)
    return popt


def cycle_mean_trend(df, t_end, troughs):
    cyc_t, cyc_s = [], []
    for k in range(len(troughs) - 1):
        sl = slice(troughs[k], troughs[k + 1] + 1)
        cyc_t.append(df["time_s"].iloc[sl].mean() - t_end)
        cyc_s.append(df["stress"].iloc[sl].mean())
    return np.array(cyc_t), np.array(cyc_s)


def fit_baseline_amplitude(shape_params, cyc_t, cyc_s):
    mask = cyc_t > 0

    def model(t, amp):
        return amp * prony3(t, *shape_params)

    popt, _ = curve_fit(model, cyc_t[mask], cyc_s[mask], p0=[cyc_s[mask][0]])
    return popt[0]


def sine_model(t, w):
    def f(t, c0, c1, amp, phi):
        return c0 + c1 * t + amp * np.sin(w * t + phi)
    return f


def normalize_amp_phase(amp, phi):
    """curve_fit's sine amplitude/phase are only unique up to (amp, phi) ->
    (-amp, phi+pi); force amp >= 0 so phase differences are meaningful."""
    if amp < 0:
        amp = -amp
        phi = phi + np.pi
    return amp, (phi + np.pi) % (2 * np.pi) - np.pi


def fit_cycle_moduli(df, resid_stress, troughs):
    t_all = df["time_s"].to_numpy()
    strain = df["strain"].to_numpy()

    rows = []
    for k in range(len(troughs) - 1):
        i0, i1 = troughs[k], troughs[k + 1]
        t_loc = t_all[i0:i1 + 1] - t_all[i0]
        period = t_loc[-1]
        w = 2 * np.pi / period
        model = sine_model(t_loc, w)

        eps = strain[i0:i1 + 1]
        sig = resid_stress[i0:i1 + 1]
        p0_e = [eps.mean(), 0, (eps.max() - eps.min()) / 2, 0]
        p0_s = [sig.mean(), 0, (sig.max() - sig.min()) / 2, 0]
        popt_e, _ = curve_fit(model, t_loc, eps, p0=p0_e)
        popt_s, _ = curve_fit(model, t_loc, sig, p0=p0_s)

        r2_e = 1 - np.var(eps - model(t_loc, *popt_e)) / np.var(eps)
        r2_s = 1 - np.var(sig - model(t_loc, *popt_s)) / np.var(sig)

        eps_amp, phi_e = normalize_amp_phase(popt_e[2], popt_e[3])
        sig_amp, phi_s = normalize_amp_phase(popt_s[2], popt_s[3])
        delta = (phi_s - phi_e + np.pi) % (2 * np.pi) - np.pi

        e_star = sig_amp / eps_amp
        freq_hz = 1 / period
        rows.append(
            {
                "cycle": k,
                "cycle_time_s": t_all[i0:i1 + 1].mean(),
                "frequency_hz": freq_hz,
                "nominal_frequency_hz": assign_nominal_frequency(freq_hz),
                "strain_amplitude": eps_amp,
                "stress_amplitude_mpa": sig_amp,
                "phase_lag_deg": np.degrees(delta),
                "E_star_mpa": e_star,
                "E_storage_mpa": e_star * np.cos(delta),
                "E_loss_mpa": e_star * np.sin(delta),
                "tan_delta": np.tan(delta),
                "fit_r2_min": min(r2_e, r2_s),
            }
        )
    df_out = pd.DataFrame(rows)
    reliable = df_out["fit_r2_min"] > 0.9
    reliable &= df_out["strain_amplitude"] > 0.2 * df_out["strain_amplitude"].median()
    df_out["reliable"] = reliable

    cycle_in_block = df_out.groupby("nominal_frequency_hz").cumcount()
    df_out["steady_state"] = df_out["reliable"] & (cycle_in_block >= TRANSIENT_CYCLES)
    return df_out


def process_test(test_id, shape_params):
    df = pd.read_csv(CLEANED_DIR / f"A0V100-{test_id}_cyclic_cleaned.csv")
    df["stress"] = -df["force_N"] / AREA_MM2
    peaks, _ = find_peaks(df["position_mm"], prominence=PROMINENCE_MM)
    troughs, _ = find_peaks(-df["position_mm"], prominence=PROMINENCE_MM)
    t_end = df["time_s"].iloc[peaks[0]]

    cyc_t, cyc_s = cycle_mean_trend(df, t_end, troughs)
    amplitude = fit_baseline_amplitude(shape_params, cyc_t, cyc_s)

    t_all = df["time_s"].to_numpy()
    baseline = np.where(
        t_all > t_end, amplitude * prony3(t_all - t_end, *shape_params), np.nan
    )
    resid_stress = df["stress"].to_numpy() - baseline

    moduli = fit_cycle_moduli(df, resid_stress, troughs)
    return df, baseline, moduli


def summarize_by_frequency(moduli, test_id):
    steady = moduli[moduli["steady_state"]]
    summary = steady.groupby("nominal_frequency_hz").agg(
        E_storage_mean=("E_storage_mpa", "mean"),
        E_storage_std=("E_storage_mpa", "std"),
        E_loss_mean=("E_loss_mpa", "mean"),
        E_loss_std=("E_loss_mpa", "std"),
        tan_delta_mean=("tan_delta", "mean"),
        tan_delta_std=("tan_delta", "std"),
        n_cycles=("tan_delta", "count"),
    ).reset_index()
    summary.insert(0, "test", test_id)
    return summary


def main():
    shape_params = fit_relaxation_shape()
    print("relaxation shape params (g_inf, g1,tau1, g2,tau2, g3,tau3):", shape_params)

    fig, ((ax_baseline, ax_estar), (ax_moduli, ax_tandelta)) = plt.subplots(
        2, 2, figsize=(2 * figsize_single[0], 2 * figsize_single[1])
    )

    all_moduli = {}
    for test_id, color in CYCLIC_TESTS.items():
        df, baseline, moduli = process_test(test_id, shape_params)
        moduli.to_csv(Path(__file__).parent / f"A0V100-{test_id}_dma_moduli.csv", index=False)
        all_moduli[test_id] = moduli

        n_dropped = (~moduli["reliable"]).sum()
        if n_dropped:
            print(f"test {test_id}: dropped {n_dropped}/{len(moduli)} low-confidence cycles (poor sine fit or collapsed amplitude)")
        good = moduli[moduli["reliable"]]

        ax_baseline.plot(df["time_s"], df["stress"], color=color, alpha=0.4, linewidth=0.6)
        ax_baseline.plot(df["time_s"], baseline, color=color, linewidth=1.5, label=f"{label_for(test_id)} baseline")

        ax_estar.plot(good["cycle"], good["E_star_mpa"], color=color, marker=".", markersize=3, label=label_for(test_id))

        ax_moduli.plot(good["cycle"], good["E_storage_mpa"], color=color, linestyle="-", label=f"{label_for(test_id)} E'")
        ax_moduli.plot(good["cycle"], good["E_loss_mpa"], color=color, linestyle="--", label=f"{label_for(test_id)} E''")

        ax_tandelta.plot(good["cycle"], good["tan_delta"], color=color, marker=".", markersize=3, label=label_for(test_id))

    for ax in (ax_estar, ax_moduli, ax_tandelta):
        block_bounds = all_moduli[7].groupby("nominal_frequency_hz")["cycle"].agg(["min", "max"])
        for freq, row in block_bounds.iterrows():
            ax.axvline(row["min"], color="grey", linewidth=0.5, linestyle=":")

    ax_baseline.set_xlabel("Time (s)")
    ax_baseline.set_ylabel("Stress (MPa)")
    ax_baseline.legend(fontsize=6)
    ax_baseline.grid(True, alpha=0.4)
    ax_baseline.set_title("Raw stress + fitted relaxation baseline", fontsize=9)

    ax_estar.set_xlabel("Cycle number")
    ax_estar.set_ylabel("|E*| (MPa)")
    ax_estar.legend(fontsize=6)
    ax_estar.grid(True, alpha=0.4)
    ax_estar.set_title("Dotted lines: frequency-sweep block boundaries", fontsize=8)

    ax_moduli.set_xlabel("Cycle number")
    ax_moduli.set_ylabel("Modulus (MPa)")
    ax_moduli.legend(fontsize=6)
    ax_moduli.grid(True, alpha=0.4)

    ax_tandelta.set_xlabel("Cycle number")
    ax_tandelta.set_ylabel(r"$\tan\delta$")
    ax_tandelta.legend(fontsize=6)
    ax_tandelta.grid(True, alpha=0.4)

    fig.tight_layout()
    out_path = Path(__file__).parent / "dma_moduli.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.png')}")

    # frequency-sweep summary (the actual DMA deliverable)
    summaries = pd.concat(
        [summarize_by_frequency(all_moduli[t], t) for t in CYCLIC_TESTS], ignore_index=True
    )
    summaries.to_csv(Path(__file__).parent / "dma_frequency_sweep_summary.csv", index=False)

    fig2, (ax_mod, ax_tan) = plt.subplots(1, 2, figsize=(2 * figsize_single[0], figsize_single[1]))
    for test_id, color in CYCLIC_TESTS.items():
        s = summaries[summaries["test"] == test_id]
        ax_mod.errorbar(s["nominal_frequency_hz"], s["E_storage_mean"], yerr=s["E_storage_std"],
                         color=color, marker="o", linestyle="-", label=f"{label_for(test_id)} E'")
        ax_mod.errorbar(s["nominal_frequency_hz"], s["E_loss_mean"], yerr=s["E_loss_std"],
                         color=color, marker="s", linestyle="--", label=f"{label_for(test_id)} E''")
        ax_tan.errorbar(s["nominal_frequency_hz"], s["tan_delta_mean"], yerr=s["tan_delta_std"],
                         color=color, marker="o", linestyle="-", label=label_for(test_id))

    ax_mod.set_xscale("log")
    ax_mod.set_xlabel("Frequency (Hz)")
    ax_mod.set_ylabel("Modulus (MPa)")
    ax_mod.legend(fontsize=7)
    ax_mod.grid(True, alpha=0.4)

    ax_tan.set_xscale("log")
    ax_tan.set_xlabel("Frequency (Hz)")
    ax_tan.set_ylabel(r"$\tan\delta$")
    ax_tan.legend(fontsize=7)
    ax_tan.grid(True, alpha=0.4)

    fig2.tight_layout()
    out_path2 = Path(__file__).parent / "dma_frequency_sweep.pdf"
    fig2.savefig(out_path2, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {out_path2} and {out_path2.with_suffix('.png')}")


if __name__ == "__main__":
    main()
