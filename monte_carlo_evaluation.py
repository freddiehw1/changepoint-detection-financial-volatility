"""General-purpose Monte Carlo evaluation of all six methods against the
mean-shift and variance-change scenarios, reporting detection rate,
premature false-alarm rate, EDD, and SD of delay."""

import numpy as np
from functools import partial
from Methods_and_base_simulation import (simulate_spike, simulate_variance_change,
                           w_cusum, w2_cusum, icss, bde_mean_cusum, max_dewma, aewma_huber)


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(base_detector, sim_func, sim_kwargs, zeta=None, threshold=None,
                     lam=None, k=None, n_reps=1000, base_seed=0):
    """Run n_reps replications of one method against one scenario.

    zeta/threshold/lam/k are passed through to the detector only if given,
    so this works for detectors that don't take all four (e.g. most
    methods ignore lam/k, only AEWMA uses them).
    """

    extra_kwargs = {"threshold": threshold}
    if zeta is not None:
        extra_kwargs["zeta"] = zeta
    if lam is not None:
        extra_kwargs["lam"] = lam
    if k is not None:
        extra_kwargs["k"] = k
    detector = partial(base_detector, **extra_kwargs)

    delays = []
    n_detected = 0
    n_premature_false_alarms = 0

    for seed in range(base_seed, base_seed + n_reps):
        data, changepoint = sim_func(random_seed=seed, **sim_kwargs)

        score = {}
        history = []
        alarm_time = None
        for t in range(len(data)):
            history.append(data[t])
            score, detected = detector(data[t], score, history)
            if detected:
                alarm_time = t
                break

        if alarm_time is not None:
            if alarm_time >= changepoint:
                n_detected += 1
                delays.append(alarm_time - changepoint)
            else:
                n_premature_false_alarms += 1
        # else: no alarm at all within the run -- counted as a miss

    detection_rate = n_detected / n_reps
    premature_fa_rate = n_premature_false_alarms / n_reps
    edd = np.mean(delays) if delays else None
    sd_delay = np.std(delays) if delays else None

    return {
        "detection_rate": detection_rate,
        "premature_fa_rate": premature_fa_rate,
        "EDD": edd,
        "SD_delay": sd_delay,
        "n_reps": n_reps,
        "n_detected": n_detected,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(title, results_dict):
    """Print a formatted results table for one scenario across all methods."""

    print(f"\n{title}")
    print(f"{'Method':<20} {'Detection Rate':>15} {'Premature FA':>13} {'EDD':>10} {'SD Delay':>10}")
    print("-" * 72)
    for name, r in results_dict.items():
        edd_str = f"{r['EDD']:.2f}" if r['EDD'] is not None else "—"
        sd_str = f"{r['SD_delay']:.2f}" if r['SD_delay'] is not None else "—"
        flag = "  *" if r['detection_rate'] < 0.80 else ""
        print(f"{name:<20} {r['detection_rate']*100:>14.1f}% {r['premature_fa_rate']*100:>12.1f}% "
              f"{edd_str:>10} {sd_str:>10}{flag}")
    print("  * detection rate < 80%: EDD unreliable per project spec")


# ---------------------------------------------------------------------------
# Locked calibrated parameters
# ---------------------------------------------------------------------------

PARAMS = {
    "w_cusum":         dict(zeta=0.98,   threshold=2.5235),
    "w2_cusum":        dict(zeta=0.533, threshold=6.2902),
    "icss":            dict(zeta=None,   threshold=1.7206),
    "bde_mean_cusum":  dict(zeta=None,   threshold=1.7133),
    "max_dewma":       dict(zeta=None,   threshold=3.1551),
    "aewma_huber":     dict(zeta=None,   threshold=1.2021)
}

N_REPS = 5000


# ---------------------------------------------------------------------------
# Full cross-comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mean_shift_kwargs = dict(T=500, changepoint=250, spike_size_std=2.0,
                              duration=25, autocorrelation=0.0)
    variance_kwargs = dict(T=500, changepoint=250, variance_multiplier=6.94,
                            duration=25, autocorrelation=0.0)

    all_methods = [
        ("w_cusum", w_cusum),
        ("w2_cusum", w2_cusum),
        ("icss", icss),
        ("bde_mean_cusum", bde_mean_cusum),
        ("max_dewma", max_dewma),
        ("aewma_huber", aewma_huber)
    ]

    mean_results = {}
    variance_results = {}

    for name, detector in all_methods:
        print(name, detector)
        mean_results[name] = run_monte_carlo(
            base_detector=detector,
            sim_func=simulate_spike,
            sim_kwargs=mean_shift_kwargs,
            zeta=PARAMS[name]["zeta"],
            threshold=PARAMS[name]["threshold"],
            n_reps=N_REPS
        )
        variance_results[name] = run_monte_carlo(
            base_detector=detector,
            sim_func=simulate_variance_change,
            sim_kwargs=variance_kwargs,
            zeta=PARAMS[name]["zeta"],
            threshold=PARAMS[name]["threshold"],
            n_reps=N_REPS
        )

    print_table(f"=== All methods on mean-shift scenario (duration={mean_shift_kwargs['duration']}) ===", mean_results)
    print_table(f"=== All methods on variance-change scenario (duration={variance_kwargs['duration']}) ===", variance_results)