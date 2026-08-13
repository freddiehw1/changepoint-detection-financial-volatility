"""
Threshold calibration via binary search.

Reproduces the procedure described in Appendix A: for a given method, the
control limit is searched over until the empirical false-alarm rate under
2,000 replications of pure noise (T=500) matches the 5% target. Any
reference value (zeta) is fixed at its analytically-derived value and
passed straight through, not searched over.
"""

import numpy as np
from functools import partial
from Methods_and_base_simulation import simulate_spike, w_cusum, w2_cusum, icss, bde_mean_cusum, max_dewma, aewma_huber


# ---------------------------------------------------------------------------
# False-alarm rate estimation
# ---------------------------------------------------------------------------

def false_alarm_rate(base_detector, threshold, zeta=None, T=500, n_reps=2000, base_seed=0):
    """
    Estimates the false alarm rate at a given threshold: the fraction of
    T-length pure-noise replications (simulate_spike with spike_size_std=0.0,
    i.e. no actual changepoint) on which the detector fires at least once.

    zeta is optional -- only forwarded if the detector actually has a zeta
    parameter (w_cusum, w2_cusum). Methods without a drift term (icss,
    bde_mean_cusum) just take threshold.
    """
    extra_kwargs = {"threshold": threshold}
    if zeta is not None:
        extra_kwargs["zeta"] = zeta

    detector = partial(base_detector, **extra_kwargs)
    n_false_alarms = 0

    for seed in range(base_seed, base_seed + n_reps):
        data, _ = simulate_spike(T=T, spike_size_std=0.0, random_seed=seed)

        score = {}
        history = []
        fired = False
        for t in range(T):
            history.append(data[t])
            score, detected = detector(data[t], score, history)
            if detected:
                fired = True
                break   # only need to know IF it fired, not when, for this metric

        if fired:
            n_false_alarms += 1

    return n_false_alarms / n_reps


# ---------------------------------------------------------------------------
# Binary search
# ---------------------------------------------------------------------------

def calibrate_threshold_binary_search(base_detector, zeta=None, target_rate=0.05,
                                        T=500, n_reps=2000,
                                        low=0.1, high=30.0,
                                        tol=0.001, max_iter=20, base_seed=0):
    """
    Binary searches threshold until the empirical false alarm rate over
    n_reps replications of T-length pure noise is within `tol` of
    target_rate (default 5%).

    Assumes false alarm rate is monotonically decreasing in threshold
    (higher threshold = harder to trigger = fewer false alarms).
    """
    for iteration in range(max_iter):
        mid = (low + high) / 2
        rate = false_alarm_rate(base_detector, mid, zeta=zeta, T=T, n_reps=n_reps, base_seed=base_seed)

        print(f"iter {iteration+1:2d}: threshold={mid:.4f}  "
              f"false_alarm_rate={rate:.4f}  (target={target_rate})")

        if abs(rate - target_rate) <= tol:
            print(f"\nConverged: threshold={mid:.4f} gives false_alarm_rate={rate:.4f}")
            return mid, rate

        if rate > target_rate:
            # too many false alarms -> threshold too low -> raise it
            low = mid
        else:
            # too few false alarms -> threshold too high -> lower it
            high = mid

    print(f"\nMax iterations reached. Best estimate: threshold={mid:.4f}, "
          f"false_alarm_rate={rate:.4f}")
    return mid, rate


# ---------------------------------------------------------------------------
# Run all six calibrations
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=== Calibrating w_cusum")
    calibrate_threshold_binary_search(
        base_detector=w_cusum,
        zeta=0.98,
        target_rate=0.05,
        T=500,
        n_reps=2000
    )

    print("\n=== Calibrating w2_cusum")
    calibrate_threshold_binary_search(
        base_detector=w2_cusum,
        zeta=0.533,
        target_rate=0.05,
        T=500,
        n_reps=2000
    )

    print("\n=== Calibrating icss")
    calibrate_threshold_binary_search(
        base_detector=icss,
        target_rate=0.05,
        T=500,
        n_reps=2000
    )

    print("\n=== Calibrating normalised mean cusum")
    calibrate_threshold_binary_search(
        base_detector=bde_mean_cusum,
        target_rate=0.05,
        T=500,
        n_reps=2000
    )

    print("\n=== Calibrating max_dewma")
    calibrate_threshold_binary_search(
        base_detector=max_dewma,
        target_rate=0.05,
        T=500,
        n_reps=2000,
        low=0.1,
        high=10.0
    )

    print("\n=== Calibrating AEWMA")
    calibrate_threshold_binary_search(
        base_detector=aewma_huber,
        target_rate=0.05,
        T=500,
        n_reps=2000,
        low=0.1,
        high=10.0,
        tol=0.001,
        max_iter=20
    )