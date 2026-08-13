"""
Published/asymptotic thresholds vs. calibrated thresholds, false-alarm rate only.
Uses the exact same noise-generation and seeding scheme as run_monte_carlo.py:
simulate_spike with spike_size_std=0 (no injected change), random_seed = 0..NREPS-1.

Reproduces the comparisons discussed in Section 4.1 of the thesis: for each
method with a published, non-simulation-derived threshold, this checks the
empirical false-alarm rate that threshold gives under this thesis's
fixed-horizon (T=500) setting, against the empirically calibrated threshold
actually used throughout the rest of the analysis.
"""
import numpy as np
from functools import partial

from Methods_and_base_simulation import (
    simulate_spike,
    icss,
    bde_mean_cusum,
    aewma_huber,
    max_dewma,
)

T = 500
NREPS = 2000
BASE_SEED = 0


def fa_rate(base_detector, zeta=None, threshold=None, lam=None, k=None,
            n_reps=NREPS, base_seed=BASE_SEED):
    """Empirical false-alarm rate for one method/threshold setting under
    pure noise (no injected change)."""

    extra_kwargs = {"threshold": threshold}
    if zeta is not None:
        extra_kwargs["zeta"] = zeta
    if lam is not None:
        extra_kwargs["lam"] = lam
    if k is not None:
        extra_kwargs["k"] = k
    detector = partial(base_detector, **extra_kwargs)

    n_fired = 0
    for seed in range(base_seed, base_seed + n_reps):
        data, _ = simulate_spike(
            T=T, changepoint=T + 1, spike_size_std=0.0, duration=0,
            autocorrelation=0.0, random_seed=seed,
        )
        score = {}
        history = []
        for t in range(len(data)):
            history.append(data[t])
            score, detected = detector(data[t], score, history)
            if detected:
                n_fired += 1
                break
    return 100 * n_fired / n_reps


if __name__ == "__main__":
    rows = [
        ("AEWMA (published, full triple lam=.1253 k=2.7765 h=.8238)",
         aewma_huber, dict(threshold=0.8238, lam=0.1253, k=2.7765)),
        ("AEWMA (published h only, thesis lam=.10 k=3)",
         aewma_huber, dict(threshold=0.8238, lam=0.10, k=3.0)),
        ("AEWMA (calibrated, thesis lam=.10 k=3, h=1.2021)",
         aewma_huber, dict(threshold=1.2021, lam=0.10, k=3.0)),

        ("Max-DEWMA (published K2=2.969, ARL0=250)",
         max_dewma, dict(threshold=2.969)),
        ("Max-DEWMA (calibrated h=3.1551)",
         max_dewma, dict(threshold=3.1551)),

        ("ICSS (published D*=1.358)",
         icss, dict(threshold=1.358)),
        ("ICSS (calibrated h=1.7206)",
         icss, dict(threshold=1.7206)),

        ("Norm-Mean-CUSUM (published D*=1.358, shared w/ ICSS)",
         bde_mean_cusum, dict(threshold=1.358)),
        ("Norm-Mean-CUSUM (calibrated h=1.7133)",
         bde_mean_cusum, dict(threshold=1.7133)),
    ]

    print(f"{'Setting':<60}{'FA rate':>10}")
    print("-" * 70)
    for name, detector, kwargs in rows:
        rate = fa_rate(detector, **kwargs)
        print(f"{name:<60}{rate:>9.2f}%")