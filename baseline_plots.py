"""
Single-realisation baseline plots for all six methods, both scenarios.

Reproduces Figure 4.1 (the illustrative example in the main text) and the
full set of examples in Appendix B, all from the same fixed seed so the
underlying data is identical across methods within a scenario.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import partial
from pathlib import Path
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

from Methods_and_base_simulation import (
    simulate_spike, simulate_variance_change,
    w_cusum, w2_cusum, icss, bde_mean_cusum, max_dewma, aewma_huber,
    cpd_simulation,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline_plots"

ALL_METHODS = {
    "w_cusum": w_cusum, "w2_cusum": w2_cusum, "icss": icss,
    "bde_mean_cusum": bde_mean_cusum, "max_dewma": max_dewma, "aewma_huber": aewma_huber,
}
DISPLAY_NAME = {
    "w_cusum": "W-CUSUM",
    "w2_cusum": "W²-CUSUM",
    "icss": "ICSS",
    "bde_mean_cusum": "Normalised Mean CUSUM",
    "max_dewma": "MAX-DEWMA",
    "aewma_huber": "AEWMA",
}
# Same colors as the real-data event-study script
METHOD_COLOR = {
    "w_cusum":        "#1B4F8C",
    "w2_cusum":       "#5B9BD5",
    "icss":           "#1B8C5A",
    "bde_mean_cusum": "#6FBF73",
    "aewma_huber":    "#8C1B3D",
    "max_dewma":      "#E08BA0",
}

# Locked calibrated parameters (Table 4.1 / Appendix A)
PARAMS = {
    "w_cusum":         dict(zeta=0.98,  threshold=2.5235),
    "w2_cusum":        dict(zeta=0.533, threshold=6.2902),
    "icss":            dict(zeta=None,  threshold=1.7206),
    "bde_mean_cusum":  dict(zeta=None,  threshold=1.7133),
    "max_dewma":       dict(zeta=None,  threshold=3.1551),
    "aewma_huber":     dict(zeta=None,  threshold=1.2021),
}

MEAN_SHIFT_KWARGS = dict(T=500, changepoint=250, spike_size_std=2.0, duration=25, autocorrelation=0.0)
VARIANCE_KWARGS   = dict(T=500, changepoint=250, variance_multiplier=6.94, duration=25, autocorrelation=0.0)
BASELINE_SEED = 40  # default seed used throughout the existing baseline figures


def baseline_filename(method_name, scenario):
    """Matches the existing naming convention: '{Display Name} on the
    Mean-Shift Baseline.png' / '{Display Name} on the Variance-Change Baseline.png'"""
    scenario_label = "Mean-Shift Baseline" if scenario == "mean" else "Variance-Change Baseline"
    return OUTPUT_DIR / f"{DISPLAY_NAME[method_name]} on the {scenario_label}.png"


def plot_baseline(data, scores, changepoint, threshold, method, detection_time=None,
                   filename=None):
    """Two-panel plot for one method on one scenario: raw observations on
    top, the method's score below, with the true change-point,
    detection time, and threshold marked."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax1.plot(data, color="black", linewidth=0.9)
    ax1.axvline(changepoint, color="#348ABD", ls="--", linewidth=1, alpha=0.7, label="True change-point")
    if detection_time is not None:
        ax1.axvline(detection_time, color="orange", ls="--", linewidth=1.2, label="Detection")
    ax1.set_ylabel("Observation")
    ax1.minorticks_off()
    ax1.legend(frameon=False, loc="upper left")

    method_color = METHOD_COLOR[method]
    ax2.plot(scores, color=method_color, linewidth=1.2)
    ax2.axvline(changepoint, color="#348ABD", ls="--", linewidth=1, alpha=0.7)
    if detection_time is not None:
        ax2.axvline(detection_time, color="orange", ls="--", linewidth=1.2)
    ax2.axhline(threshold, color="purple", ls=":", linewidth=1, label=f"Threshold ({threshold})")
    if np.min(scores) < 0:
        ax2.axhline(-threshold, color="purple", ls=":", linewidth=1)
    ax2.set_ylabel(f"{DISPLAY_NAME[method]} Score")
    ax2.set_xlabel("Time")
    ax2.minorticks_off()
    ax2.legend(frameon=False, loc="upper left")

    if filename is not None:
        plt.savefig(filename, dpi=200, facecolor="white")
        print(f"Saved -> {filename}")
    plt.close(fig)


def run_and_plot(method_name, sim_func, sim_kwargs, scenario_label, filename):
    """Simulate one scenario at the fixed baseline seed, run the method
    over it, and save the resulting plot."""

    detector_fn = ALL_METHODS[method_name]
    params = PARAMS[method_name]
    extra_kwargs = {"threshold": params["threshold"]}
    if params["zeta"] is not None:
        extra_kwargs["zeta"] = params["zeta"]
    detector = partial(detector_fn, **extra_kwargs)

    data, changepoint = sim_func(random_seed=BASELINE_SEED, **sim_kwargs)

    scores, detection_time, delay = cpd_simulation(
        data, changepoint, detector=detector,
        initial_score={"stat": 0, "Y": 0, "Z": 0, "W": 0, "Q": 0},
    )

    plot_baseline(data, scores, changepoint, params["threshold"], method_name,
                  detection_time=detection_time, filename=filename)

    if detection_time is not None:
        print(f"{method_name} ({scenario_label}): detected at t={detection_time}, delay={delay}")
    else:
        print(f"{method_name} ({scenario_label}): no detection")

    return detection_time, delay


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for method_name in ALL_METHODS:
        run_and_plot(
            method_name, simulate_spike, MEAN_SHIFT_KWARGS, "mean-shift",
            baseline_filename(method_name, "mean"),
        )
        run_and_plot(
            method_name, simulate_variance_change, VARIANCE_KWARGS, "variance-change",
            baseline_filename(method_name, "var"),
        )