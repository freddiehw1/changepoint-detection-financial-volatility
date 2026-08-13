"""Shift-magnitude sweep: detection rate and EDD for all six methods
across a range of mean-shift and KL-matched variance-change magnitudes."""

import numpy as np
from functools import partial
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import scienceplots
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 16,
    "legend.fontsize": 10,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

from pathlib import Path
from Methods_and_base_simulation import (
    simulate_spike, simulate_variance_change,
    w_cusum, w2_cusum, icss, bde_mean_cusum, max_dewma, aewma_huber,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T = 500
CHANGEPOINT = 250
DURATION = 25
N_REPS = 5000
BASE_SEED = 0

MEAN_SHIFT_MAGNITUDES = list(np.round(np.arange(0.25, 5.01, 0.25), 2))   # 20 points, 0.25 step

ERROR_BAR_MODE = "binomial_se"

METHOD_STYLE = {
    # Rank-based CUSUM pair (Lombard & van Zyl) -- blue family
    "W-CUSUM":                dict(color="#1B4F8C", marker="o"),
    "W2-CUSUM":                dict(color="#5B9BD5", marker="s"),
    # Retrospective/offline pair -- green family
    "ICSS":                    dict(color="#1B8C5A", marker="^"),
    "Normalised mean CUSUM":   dict(color="#6FBF73", marker="D"),
    # EWMA-based pair -- red/pink family
    "AEWMA":                   dict(color="#8C1B3D", marker="P"),
    "MAX-DEWMA":               dict(color="#E08BA0", marker="v"),
}
DISPLAY_LABEL = {
    "W2-CUSUM": "W\u00b2-CUSUM",
}

METHOD_FUNCS = {
    "W-CUSUM": w_cusum, "W2-CUSUM": w2_cusum, "ICSS": icss,
    "Normalised mean CUSUM": bde_mean_cusum, "MAX-DEWMA": max_dewma, "AEWMA": aewma_huber,
}

PARAMS = {
    "W-CUSUM":                dict(zeta=0.98,  threshold=2.5235),
    "W2-CUSUM":                dict(zeta=0.533, threshold=6.2902),
    "ICSS":                    dict(zeta=None,  threshold=1.7206),
    "Normalised mean CUSUM":   dict(zeta=None,  threshold=1.7133),
    "MAX-DEWMA":               dict(zeta=None,  threshold=3.1551),
    "AEWMA":                   dict(zeta=None,  threshold=1.2021),
}


# ---------------------------------------------------------------------------
# KL-divergence matching
# ---------------------------------------------------------------------------

def kl_mean_shift(delta):
    """KL divergence of a N(delta, 1) shift from N(0, 1)."""
    return delta ** 2 / 2

def kl_variance_ratio(vm):
    """KL divergence of a N(0, vm) shift from N(0, 1), vm = post/pre variance ratio."""
    return -0.5 * np.log(vm) + vm / 2 - 0.5

def kl_matched_variance_multiplier(delta):
    """Solve for the variance multiplier with the same KL divergence as a
    mean shift of size delta."""
    target_kl = kl_mean_shift(delta)
    return brentq(lambda vm: kl_variance_ratio(vm) - target_kl, 1.0 + 1e-6, 200)


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(base_detector, sim_func, sim_kwargs, zeta=None, threshold=None,
                     n_reps=N_REPS, base_seed=BASE_SEED):
    """Run n_reps replications of one method against one scenario and
    return detection rate, EDD, SD of delay, and premature false-alarm rate."""

    extra_kwargs = {"threshold": threshold}
    if zeta is not None:
        extra_kwargs["zeta"] = zeta
    detector = partial(base_detector, **extra_kwargs)

    delays = []
    n_detected = 0
    n_premature_false_alarms = 0

    for seed in range(base_seed, base_seed + n_reps):
        data, changepoint = sim_func(random_seed=seed, **sim_kwargs)
        score, history, alarm_time = {}, [], None
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

    detection_rate = n_detected / n_reps
    edd = np.mean(delays) if delays else None
    sd_delay = np.std(delays) if delays else None

    return dict(detection_rate=detection_rate, EDD=edd, SD_delay=sd_delay,
                n_reps=n_reps, n_detected=n_detected,
                premature_fa_rate=n_premature_false_alarms / n_reps)


# ---------------------------------------------------------------------------
# Error bar helpers
# ---------------------------------------------------------------------------

def detection_rate_error(result):
    """Binomial standard error of the detection rate."""
    p = result["detection_rate"]
    n = result["n_reps"]
    return np.sqrt(p * (1 - p) / n)

def edd_error(result, mode):
    """Error bar for EDD: either the raw SD of delay, or the SD scaled
    down to a standard error of the mean (mode='sd' vs anything else)."""
    if result["SD_delay"] is None or result["EDD"] is None:
        return None
    if mode == "sd":
        return result["SD_delay"]
    if result["n_detected"] == 0:
        return None
    return result["SD_delay"] / np.sqrt(result["n_detected"])


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep():
    """Run every method against every mean-shift magnitude and its
    KL-matched variance-change counterpart, for both scenarios."""

    results = {"mean_shift": {}, "variance_change": {}}
    for name, detector in METHOD_FUNCS.items():
        results["mean_shift"][name] = []
        results["variance_change"][name] = []
        for delta in MEAN_SHIFT_MAGNITUDES:
            vm = kl_matched_variance_multiplier(delta)

            mean_kwargs = dict(T=T, changepoint=CHANGEPOINT, spike_size_std=delta,
                                duration=DURATION, autocorrelation=0.0)
            var_kwargs = dict(T=T, changepoint=CHANGEPOINT, variance_multiplier=vm,
                               duration=DURATION, autocorrelation=0.0)

            r_mean = run_monte_carlo(detector, simulate_spike, mean_kwargs,
                                      zeta=PARAMS[name]["zeta"], threshold=PARAMS[name]["threshold"])
            r_var = run_monte_carlo(detector, simulate_variance_change, var_kwargs,
                                     zeta=PARAMS[name]["zeta"], threshold=PARAMS[name]["threshold"])

            r_mean["x"] = delta
            r_var["x"] = delta
            r_var["vm"] = vm
            results["mean_shift"][name].append(r_mean)
            results["variance_change"][name].append(r_var)

            print(f"{name}: delta={delta:.2f} (KL={kl_mean_shift(delta):.3f}) -> "
                  f"vm={vm:.3f} | mean_shift det={r_mean['detection_rate']*100:.1f}% "
                  f"var_change det={r_var['detection_rate']*100:.1f}%")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stacked(results, y_field, y_label, title_mean, title_var, filename, error_mode=ERROR_BAR_MODE):
    """Two-panel stacked plot (mean-shift on top, variance-change below)
    of either detection rate or EDD against shift magnitude, one line per
    method with shaded error bands."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), sharey=(y_field == "detection_rate"))

    EDD_FLOOR = 0.1

    for ax, scenario, title in [(ax1, "mean_shift", title_mean), (ax2, "variance_change", title_var)]:
        for name in METHOD_STYLE:
            rows = results[scenario][name]
            x = [r["x"] for r in rows]
            if y_field == "detection_rate":
                y = np.array([r["detection_rate"] * 100 for r in rows])
            else:
                y = np.array([r["EDD"] if r["EDD"] is not None else np.nan for r in rows])

            if error_mode is not None:
                if y_field == "detection_rate":
                    yerr = np.array([detection_rate_error(r) * 100 for r in rows])
                else:
                    yerr = np.array([edd_error(r, error_mode) for r in rows])
                has_err = not any(e is None for e in yerr)
            else:
                has_err = False

            style = METHOD_STYLE[name]

            ax.plot(x, y, color=style["color"], label=DISPLAY_LABEL.get(name, name), linewidth=2)
            if has_err:
                lower = np.maximum(y - yerr, EDD_FLOOR) if y_field == "EDD" else y - yerr
                ax.fill_between(x, lower, y + yerr, color=style["color"], alpha=0.15, linewidth=0)

        if y_field == "detection_rate":
            ax.set_ylim(-3, 103)
        else:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda y, pos: f"{y:g}"
            ))

        ax.axvline(2.0, color="#8a8a8a", linestyle=":", linewidth=1.3, zorder=0)

        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, loc="left", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)

    ax1.set_xticks([0.25, 1, 2, 3, 4, 5])

    tick_deltas = [0.25, 1.0, 2.0, 3.0, 4.0, 5.0]
    tick_vms = []
    for d in tick_deltas:
        vm = kl_matched_variance_multiplier(d)
        tick_vms.append(f"{vm:.1f}")
    ax2.set_xticks(tick_deltas)
    ax2.set_xticklabels(tick_vms)

    ax1.set_xlabel("Mean-shift magnitude", fontsize=10)
    ax2.set_xlabel("KL-matched variance ratio", fontsize=10)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        frameon=False, fontsize=12,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        bbox_transform=fig.transFigure,
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved -> {filename}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = run_sweep()

    plot_stacked(results, "detection_rate", "Detection rate (%)",
                 "Mean-shift Scenario", "Variance-change Scenario",
                 OUTPUT_DIR / "sweep_detection_rate_stacked.png")

    plot_stacked(results, "EDD", "Expected detection delay",
                 "Mean-shift Scenario", "Variance-change Scenario",
                 OUTPUT_DIR / "sweep_edd_stacked.png")

    print("Saved 2 stacked figures.")