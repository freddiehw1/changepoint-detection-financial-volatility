"""
Illustration of the two baseline simulation scenarios.

Reproduces Figure 3.1 in the thesis (Section 3.4.1): a single realisation
each of the mean-shift and variance-change scenarios, with the shifted
region highlighted against the surrounding in-control series. The two
individual (non-stacked) panels are also saved separately; these were used
in presentation slides and are not referenced directly in the thesis text.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

from pathlib import Path
from Methods_and_base_simulation import simulate_spike, simulate_variance_change

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline_scenario_figure"


T = 500
CHANGEPOINT = 250
DURATION = 25
RANDOM_SEED = 1

MEAN_SHIFT_KWARGS = dict(T=T, changepoint=CHANGEPOINT, spike_size_std=2.0,
                          duration=DURATION, autocorrelation=0.0, random_seed=RANDOM_SEED)
VARIANCE_CHANGE_KWARGS = dict(T=T, changepoint=CHANGEPOINT, variance_multiplier=6.94,
                               duration=DURATION, autocorrelation=0.0, random_seed=RANDOM_SEED)


def draw_scenario(ax, data, changepoint, duration, title, ylabel, show_legend=False):
    """Plot one scenario onto a given axis: in-control series in gray,
    the shifted region in red, and dashed/dotted markers at the
    change-point and the end of the shifted region."""

    t = np.arange(len(data))
    pre = slice(0, changepoint)
    injected = slice(changepoint, min(changepoint + duration, len(data)))
    post = slice(min(changepoint + duration, len(data)), len(data))

    ax.plot(t[pre], data[pre], color="#333333", linewidth=0.9, label="In-control")
    ax.plot(t[injected], data[injected], color="#D1495B", linewidth=1.1, label="Shifted region")
    if post.stop > post.start:
        ax.plot(t[post], data[post], color="#333333", linewidth=0.9)

    ax.axvline(changepoint, color="#348ABD", linestyle="--", linewidth=1, alpha=0.7, label="Change-point")
    if duration is not None and changepoint + duration < len(data):
        ax.axvline(changepoint + duration, color="#348ABD", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if show_legend:
        ax.legend(frameon=False, loc="upper left")


def plot_scenario(data, changepoint, duration, title, ylabel, filename):
    """Single-panel plot of one scenario. Supplementary -- not referenced
    directly in the thesis text."""

    fig, ax = plt.subplots(figsize=(9, 4))
    draw_scenario(ax, data, changepoint, duration, title, ylabel, show_legend=True)
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, facecolor="white")
    plt.close()
    print(f"Saved -> {filename}")


def plot_stacked_scenarios(mean_data, mean_cp, var_data, var_cp, duration, filename):
    """Two-panel stacked plot: mean-shift scenario on top, variance-change
    scenario below. This is Figure 3.1 in the thesis."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    draw_scenario(ax1, mean_data, mean_cp, duration, "Mean-shift Scenario", "Value", show_legend=True)
    draw_scenario(ax2, var_data, var_cp, duration, "Variance-change Scenario", "Value", show_legend=False)

    ax2.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, facecolor="white")
    plt.close()
    print(f"Saved -> {filename}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mean_data, mean_cp = simulate_spike(**MEAN_SHIFT_KWARGS)
    plot_scenario(
        mean_data, mean_cp, DURATION,
        "",
        "Value",
        OUTPUT_DIR / "baseline_mean_shift.png",
    )

    var_data, var_cp = simulate_variance_change(**VARIANCE_CHANGE_KWARGS)
    plot_scenario(
        var_data, var_cp, DURATION,
        "",
        "Value",
        OUTPUT_DIR / "baseline_variance_change.png",
    )

    plot_stacked_scenarios(
        mean_data, mean_cp, var_data, var_cp, DURATION,
        OUTPUT_DIR / "baseline_stacked.png",
    )