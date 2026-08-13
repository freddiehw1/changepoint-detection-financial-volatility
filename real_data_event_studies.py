"""
Real-data event studies: raw (uncalibrated) score trajectories for all six
methods around three historical volatility events, on both log(VIX) and
S&P 500 log-returns.

No detection threshold is applied here -- this reproduces the qualitative,
shape-based comparison used in Section 5.3 of the thesis, following the
switch from threshold-based detection to raw score comparison described in
Section 5.2 (real-data thresholds proved unreliable to calibrate; see
Appendix A for why).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "event_studies"

from Methods_and_base_simulation import (
    w_cusum, w2_cusum, icss, bde_mean_cusum, max_dewma, aewma_huber
)

ALL_METHODS = {
    "w_cusum": w_cusum, "w2_cusum": w2_cusum, "icss": icss,
    "bde_mean_cusum": bde_mean_cusum, "max_dewma": max_dewma, "aewma_huber": aewma_huber,
}
METHOD_ORDER = ["w_cusum", "w2_cusum", "icss", "bde_mean_cusum", "max_dewma", "aewma_huber"]
MEAN_METHODS = ["w_cusum", "bde_mean_cusum", "aewma_huber"]
VAR_METHODS = ["w2_cusum", "icss", "max_dewma"]
ZETA = {
    "w_cusum": 0.49, "w2_cusum": 0.3157, "icss": None,
    "bde_mean_cusum": None, "max_dewma": None, "aewma_huber": None,
}
LOCAL_RETARGET_WINDOW = {"aewma_huber": 20}
DISPLAY_NAME = {
    "w_cusum": "W-CUSUM",
    "w2_cusum": "W²-CUSUM",
    "icss": "ICSS",
    "bde_mean_cusum": "Normalised Mean CUSUM",
    "max_dewma": "MAX-DEWMA",
    "aewma_huber": "AEWMA",
}
# Same paired colors as the sweep scripts: rank-based CUSUM (blue),
# retrospective/offline (green), EWMA-based (red/pink)
METHOD_COLOR = {
    "w_cusum":        "#1B4F8C",
    "w2_cusum":        "#5B9BD5",
    "icss":            "#1B8C5A",
    "bde_mean_cusum":  "#6FBF73",
    "aewma_huber":     "#8C1B3D",
    "max_dewma":       "#E08BA0",
}

# --------------------------------------------------------------------
# EDIT THIS LIST to add/change which events get a case study
# --------------------------------------------------------------------
EVENTS_TO_STUDY = [
    ("2008-09-15", "Lehman Brothers collapse"),
    ("2020-02-20", "COVID crash begins"),
    ("2018-02-05", "Volmageddon (XIV collapse)"),
]

PRE_DAYS, POST_DAYS = 120, 120
BASELINE_START, BASELINE_END = "2004-01-01", "2006-12-31"

FETCH_START = "1990-01-01"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_vix():
    """Download the full VIX history from FRED and return log(VIX)."""

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
        f"&cosd={FETCH_START}&coed={end}"
    )
    print("Downloading VIX from FRED...")
    vix = pd.read_csv(url)
    vix.columns = ["Date", "VIX"]
    vix["Date"] = pd.to_datetime(vix["Date"])
    vix["VIX"] = pd.to_numeric(vix["VIX"], errors="coerce")
    vix = vix.dropna().set_index("Date")
    print(f"  Got {len(vix)} rows ({vix.index.min().date()} to {vix.index.max().date()})")
    return np.log(vix["VIX"])


def fetch_sp500():
    """Download S&P 500 daily closes from Yahoo Finance and return log-returns."""

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance not installed. Run:\n"
            "    pip install yfinance --break-system-packages\n"
            "then re-run this script."
        )
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print("Downloading S&P 500 from Yahoo Finance...")
    sp500 = yf.download("^GSPC", start=FETCH_START, end=end, progress=False)

    if sp500.empty:
        raise SystemExit(
            "yfinance returned no data. Yahoo occasionally changes its API and "
            "breaks yfinance -- if this happens, download the CSV manually from\n"
            "    https://finance.yahoo.com/quote/%5EGSPC/history/\n"
            "and adapt this function to read from that file instead."
        )

    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    sp500 = sp500[["Close"]]
    log_ret = np.log(sp500["Close"]).diff().dropna()
    print(f"  Got {len(log_ret)} rows ({log_ret.index.min().date()} to {log_ret.index.max().date()})")
    return log_ret


def load_series():
    """Fetch both series used throughout the real-data analysis."""
    return fetch_vix(), fetch_sp500()


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def run_scores_only(x_series, detector_func, zeta, extra_kwargs=None, local_retarget_window=None):
    """Run one detector over a real-data series and return the raw score
    trajectory only -- no threshold, no detection tracking.

    local_retarget_window, if set, resets the detector's target to the
    mean of the first local_retarget_window observations before scoring
    begins (used for AEWMA so it targets the local pre-event mean rather
    than zero).
    """

    extra_kwargs = extra_kwargs or {}
    score, history, warmup, local_target = {}, [], [], None
    if local_retarget_window is None:
        local_target = extra_kwargs.get("target")

    scores, dates = [], []
    for date, x in x_series.items():
        history.append(x)

        if local_retarget_window is not None and local_target is None:
            warmup.append(x)
            scores.append(np.nan)
            dates.append(date)
            if len(warmup) >= local_retarget_window:
                local_target = float(np.mean(warmup))
                warmup = []
            continue

        kwargs = dict(extra_kwargs)
        if zeta is not None:
            kwargs["zeta"] = zeta
        if local_retarget_window is not None:
            kwargs["target"] = local_target

        score, _ = detector_func(x, score, history, **kwargs)
        scores.append(score.get("stat", np.nan))
        dates.append(date)

    return pd.Series(scores, index=dates)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scores_only(data_series, data_label, scores_by_method, event_date, filename):
    """Tall, two-column layout: raw series on top, mean-shift methods in
    the left column, variance-shift methods in the right column."""

    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(7, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(5, 2, height_ratios=[1.3, 0.12, 1, 1, 1])

    formatter = mdates.DateFormatter("%Y-%m")
    x_min, x_max = data_series.index.min(), data_series.index.max()

    data_min, x_max = data_series.index.min(), data_series.index.max()
    tick_start = data_min.to_period("M").to_timestamp()  # floored, for ticks only

    ticks = pd.date_range(tick_start, x_max, freq="2MS")
    ticks = [t for t in ticks if data_min <= t <= x_max]
    top_ticks = pd.date_range(tick_start, x_max, freq="MS")
    top_ticks = [t for t in top_ticks if data_min <= t <= x_max]

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(data_series.index, data_series.values, color="#333333", linewidth=0.8)
    ax0.axvline(event_date, color="#348ABD", linestyle="--", linewidth=1, alpha=0.7)
    ax0.set_ylabel(data_label, fontsize=11)
    ax0.set_xlim(data_min, x_max)
    ax0.set_xlabel("Date", fontsize=11)
    ax0.set_xticks(top_ticks)
    ax0.xaxis.set_major_formatter(formatter)
    ax0.tick_params(labelsize=12, rotation=30)
    ax0.minorticks_off()
    ax0.legend(handles=[
        Line2D([0], [0], color="#348ABD", linestyle="--", linewidth=1, label="Event date"),
    ], frameon=False, loc="upper left", fontsize=12)

    ax_mean_header = fig.add_subplot(gs[1, 0])
    ax_mean_header.axis("off")
    ax_mean_header.text(0.0, 0.0, "Mean-shift methods", fontsize=12, fontweight="bold", va="bottom", ha="left")

    ax_var_header = fig.add_subplot(gs[1, 1])
    ax_var_header.axis("off")
    ax_var_header.text(0.0, 0.0, "Variance-shift methods", fontsize=12, fontweight="bold", va="bottom", ha="left")

    mean_axes = [fig.add_subplot(gs[row, 0]) for row in range(2, 5)]
    var_axes = [fig.add_subplot(gs[row, 1]) for row in range(2, 5)]

    for ax, name in zip(mean_axes, MEAN_METHODS):
        ax.plot(scores_by_method[name].index, scores_by_method[name].values,
                 color=METHOD_COLOR[name], linewidth=1.0)
        ax.axvline(event_date, color="#348ABD", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(DISPLAY_NAME[name], fontsize=11, loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(labelsize=9, rotation=30)
        ax.minorticks_off()
        ax.set_ylabel("Score", fontsize=12)

    for ax, name in zip(var_axes, VAR_METHODS):
        ax.plot(scores_by_method[name].index, scores_by_method[name].values,
                 color=METHOD_COLOR[name], linewidth=1.0)
        ax.axvline(event_date, color="#348ABD", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(DISPLAY_NAME[name], fontsize=11, loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(labelsize=12, rotation=30)
        ax.minorticks_off()
        ax.set_ylabel("Score", fontsize=12)

    plt.savefig(filename, dpi=200, facecolor="white")
    plt.close()
    print(f"Saved -> {filename}")


def plot_scores_only_wide(data_series, data_label, scores_by_method, event_date, filename):
    """Wide, three-column layout: raw series on top, mean-shift methods in
    one row, variance-shift methods in a second row. Used for the
    figures embedded directly in the thesis."""

    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(5, 3, height_ratios=[1.3, 0.12, 1, 0.12, 1])

    formatter = mdates.DateFormatter("%Y-%m")
    x_min, x_max = data_series.index.min(), data_series.index.max()

    data_min, x_max = data_series.index.min(), data_series.index.max()
    tick_start = data_min.to_period("M").to_timestamp()  # floored, for ticks only

    ticks = pd.date_range(tick_start, x_max, freq="2MS")
    ticks = [t for t in ticks if data_min <= t <= x_max]
    top_ticks = pd.date_range(tick_start, x_max, freq="MS")
    top_ticks = [t for t in top_ticks if data_min <= t <= x_max]

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(data_series.index, data_series.values, color="#333333", linewidth=0.8)
    ax0.axvline(event_date, color="#348ABD", linestyle="--", linewidth=1, alpha=0.7)
    ax0.set_ylabel(data_label, fontsize=13)
    ax0.set_xlim(data_min, x_max)
    ax0.set_xlabel("Date", fontsize=12)
    ax0.set_xticks(top_ticks)
    ax0.xaxis.set_major_formatter(formatter)
    ax0.tick_params(labelsize=12, rotation=30)
    ax0.minorticks_off()
    ax0.legend(handles=[
        Line2D([0], [0], color="#348ABD", linestyle="--", linewidth=1, label="Event date"),
    ], frameon=False, loc="upper left", fontsize=12)

    ax_mean_header = fig.add_subplot(gs[1, :])
    ax_mean_header.axis("off")
    ax_mean_header.text(0.0, 0.0, "Mean-shift methods", fontsize=14, fontweight="bold", va="bottom", ha="left")

    mean_axes = [fig.add_subplot(gs[2, col]) for col in range(3)]
    for ax, name in zip(mean_axes, MEAN_METHODS):
        ax.plot(scores_by_method[name].index, scores_by_method[name].values,
                 color=METHOD_COLOR[name], linewidth=1.0)
        ax.axvline(event_date, color="#348ABD", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(DISPLAY_NAME[name], fontsize=12, loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(labelsize=12, rotation=30)
        ax.minorticks_off()
    mean_axes[0].set_ylabel("Score", fontsize=13)

    ax_var_header = fig.add_subplot(gs[3, :])
    ax_var_header.axis("off")
    ax_var_header.text(0.0, 0.0, "Variance-shift methods", fontsize=14, fontweight="bold", va="bottom", ha="left")

    var_axes = [fig.add_subplot(gs[4, col]) for col in range(3)]
    for ax, name in zip(var_axes, VAR_METHODS):
        ax.plot(scores_by_method[name].index, scores_by_method[name].values,
                 color=METHOD_COLOR[name], linewidth=1.0)
        ax.axvline(event_date, color="#348ABD", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(DISPLAY_NAME[name], fontsize=12, loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(labelsize=12, rotation=30)
        ax.minorticks_off()
    var_axes[0].set_ylabel("Score", fontsize=13)

    plt.savefig(filename, dpi=200, facecolor="white")
    plt.close()
    print(f"Saved -> {filename}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vix, sp500 = load_series()

    vix_baseline = vix.loc[BASELINE_START:BASELINE_END]
    sp500_baseline = sp500.loc[BASELINE_START:BASELINE_END]

    for date_str, label in EVENTS_TO_STUDY:
        event_date = pd.Timestamp(date_str)
        x_start = event_date - pd.Timedelta(days=PRE_DAYS)
        x_end = event_date + pd.Timedelta(days=POST_DAYS)
        safe_name = re.sub(r"[^\w\-]+", "_", label).strip("_")[:60]

        print(f"\n=== {label} ({x_start.date()} to {x_end.date()}) ===")

        for series_name, full_series, baseline in [("VIX", vix, vix_baseline), ("SP500", sp500, sp500_baseline)]:
            window = full_series.loc[x_start:x_end]
            if window.empty:
                print(f"  [{series_name}] skipped -- no data in this window")
                continue

            scores_by_method = {}
            for name in METHOD_ORDER:
                detector = ALL_METHODS[name]
                extra_kwargs = {"sigma": baseline.std()} if name == "aewma_huber" else {}
                retarget = LOCAL_RETARGET_WINDOW.get(name)
                scores_by_method[name] = run_scores_only(window, detector, ZETA[name], extra_kwargs,
                                                           local_retarget_window=retarget)
                s = scores_by_method[name]
                print(f"  [{series_name}] {name}: score range [{s.min():.3f}, {s.max():.3f}]")

            data_label = "log(VIX)" if series_name == "VIX" else "S&P500 log-returns"
            plot_scores_only(
                window, data_label, scores_by_method, event_date,
                OUTPUT_DIR / f"scores_only_{safe_name}_{series_name}.png"
            )
            plot_scores_only_wide(
                window, data_label, scores_by_method, event_date,
                OUTPUT_DIR / f"scores_only_wide_{safe_name}_{series_name}.png"
            )