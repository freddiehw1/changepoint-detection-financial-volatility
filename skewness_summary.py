"""
Skewness summary for VIX, log(VIX), and S&P 500 log-returns.

Reproduces Table 5.1 and Figure 5.1 in the thesis: quantifies how much the
log transform reduces VIX's positive skew, motivating its use throughout
Chapter 5 (Section 5.1).
"""

import numpy as np
import pandas as pd
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
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})

from scipy.stats import skew
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "skewness_summary"
FETCH_START = "1990-01-01"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_vix():
    """Download the full raw VIX history from FRED."""

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
    return vix["VIX"]


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

    log_ret = np.log(sp500["Close"]).diff().dropna()
    print(f"  Got {len(log_ret)} rows ({log_ret.index.min().date()} to {log_ret.index.max().date()})")
    return log_ret


# ---------------------------------------------------------------------------
# Skewness summary
# ---------------------------------------------------------------------------

def print_skew_table(vix, log_vix, sp500_ret):
    """Print sample skewness for all three series, plus ready-to-paste
    LaTeX rows for tab:skew-comparison."""

    rows = [
        ("VIX (raw)", vix),
        ("log(VIX)", log_vix),
        ("S&P 500 log-returns", sp500_ret),
    ]

    print("\n=== Skewness summary ===")
    print(f"{'Series':<22}{'N':>8}{'Skewness':>12}")
    for name, series in rows:
        s = float(skew(series.dropna().values))
        print(f"{name:<22}{len(series):>8}{s:>12.4f}")

    print("\nLaTeX table row values (paste into tab:skew-comparison):")
    for name, series in rows:
        s = float(skew(series.dropna().values))
        print(f"{name} & {s:.3f} \\\\")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_skew_histograms(vix, log_vix, filename):
    """Side-by-side histograms of raw VIX and log(VIX), illustrating the
    reduction in skew from the log transform."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax1.hist(vix.values, bins=60, color="#7F7F7F", edgecolor="white", linewidth=0.3)
    ax1.set_title("Raw VIX", fontsize=13, loc="left")
    ax1.set_xlabel("VIX")
    ax1.set_ylabel("Frequency")

    ax2.hist(log_vix.values, bins=60, color="#4D4D4D", edgecolor="white", linewidth=0.3)
    ax2.set_title("log(VIX)", fontsize=13, loc="left")
    ax2.set_xlabel("log(VIX)")

    plt.savefig(filename, dpi=200, facecolor="white")
    plt.close()
    print(f"\nSaved -> {filename}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vix = fetch_vix()
    log_vix = np.log(vix)
    sp500_ret = fetch_sp500()

    print_skew_table(vix, log_vix, sp500_ret)
    plot_skew_histograms(vix, log_vix, OUTPUT_DIR / "vix_skew_histograms.png")