# CUSUM/EWMA Volatility Change-Point Detection

Implementation of six change-point detection methods — W-CUSUM, W²-CUSUM, ICSS,
Normalised Mean-CUSUM, Max-DEWMA, and AEWMA — for detecting mean and variance
shifts in financial volatility. This repository contains the full simulation,
calibration, and real-data analysis code behind the thesis *[thesis title]*.

## Methods

| Method | Family | Update | Target |
|---|---|---|---|
| W-CUSUM | CUSUM | Sequential | Mean |
| W²-CUSUM | CUSUM | Sequential | Variance |
| ICSS | CUSUM | Retrospective | Variance |
| Normalised Mean-CUSUM | CUSUM | Retrospective | Mean |
| Max-DEWMA | EWMA | Sequential | Mean & variance (joint) |
| AEWMA | EWMA | Sequential | Mean |

## Requirements

pip install numpy scipy matplotlib scienceplots pandas yfinance --break-system-packages
`yfinance` is only needed for `real_data_event_studies.py`, which pulls S&P 500 data
from Yahoo Finance and VIX data from FRED.

`yfinance` is only needed for `real_data_event_studies.py`, which pulls S&P 500 data
from Yahoo Finance and VIX data from FRED.

## Repository structure

| File | Description | Reproduces |
|---|---|---|
| `Methods_and_base_simulation.py` | Core implementation of all six methods, the two baseline simulators, and shared plotting utilities. Every other script imports from this. | Chapter 3 |
| `calibration.py` | Binary-search calibration of each method's threshold to a 5% false-alarm rate. | Table 4.1, Appendix A |
| `baseline_scenario_figure.py` | Illustrates the mean-shift and variance-change scenarios themselves, as a single realisation. | Figure 3.1 |
| `baseline_plots.py` | Runs all six methods on both baseline scenarios and plots the result. | Figure 4.1, Appendix B |
| `monte_carlo_evaluation.py` | Full Monte Carlo comparison of all six methods against both baseline scenarios (detection rate, EDD, SD of delay, premature false-alarm rate). | Tables 4.2, 4.3 |
| `Change_point_magnitude_sweep.py` | Sweeps mean-shift magnitude (with KL-matched variance-change magnitude) and plots detection rate and EDD. | Figures 4.2, 4.3 |
| `Change_point_duration_sweep.py` | Sweeps change duration at a fixed magnitude and plots detection rate and EDD. | Figures 4.4, 4.5 |
| `false_alarm_published_thresholds.py` | Compares each method's published/asymptotic threshold against its empirically calibrated one. | Section 4.1 discussion |
| `skewness_summary.py` | Computes and compares sample skewness for raw VIX, log(VIX), and S&P 500 log-returns. | Table 5.1, Figure 5.1 |
| `real_data_event_studies.py` | Downloads VIX and S&P 500 data and plots raw method scores around three historical volatility events (Lehman Brothers, Volmageddon, COVID-19). | Figures 5.2–5.7 |

## Usage

Each script is runnable standalone:Output figures are written to `outputs/<script_name>/` and are not committed to this repository — run the relevant script to regenerate them.


## Data sources

- **VIX**: CBOE Volatility Index, sourced from [FRED](https://fred.stlouisfed.org/series/VIXCLS) (series `VIXCLS`).
- **S&P 500**: Daily closing prices via [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/), accessed through `yfinance`.

