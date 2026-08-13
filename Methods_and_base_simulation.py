"""Core detection methods, baseline simulators, and plotting utilities
for the CUSUM/EWMA change-point comparison."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from functools import partial
from scipy.stats import norm, chi2
from scipy.special import chdtr, ndtri


# ---------------------------------------------------------------------------
# Baseline simulators
# ---------------------------------------------------------------------------

def simulate_variance_change(T: int = 500,
    changepoint: int = 250,
    variance_multiplier: float = 6.94,
    duration: int = 25,
    autocorrelation: float = 0.0,
    random_seed: int = 40):
    """Generate a series with a temporary variance increase.

    Noise outside the shifted window has unit variance; within
    [changepoint, changepoint + duration) it is scaled by
    sqrt(variance_multiplier). AR(1) dependence is applied afterwards if
    autocorrelation != 0.
    """

    np.random.seed(random_seed)
    eps = np.random.normal(0, 1, T)
    eps = eps / eps.std()

    std_multiplier = np.sqrt(variance_multiplier)
    scale = np.ones(T)
    if duration is None:
        scale[changepoint:] = std_multiplier
    else:
        scale[changepoint : changepoint + duration] = std_multiplier

    eps_scaled = eps * scale

    X = np.zeros(T)
    X[0] = eps_scaled[0]
    for t in range(1, T):
        X[t] = autocorrelation * X[t - 1] + eps_scaled[t]

    return X, changepoint


def simulate_spike(T: int = 500,
    changepoint: int = 250,
    spike_size_std: float = 2,
    duration: int = 15,
    autocorrelation: float = 0.0,
    random_seed: int = 40):
    """Generate a series with a temporary additive mean shift.

    A constant of size spike_size_std (in standard deviation units) is
    added over [changepoint, changepoint + duration). AR(1) dependence
    is applied to the noise before the shift is added.
    """

    np.random.seed(random_seed)
    eps = np.random.normal(0, 1, T)
    eps = eps / eps.std()

    X = np.zeros(T)
    X[0] = eps[0]
    for t in range(1, T):
        X[t] = autocorrelation * X[t - 1] + eps[t]

    X[changepoint : changepoint + duration] += spike_size_std

    return X, changepoint


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def icss(x, score, history, threshold=1.7206):
    """ICSS variance detector, recomputed from the full history at each call."""

    score = score.copy()
    data = np.array(history, dtype=float)
    T = len(data)

    if T < 5:
        score["stat"] = 0.0
        return score, False

    segment = data - np.mean(data)
    squared_data = segment ** 2
    cumulative_sum = np.cumsum(squared_data)
    total_sum = cumulative_sum[-1]

    k_range = np.arange(1, T)
    D_values = cumulative_sum[k_range - 1] / total_sum - k_range / T
    abs_D_values = np.abs(D_values)

    k_star = np.argmax(abs_D_values) + 1
    M = float(np.sqrt(T / 2) * abs_D_values[k_star - 1])

    score["stat"] = M
    detection = M > threshold
    return score, detection


def bde_mean_cusum(x, score, history, threshold=1.7133):
    """Self-normalised CUSUM for a mean shift, recomputed from the full history at each call."""

    score = score.copy()
    data = np.array(history, dtype=float)
    T = len(data)

    if T < 5:
        score["stat"] = 0.0
        return score, False

    sigma_hat = np.std(data, ddof=1)
    if sigma_hat == 0:
        score["stat"] = 0.0
        return score, False

    centered = data - np.mean(data)
    cumulative_sum = np.cumsum(centered)

    U = cumulative_sum / (sigma_hat * np.sqrt(T))
    abs_U = np.abs(U)

    k_star = int(np.argmax(abs_U) + 1)
    M = float(abs_U[k_star - 1])

    score["stat"] = M
    detection = M > threshold
    return score, detection


def w_cusum(x, score, history, zeta=0.98, threshold=2.5235):
    """Signed sequential rank CUSUM (mean), updated incrementally."""

    score = score.copy()

    i = len(history)
    abs_hist = np.abs(np.asarray(history))
    abs_x = abs(x)
    r_plus = np.sum(abs_hist <= abs_x)
    s = np.sign(x) if x != 0 else 0.0

    xi = s * r_plus * np.sqrt(6.0 / ((2 * i + 1) * (i + 1)))

    prev_stat = score.get("stat", 0.0)
    new_stat = max(0.0, prev_stat + xi - zeta)
    score["stat"] = new_stat

    detection = new_stat > threshold
    return score, detection


def w2_cusum(x, score, history, zeta=0.533, threshold=6.2902):
    """Squared sequential rank CUSUM (variance), updated incrementally."""

    score = score.copy()

    i = len(history)
    abs_hist = np.abs(np.asarray(history))
    abs_x = abs(x)
    r_plus = np.sum(abs_hist <= abs_x)

    xi = (6 * r_plus**2) / ((2 * i + 1) * (i + 1)) - 1

    prev_stat = score.get("stat", 0.0)
    new_stat = max(0.0, prev_stat + xi - zeta)
    score["stat"] = new_stat

    detection = new_stat > threshold
    return score, detection


def aewma_huber(x, score, history, lam=0.10, k=3.0, threshold=1.2021, target=0.0, sigma=1.0):
    """Adaptive EWMA with a Huber-type score function (mean)."""

    del history

    if not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")
    if k <= 0 or threshold <= 0 or sigma <= 0:
        raise ValueError("k, threshold, and sigma must be positive")

    score = score.copy()
    previous_stat = float(score.get("stat", target))
    error = float(x) - previous_stat
    k_abs = k * sigma

    if error < -k_abs:
        phi = error + (1.0 - lam) * k_abs
    elif error > k_abs:
        phi = error - (1.0 - lam) * k_abs
    else:
        phi = lam * error

    new_stat = previous_stat + phi
    score["stat"] = new_stat
    score["weight"] = phi / error if abs(error) > 1e-12 else lam

    detection = abs(new_stat - target) > threshold * sigma
    return score, detection


LOCKED_PARAMETERS = {
    "lam": 0.10,
    "k": 3.0,
    "target": 0.0,
    "sigma": 1.0,
}


def max_dewma(x, score, history, threshold=3.1551):
    """Max-DEWMA joint mean/variance detector.

    Uses a fixed 50-observation baseline (estimated once) for mu0, sigma0,
    and a rolling 5-observation subgroup for the mean and variance
    statistics. Produces no output until 55 observations are available.
    """

    lam = 0.40
    subgroup_size = 5
    baseline_size = 50
    UCL = threshold

    if len(history) < baseline_size + subgroup_size:
        score["stat"] = 0.0
        score["Y"] = 0.0
        score["Z"] = 0.0
        score["W"] = 0.0
        score["Q"] = 0.0
        return score, False

    baseline = np.array(history[:baseline_size])
    mu0 = np.mean(baseline)
    sigma0 = np.std(baseline, ddof=1)

    if sigma0 == 0:
        sigma0 = 1e-8

    recent = np.array(history[-subgroup_size:])
    xbar = np.mean(recent)
    s2 = np.var(recent, ddof=1)

    U = (xbar - mu0) / (sigma0 / np.sqrt(subgroup_size))

    chi_value = (subgroup_size - 1) * s2 / (sigma0 ** 2)
    p_value = chdtr(subgroup_size - 1, chi_value)

    p_value = np.clip(p_value, 1e-6, 1 - 1e-6)
    V = ndtri(p_value)

    Y_prev = score.get("Y", 0.0)
    Z_prev = score.get("Z", 0.0)
    W_prev = score.get("W", 0.0)
    Q_prev = score.get("Q", 0.0)

    Y = (1 - lam) * Y_prev + lam * U
    Z = (1 - lam) * Z_prev + lam * V

    W = (1 - lam) * W_prev + lam * Y
    Q = (1 - lam) * Q_prev + lam * Z

    L = max(abs(W), abs(Q))

    score["Y"] = Y
    score["Z"] = Z
    score["W"] = W
    score["Q"] = Q
    score["stat"] = L
    score["ucl"] = UCL

    detection = L > UCL

    return score, detection


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def cpd_simulation(data, changepoint, detector, initial_score):
    """Run a detector over a full series.

    Returns the score trajectory, the first detection time (or None if it
    never fires), and the detection delay relative to changepoint.
    """

    score = initial_score.copy()
    history = []
    detection_time = None
    scores = np.zeros(len(data))

    for t in range(len(data)):
        history.append(data[t])
        score, detected = detector(data[t], score, history)
        scores[t] = score["stat"]
        if detected and detection_time is None:
            detection_time = t

    delay = detection_time - changepoint if detection_time is not None else None
    return scores, detection_time, delay


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data, scores, changepoint, threshold, detection_time=None):
    """Two-panel static plot of a single run: raw observations and the
    detector's score, with the true change-point, detection time, and
    threshold marked.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Change Point Detection")

    ax1.plot(data, color='darkblue')
    ax1.axvline(changepoint, color='r', ls='--', label='true changepoint')
    if detection_time is not None:
        ax1.axvline(detection_time, color='orange', ls='--', label='detection')
    ax1.set_ylabel("Observation")
    ax1.legend()

    ax2.plot(scores, color='green')
    ax2.axvline(changepoint, color='r', ls='--')
    if detection_time is not None:
        ax2.axvline(detection_time, color='orange', ls='--')
    ax2.axhline(threshold, color='purple', ls=':', label=f'threshold ({threshold})')
    if min(scores) < 0:
        ax2.axhline(-threshold, color='purple', ls=':')
    ax2.set_ylabel("CUSUM Score")
    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.show()


def animate(data, scores, changepoint, threshold, detection_time=None, freeze=False,
            slow_motion=True, slow_factor=15, slow_window=15):
    """Animated version of plot(), optionally slowed down around the
    change-point/detection window (slow_motion=True).
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Change Point Detection")

    ax1.set_xlim(0, len(data))
    ax1.set_ylim(min(data) - 0.5, max(data) + 0.5)
    ax1.set_ylabel("observation")

    ax2.set_xlim(0, len(data))
    if min(scores) < 0:
        ax2.set_ylim(min(min(scores), -threshold) - 0.5, max(max(scores), threshold) + 0.5)
    else:
        ax2.set_ylim(0, max(max(scores), threshold) + 0.5)
    ax2.set_ylabel("CUSUM Score")
    ax2.set_xlabel("time")

    line1, = ax1.plot([], [], color='darkblue')
    line2, = ax2.plot([], [], color='green')

    cp_line1 = ax1.axvline(changepoint, color='r', ls='--', visible=False)
    cp_line2 = ax2.axvline(changepoint, color='r', ls='--', visible=False)
    det_line1 = ax1.axvline(detection_time if detection_time is not None else 0, color='orange', ls='--', visible=False)
    det_line2 = ax2.axvline(detection_time if detection_time is not None else 0, color='orange', ls='--', visible=False)

    ax2.axhline(threshold, color='purple', ls=':')
    if min(scores) < 0:
        ax2.axhline(-threshold, color='purple', ls=':')

    ax1.legend(handles=[
        Line2D([0], [0], color='r', ls='--', label='true changepoint'),
        Line2D([0], [0], color='orange', ls='--', label='detection'),
    ], handlelength=3, loc='upper right')

    ax2.legend(handles=[
        Line2D([0], [0], color='purple', ls=':', label=f'threshold ({threshold})'),
    ], handlelength=3, loc='upper right')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        cp_line1.set_visible(False)
        cp_line2.set_visible(False)
        det_line1.set_visible(False)
        det_line2.set_visible(False)
        return line1, line2, cp_line1, cp_line2, det_line1, det_line2

    if slow_motion:
        slow_start = max(0, changepoint - slow_window)
        slow_end   = (detection_time + 10) if detection_time is not None else (changepoint + slow_window)
        slow_end   = min(slow_end, len(data))

        frames = []
        for t in range(len(data)):
            if slow_start <= t <= slow_end:
                frames.extend([t] * slow_factor)
            else:
                frames.append(t)
    else:
        frames = list(range(len(data)))

    def update(t):
        line1.set_data(range(t), data[:t])
        line2.set_data(range(t), scores[:t])

        cp_line1.set_visible(t >= changepoint)
        cp_line2.set_visible(t >= changepoint)

        if detection_time is not None:
            det_line1.set_visible(t >= detection_time)
            det_line2.set_visible(t >= detection_time)

        return line1, line2, cp_line1, cp_line2, det_line1, det_line2

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=20,
        blit=False,
        repeat=not freeze
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    THRESHOLD = 10
    detector = partial(max_dewma, threshold=THRESHOLD)

    data, changepoint = simulate_spike(duration=25)

    scores, detection_time, delay = cpd_simulation(
        data,
        changepoint,
        detector=detector,
        initial_score={
            "stat": 0,
            "Y": 0,
            "Z": 0,
            "W": 0,
            "Q": 0
        }
    )
    print(detection_time)
    animate(data, scores, changepoint, THRESHOLD, detection_time, freeze=True, slow_motion=False)