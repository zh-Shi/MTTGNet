import numpy as np
from scipy import stats


def _ensure_2d(prediction, target):
    """Reshape 1D [N] sample arrays to [N, 1] for a single per-horizon value.

    The axis=0 reduction in the metrics treats rows as samples and columns as
    horizons.  A 1D input is therefore an N-sample, single-horizon array, NOT
    a horizon vector: reshaping [N] to [1, N] would silently turn N samples
    into N "horizons", so ``horizon_metrics(pred[:,0], target[:,0])`` returned
    an N-length array instead of one scalar RMSE.
    """
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    return prediction, target


def horizon_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    prediction, target = _ensure_2d(prediction, target)
    """Compute per-horizon MAE, RMSE, and Bias."""
    error = prediction - target

    return {
        "mae": np.mean(
            np.abs(error),
            axis=0,
        ),
        "rmse": np.sqrt(
            np.mean(error ** 2, axis=0)
        ),
        "bias": np.mean(
            error,
            axis=0,
        ),
    }


def skill_score(
    model_rmse: np.ndarray,
    baseline_rmse: np.ndarray,
) -> np.ndarray:
    """Skill score: 1 - RMSE_model / RMSE_baseline (positive = better)."""
    return (
        1.0
        - model_rmse
        / np.maximum(baseline_rmse, 1e-8)
    )


def compute_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Per-horizon Pearson correlation coefficient."""
    prediction, target = _ensure_2d(prediction, target)
    pred_mean = prediction - prediction.mean(axis=0, keepdims=True)
    true_mean = target - target.mean(axis=0, keepdims=True)
    num = (pred_mean * true_mean).sum(axis=0)
    den = np.sqrt(
        (pred_mean ** 2).sum(axis=0)
        * (true_mean ** 2).sum(axis=0)
    )
    return num / np.maximum(den, 1e-8)


# ==================== Extended Metrics (总纲 §十) ====================


def nse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Nash-Sutcliffe Efficiency per horizon.  1.0 = perfect, <0 = worse than mean.

    NSE = 1 - Σ(pred - true)² / Σ(true - mean(true))²
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.sum((prediction - target) ** 2, axis=0)
        denominator = np.sum(
            (target - target.mean(axis=0, keepdims=True)) ** 2, axis=0
        )
        val = 1.0 - numerator / np.maximum(denominator, 1e-8)
    return val


def acc(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Anomaly Correlation Coefficient (ACC) per horizon.

    Meteorological ACC is ``corr(pred - climo, target - climo)`` where
    ``climo`` is the day-of-year/hour climatological mean — which is *not*
    a constant across samples.  This function has no access to a climatology,
    so it computes the plain Pearson correlation instead.

    That equals ACC in exactly two cases:
      * the target is already an anomaly (climatology pre-subtracted), e.g.
        Datasets A/C;
      * the climatology is constant across samples (rare).

    On raw-temperature targets (Datasets B/D) the seasonal cycle dominates,
    so this returns the seasonal-cycle correlation (≈0.99), NOT forecast
    skill.  Use only on anomaly targets, or subtract the climatology first.
    """
    return compute_correlation(prediction, target)


def kge(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kling-Gupta Efficiency per horizon.

    KGE = 1 - sqrt( (r-1)² + (α-1)² + (β-1)² )
    where r = correlation, α = σ_pred/σ_true, β = μ_pred/μ_true
    """
    prediction, target = _ensure_2d(prediction, target)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = compute_correlation(prediction, target)
        alpha = np.std(prediction, axis=0) / np.maximum(np.std(target, axis=0), 1e-8)
        beta = prediction.mean(axis=0) / np.maximum(target.mean(axis=0), 1e-8)

        val = 1.0 - np.sqrt(
            (r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
        )
    return val


def explained_variance(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Explained Variance per horizon.

    EV = 1 - Var(target - prediction) / Var(target)
    Best = 1.0
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        error_var = np.var(target - prediction, axis=0)
        target_var = np.var(target, axis=0)
        val = 1.0 - error_var / np.maximum(target_var, 1e-8)
    return val


def smape(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error per horizon (0–200%).

    sMAPE = 200 * mean(|pred - true| / (|pred| + |true| + ε))
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.abs(prediction - target)
        denominator = np.abs(prediction) + np.abs(target)
        val = 200.0 * np.mean(
            numerator / np.maximum(denominator, 1e-8), axis=0
        )
    return val


def extreme_recall(
    prediction: np.ndarray,
    target: np.ndarray,
    quantile: float = 0.95,
) -> np.ndarray:
    """Extreme-value recall per horizon.

    Measures the fraction of true extreme events (top 1-quantile) that the
    model also predicts as extreme (top 1-quantile of predictions).

    Args:
        prediction: [N, H]
        target: [N, H]
        quantile: threshold for defining "extreme" (default 0.95 = top 5%)

    Returns:
        recall per horizon step
    """
    prediction, target = _ensure_2d(prediction, target)
    h = target.shape[1]
    recall = np.zeros(h)

    for step in range(h):
        pred_s = prediction[:, step]
        true_s = target[:, step]

        pred_thresh = np.quantile(pred_s, quantile)
        true_thresh = np.quantile(true_s, quantile)

        # Strict `>` — with `>=`, any samples sitting exactly ON the quantile
        # are counted as extreme; for discrete/rounded predictions that can
        # inflate the "top 5%" beyond its intended size and bias recall up.
        true_extreme = true_s > true_thresh
        pred_extreme = pred_s > pred_thresh

        hits = (true_extreme & pred_extreme).sum()
        total_extreme = true_extreme.sum()

        recall[step] = hits / max(total_extreme, 1)

    return recall


def trend_score(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-horizon sign accuracy of the change direction.

    Measures whether pred and target change in the same direction
    relative to the first step.

    Returns:
        fraction of correct direction predictions per horizon
    """
    prediction, target = _ensure_2d(prediction, target)
    n, h = target.shape
    if h < 2:
        return np.ones(h)

    pred_diff = prediction[:, 1:] - prediction[:, :-1]
    true_diff = target[:, 1:] - target[:, :-1]

    correct_sign = (np.sign(pred_diff) == np.sign(true_diff)).astype(np.float32)

    trend = np.zeros(h)
    trend[0] = 1.0  # first step has no delta reference
    trend[1:] = correct_sign.mean(axis=0)
    return trend


def r2_score(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-horizon R² (coefficient of determination)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = np.sum((target - prediction) ** 2, axis=0)
        ss_tot = np.sum(
            (target - target.mean(axis=0, keepdims=True)) ** 2, axis=0
        )
        val = 1.0 - ss_res / np.maximum(ss_tot, 1e-8)
    return val


def ioa(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Willmott's Index of Agreement per horizon.

    IoA = 1 - Σ(pred - true)² / Σ(|pred - μ_true| + |true - μ_true|)²
    Best = 1.0 (perfect agreement), 0 = no agreement.
    More sensitive to systematic bias than NSE.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        true_mean = target.mean(axis=0, keepdims=True)
        numerator = np.sum((prediction - target) ** 2, axis=0)
        pred_dev = np.abs(prediction - true_mean)
        true_dev = np.abs(target - true_mean)
        denominator = np.sum((pred_dev + true_dev) ** 2, axis=0)
        val = 1.0 - numerator / np.maximum(denominator, 1e-8)
    return val


def max_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Maximum absolute error per horizon (worst-case performance)."""
    prediction, target = _ensure_2d(prediction, target)
    return np.max(np.abs(prediction - target), axis=0)


def median_abs_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Median absolute error per horizon (robust to outliers)."""
    prediction, target = _ensure_2d(prediction, target)
    return np.median(np.abs(prediction - target), axis=0)


def relative_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Relative RMSE per horizon (%).

    rRMSE = RMSE / mean(|true|) * 100
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=0))
        true_mean_abs = np.mean(np.abs(target), axis=0)
        val = rmse / np.maximum(true_mean_abs, 1e-8) * 100.0
    return val


def relative_bias(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Percent bias (PBIAS) per horizon.

    PBIAS = mean(pred - true) / mean(true) * 100
    Positive = model overestimates, Negative = underestimates.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = np.mean(prediction - target, axis=0)
        true_mean = np.mean(target, axis=0)
        val = bias / np.maximum(np.abs(true_mean), 1e-8) * 100.0
    return val


# ==================== SHASH probabilistic metrics ====================


def shash_params_from_dist(dist_params: np.ndarray):
    """Split [..., 4] SHASH parameters into (mu, sigma, gamma, tau)."""
    return (dist_params[..., 0], dist_params[..., 1],
            dist_params[..., 2], dist_params[..., 3])


def shash_quantile(u: np.ndarray, mu, sigma, gamma, tau) -> np.ndarray:
    """SHASH quantile: q = mu + sigma*sinh((Phi^-1(u) + gamma)/tau)."""
    return mu + sigma * np.sinh((stats.norm.ppf(u) + gamma) / tau)


def shash_sample(mu, sigma, gamma, tau, n_samples=1000, seed=0) -> np.ndarray:
    """Draw X = mu + sigma*sinh((Z+gamma)/tau), Z~N(0,1).  Returns [S, N, H]."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples,) + mu.shape)
    return mu + sigma * np.sinh((z + gamma) / tau)


def shash_crps(dist_params: np.ndarray, target: np.ndarray,
               n_samples: int = 500, seed: int = 0) -> np.ndarray:
    """Continuous Ranked Probability Score via the E|X-y| - 0.5*E|X-X'| identity.

    X, X' iid ~ SHASH(mu, sigma, gamma, tau); the expectation is approximated
    by Monte Carlo.  Returns a per-horizon [H] array of mean CRPS.
    """
    mu, sigma, gamma, tau = shash_params_from_dist(dist_params)
    x = shash_sample(mu, sigma, gamma, tau, n_samples, seed)   # [S, N, H]
    e1 = np.abs(x - target).mean(axis=0)                       # [N, H]
    half = n_samples // 2
    e2 = np.abs(x[:half] - x[half:2 * half]).mean(axis=0)      # [N, H]
    return (e1 - 0.5 * e2).mean(axis=0)                        # [H]


def shash_interval_coverage(dist_params: np.ndarray, target: np.ndarray,
                            alpha: float = 0.9) -> np.ndarray:
    """Empirical coverage of the (1-alpha) central predictive interval, [H]."""
    mu, sigma, gamma, tau = shash_params_from_dist(dist_params)
    lo = shash_quantile((1.0 - alpha) / 2.0, mu, sigma, gamma, tau)
    hi = shash_quantile(1.0 - (1.0 - alpha) / 2.0, mu, sigma, gamma, tau)
    return ((target >= lo) & (target <= hi)).mean(axis=0)


def compute_all_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    extreme_quantile: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute all extended metrics on [N, H] arrays.

    Returns a dict with per-horizon arrays for each metric.
    """
    base = horizon_metrics(prediction, target)
    corr = compute_correlation(prediction, target)

    return {
        "mae": base["mae"],
        "rmse": base["rmse"],
        "bias": base["bias"],
        "correlation": corr,
        "r2": r2_score(prediction, target),
        "nse": nse(prediction, target),
        "acc": acc(prediction, target),
        "kge": kge(prediction, target),
        "explained_variance": explained_variance(prediction, target),
        "smape": smape(prediction, target),
        "extreme_recall": extreme_recall(prediction, target, quantile=extreme_quantile),
        "trend_score": trend_score(prediction, target),
        "ioa": ioa(prediction, target),
        "max_error": max_error(prediction, target),
        "median_abs_error": median_abs_error(prediction, target),
        "relative_rmse": relative_rmse(prediction, target),
        "relative_bias": relative_bias(prediction, target),
    }
