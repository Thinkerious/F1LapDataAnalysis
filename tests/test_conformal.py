import numpy as np
import pandas as pd

from f1lab.conformal import CQR, ConstantWidth, conformal_quantile, coverage, mean_width


def test_conformal_quantile_uses_finite_sample_rank_and_goes_infinite_when_too_small():
    scores = np.arange(1.0, 100.0)  # n = 99
    assert conformal_quantile(scores, 0.1) == 90.0  # ceil(100 * 0.9) = 90th smallest
    assert conformal_quantile(np.array([1.0, 2.0]), 0.1) == float("inf")  # cannot support 90%


def _noisy(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = 2 * x + rng.normal(0, 0.2 + 0.8 * x)  # noise grows with x
    X = pd.DataFrame(
        {"fuel_frac": x, "laps_remaining": x * 50, "race_laps": 50.0, "lap_number": 50 * (1 - x),
         "tyre_age": rng.uniform(1, 30, n), "track_temp": 30.0, "air_temp": 20.0,
         "compound": pd.Categorical(["SOFT"] * n, categories=["SOFT", "MEDIUM", "HARD"]),
         "track": pd.Categorical(["A"] * n, categories=["A"])}
    )
    return X, pd.Series(y)


def test_both_methods_cover_near_nominal_on_exchangeable_data():
    X, y = _noisy(6000, 0)
    groups = np.arange(len(y)) // 100  # 60 pseudo-races
    Xt, yt = _noisy(6000, 1)
    for method in (ConstantWidth(0.2), CQR(0.2)):
        method.fit(X, y, groups)
        lo, hi = method.predict(Xt)
        assert abs(coverage(yt, lo, hi) - 0.8) < 0.04, type(method).__name__


def test_cqr_is_narrower_where_noise_is_low_than_where_it_is_high():
    X, y = _noisy(6000, 0)
    cqr = CQR(0.2).fit(X, y, np.arange(len(y)) // 100)
    lo, hi = cqr.predict(X)
    width = pd.Series(hi - lo)
    assert width[X["fuel_frac"] < 0.2].mean() < width[X["fuel_frac"] > 0.8].mean()


def test_coverage_and_width_helpers():
    assert coverage([1, 2, 3, 4], [0, 0, 0, 0], [2, 2, 2, 2]) == 0.5
    assert mean_width([0, 1], [2, 4]) == 2.5
