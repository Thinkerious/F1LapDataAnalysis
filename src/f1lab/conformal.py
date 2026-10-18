"""Calibrated prediction intervals by split-conformal methods.

A point forecast says "1.2 s faster than base pace". These intervals say "and 80%
of the time the true value lies in this range", and the claim is checked against
held-out races instead of assumed.

Calibration scores come from cross-validation grouped by race (cross-conformal), so
every training lap gets a score from a model that never saw its race.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .models import make_gbm, make_quantile_gbm


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.

    If the sample is too small to support the requested level the honest answer
    is an unbounded interval, so this returns infinity.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else float(np.sort(scores)[k - 1])


class ConstantWidth:
    """Point model plus a fixed half-width from out-of-fold absolute residuals."""

    def __init__(self, alpha: float, n_splits: int = 5, seed: int = 0):
        self.alpha, self.n_splits, self.seed = alpha, n_splits, seed

    def fit(self, X: pd.DataFrame, y: pd.Series, groups) -> "ConstantWidth":
        oof = np.zeros(len(y))
        for fit_idx, cal_idx in GroupKFold(self.n_splits).split(X, y, groups):
            model = make_gbm(seed=self.seed).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            oof[cal_idx] = model.predict(X.iloc[cal_idx])
        self.q_ = conformal_quantile(np.abs(y.to_numpy() - oof), self.alpha)
        self.model_ = make_gbm(seed=self.seed).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.model_.predict(X)
        return p - self.q_, p + self.q_


class CQR:
    """Conformalized quantile regression: quantile models widened by a calibrated margin.

    Widths adapt to the input (wider where the model is less sure) and the
    interval can be asymmetric, which suits a target with a long slow tail.
    """

    def __init__(self, alpha: float, n_splits: int = 5, seed: int = 0):
        self.alpha, self.n_splits, self.seed = alpha, n_splits, seed

    def _quantiles(self):
        return self.alpha / 2, 1 - self.alpha / 2

    def fit(self, X: pd.DataFrame, y: pd.Series, groups) -> "CQR":
        lo_a, hi_a = self._quantiles()
        yv = y.to_numpy()
        scores = np.zeros(len(y))
        for fit_idx, cal_idx in GroupKFold(self.n_splits).split(X, y, groups):
            lo = make_quantile_gbm(lo_a, seed=self.seed).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            hi = make_quantile_gbm(hi_a, seed=self.seed).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            scores[cal_idx] = np.maximum(
                lo.predict(X.iloc[cal_idx]) - yv[cal_idx], yv[cal_idx] - hi.predict(X.iloc[cal_idx])
            )
        self.q_ = conformal_quantile(scores, self.alpha)
        self.lo_ = make_quantile_gbm(lo_a, seed=self.seed).fit(X, y)
        self.hi_ = make_quantile_gbm(hi_a, seed=self.seed).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.lo_.predict(X) - self.q_, self.hi_.predict(X) + self.q_


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(((y >= lo) & (y <= hi)).mean())


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
