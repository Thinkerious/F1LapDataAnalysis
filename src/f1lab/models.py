"""Model inputs and the linear lap-time model."""

import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def linear_features(d: pd.DataFrame, tracks: list[str] | None = None) -> pd.DataFrame:
    """Inputs for the linear model.

    Fuel enters two ways: laps remaining, and the fraction of the race remaining
    (burn per lap scales with race length, so the fraction fits better). Each
    compound gets its own offset and its own wear slope.
    """
    x = pd.DataFrame(index=d.index)
    x["fuel_laps"] = d["laps_remaining"]
    x["fuel_frac"] = d["laps_remaining"] / d["race_laps"]
    for comp in COMPOUNDS:
        is_comp = (d["compound"] == comp).astype(float)
        x[f"offset_{comp}"] = is_comp
        x[f"wear_{comp}"] = is_comp * d["tyre_age"]
    x["track_temp"] = d["track_temp"]
    for track in tracks or []:
        x[f"track_{track}"] = (d["track"] == track).astype(float)
    return x


def make_linear(robust: bool = True):
    """Scaled linear model. Huber loss by default so incident laps do not drag the fit."""
    reg = HuberRegressor(max_iter=500) if robust else LinearRegression()
    return make_pipeline(StandardScaler(), reg)


def raw_coefficients(model, columns: list[str]) -> pd.Series:
    """Coefficients in original units (seconds per unit of each input)."""
    scaler, reg = model[0], model[-1]
    return pd.Series(reg.coef_ / scaler.scale_, index=columns)


GBM_FEATURES = [
    "fuel_frac", "laps_remaining", "race_laps", "lap_number", "tyre_age",
    "track_temp", "air_temp", "compound", "track",
]


def gbm_features(d: pd.DataFrame, tracks: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    """Inputs for gradient boosting. Compound and track are categorical.

    `tracks` is the list seen in training; a track outside it becomes missing.
    """
    x = pd.DataFrame(index=d.index)
    x["fuel_frac"] = d["laps_remaining"] / d["race_laps"]
    x["laps_remaining"] = d["laps_remaining"]
    x["race_laps"] = d["race_laps"]
    x["lap_number"] = d["lap_number"]
    x["tyre_age"] = d["tyre_age"]
    x["track_temp"] = d["track_temp"]
    x["air_temp"] = d["air_temp"]
    x["compound"] = pd.Categorical(d["compound"], categories=COMPOUNDS)
    x["track"] = pd.Categorical(d["track"], categories=tracks)
    return x[columns or GBM_FEATURES]


def make_gbm(num_leaves: int = 4, n_estimators: int = 200, seed: int = 0):
    """Small L1 (median) gradient-boosted model. Kept small on purpose: with two
    seasons of training data, grouped CV preferred the simplest settings."""
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="l1", learning_rate=0.05, n_estimators=n_estimators, num_leaves=num_leaves,
        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )
