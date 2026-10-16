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
