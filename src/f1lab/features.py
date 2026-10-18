"""Modelling table, target and train/validation/test split.

Target: `delta_s`, a lap's time minus the driver's own median clean lap in that
race. That removes car, driver and track pace, so what is left is tyre, fuel and
conditions. In real use the base pace would come from practice or qualifying;
using the race median here means results answer "how does pace move around the
driver's base pace", not "what is the absolute lap time".
"""

import pandas as pd

# Races whose early laps are much slower than their later laps are wet or drying
# starts the rain flag missed (it said dry for all of 2025 Spa). Natural break in
# the data: five races sit at 1.10 or above, the next is 1.04.
DRYING_RATIO = 1.05

# Races where a large share of laps were run on intermediate or wet tyres. Based
# on tyre choice, so it does not depend on the rain flag (unreliable) or on clean
# early laps existing (a wet start leaves none, which made DRYING_RATIO silently
# skip Montreal 2024). Natural break: eleven races sit at 0.24 or above, the next
# is Spa 2025 at 0.02. Added after inspecting 2024 validation coverage failures;
# it does not change which 2025 test races are excluded.
WET_TYRE_SHARE = 0.2

# A driver's median pace needs enough clean laps to be a stable base. Only 163
# laps sit in driver-races under this limit, so dropping them costs almost nothing.
MIN_DRIVER_LAPS = 10

DEV_TRAIN = (2022, 2023)
VALIDATION = 2024
FINAL_TRAIN = (2022, 2023, 2024)
TEST = 2025


def drying_races(clean: pd.DataFrame) -> set[tuple[int, int]]:
    """Races where median pace on laps 2-10 is over DRYING_RATIO x median on laps 11+."""
    key = ["season", "round"]
    early = clean[clean["lap_number"].between(2, 10)].groupby(key)["lap_time_s"].median()
    late = clean[clean["lap_number"] > 10].groupby(key)["lap_time_s"].median()
    ratio = (early / late).dropna()
    return set(ratio[ratio > DRYING_RATIO].index)


def wet_races(raw: pd.DataFrame) -> set[tuple[int, int]]:
    """Races where over WET_TYRE_SHARE of laps were on intermediates or wets."""
    key = ["season", "round"]
    wet = raw["compound"].isin(["INTERMEDIATE", "WET"]).groupby([raw["season"], raw["round"]]).mean()
    return set(wet[wet >= WET_TYRE_SHARE].index)


def prepare(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Add the target and fuel proxy to the cleaned laps and drop wet and drying races."""
    key = ["season", "round"]
    skip = drying_races(clean) | wet_races(raw)
    df = clean[[(s, r) not in skip for s, r in zip(clean["season"], clean["round"])]].copy()

    race_laps = raw.groupby(key)["lap_number"].max().rename("race_laps")
    df = df.join(race_laps, on=key)
    df["laps_remaining"] = df["race_laps"] - df["lap_number"]
    df["tyre_age"] = df["tyre_age"].astype(float)

    group = df.groupby(key + ["driver"])["lap_time_s"]
    df = df[group.transform("size") >= MIN_DRIVER_LAPS].copy()
    df["delta_s"] = df["lap_time_s"] - df.groupby(key + ["driver"])["lap_time_s"].transform("median")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, stage: str = "dev") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, evaluation) frames.

    "dev": train 2022-2023, evaluate on 2024. Used for every modelling decision.
    "final": train 2022-2024, evaluate on 2025. Touched once, at the end.
    """
    if stage == "dev":
        return df[df["season"].isin(DEV_TRAIN)], df[df["season"] == VALIDATION]
    if stage == "final":
        return df[df["season"].isin(FINAL_TRAIN)], df[df["season"] == TEST]
    raise ValueError(stage)


def mae(y, pred) -> float:
    return float((y - pred).abs().mean())


def rmse(y, pred) -> float:
    return float(((y - pred) ** 2).mean() ** 0.5)
