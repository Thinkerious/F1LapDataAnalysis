import numpy as np
import pandas as pd

from f1lab.features import drying_races, mae, prepare, rmse, split


def _laps(season, rnd, early_time, late_time, driver="VER"):
    rows = []
    for n in range(2, 31):
        rows.append(
            {
                "season": season, "round": rnd, "driver": driver, "lap_number": n,
                "lap_time_s": early_time if n <= 10 else late_time, "tyre_age": n,
                "compound": "SOFT", "track": "T", "track_temp": 30.0, "air_temp": 20.0,
            }
        )
    return rows


def test_drying_races_flags_only_slow_start_races():
    dry = pd.DataFrame(_laps(2024, 1, 90.0, 90.0))
    wet = pd.DataFrame(_laps(2024, 2, 120.0, 100.0))  # 1.2x slower early
    assert drying_races(pd.concat([dry, wet])) == {(2024, 2)}


def test_prepare_target_is_lap_minus_driver_race_median_and_fuel_proxy():
    clean = pd.DataFrame(_laps(2024, 1, 92.0, 92.0) + _laps(2024, 1, 94.0, 94.0, driver="HAM"))
    clean["lap_time_s"] += np.where(clean["lap_number"] == 20, 1.0, 0.0)
    raw = clean.copy()
    raw.loc[len(raw)] = {**raw.iloc[0], "lap_number": 50}  # race ran 50 laps

    df = prepare(raw, clean)

    ver = df[(df["driver"] == "VER") & (df["lap_number"] == 20)].iloc[0]
    assert ver["delta_s"] == 1.0  # median is 92.0, this lap is 93.0
    assert ver["laps_remaining"] == 30
    ham = df[(df["driver"] == "HAM") & (df["lap_number"] == 5)].iloc[0]
    assert ham["delta_s"] == 0.0  # car pace removed: HAM is not penalised for being slower


def test_split_never_mixes_seasons_between_train_and_evaluation():
    df = pd.DataFrame({"season": [2022, 2023, 2024, 2025]})
    train, ev = split(df, "dev")
    assert train["season"].tolist() == [2022, 2023] and ev["season"].tolist() == [2024]
    train, ev = split(df, "final")
    assert train["season"].tolist() == [2022, 2023, 2024] and ev["season"].tolist() == [2025]


def test_error_metrics():
    y, p = pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 1.0, 5.0])
    assert mae(y, p) == 1.0
    assert abs(rmse(y, p) - (5 / 3) ** 0.5) < 1e-9


def test_prepare_drops_driver_races_with_too_few_clean_laps():
    full = _laps(2024, 1, 92.0, 92.0)  # 29 laps for VER
    short = [row for row in _laps(2024, 1, 92.0, 92.0, driver="HAM")][:5]  # only 5 laps
    clean = pd.DataFrame(full + short)

    df = prepare(clean, clean)

    assert set(df["driver"]) == {"VER"}


def test_linear_features_split_wear_by_compound_and_scale_fuel_by_race_length():
    from f1lab.models import linear_features

    d = pd.DataFrame(
        {
            "laps_remaining": [30, 0], "race_laps": [60, 50], "compound": ["SOFT", "HARD"],
            "tyre_age": [10.0, 4.0], "track_temp": [30.0, 35.0], "track": ["A", "B"],
        }
    )
    x = linear_features(d, tracks=["A", "B"])
    assert x["fuel_frac"].tolist() == [0.5, 0.0]
    assert x["wear_SOFT"].tolist() == [10.0, 0.0]  # wear only counts on its own compound
    assert x["wear_HARD"].tolist() == [0.0, 4.0]
    assert x["track_A"].tolist() == [1.0, 0.0]


def test_gbm_features_mark_unseen_track_as_missing_and_keep_compound_order():
    from f1lab.models import gbm_features

    d = pd.DataFrame(
        {
            "laps_remaining": [10, 20], "race_laps": [50, 50], "lap_number": [40, 30],
            "tyre_age": [5.0, 9.0], "track_temp": [30.0, 31.0], "air_temp": [20.0, 21.0],
            "compound": ["HARD", "SOFT"], "track": ["Monza", "Shanghai"],
        }
    )
    x = gbm_features(d, tracks=["Monza"])
    assert x["track"].isna().tolist() == [False, True]
    assert list(x["compound"].cat.categories) == ["SOFT", "MEDIUM", "HARD"]
    assert x["fuel_frac"].tolist() == [0.2, 0.4]
