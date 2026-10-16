import pandas as pd

from f1lab.clean import clean_laps
from f1lab.schema import COLUMNS


def _row(**over):
    base = {
        "season": 2024, "round": 1, "track": "Sakhir", "driver": "VER", "lap_number": 5,
        "lap_time_s": 95.0, "compound": "SOFT", "tyre_age": 4, "stint": 1,
        "is_pit_in": False, "is_pit_out": False, "track_temp": 30.0, "air_temp": 20.0,
        "rainfall": False, "track_status": "green", "source": "openf1",
    }
    base.update(over)
    return base


def test_each_rule_drops_only_its_lap_and_report_counts_them():
    rows = [
        _row(),  # kept
        _row(is_pit_in=True),
        _row(is_pit_out=True),
        _row(lap_number=1),
        _row(lap_time_s=None),
        _row(track_status="sc"),
        _row(rainfall=True),
        _row(compound="INTERMEDIATE"),
        _row(compound="None"),
        _row(tyre_age=pd.NA),
    ]
    df = pd.DataFrame(rows)[COLUMNS]
    df["tyre_age"] = df["tyre_age"].astype("Int64")

    clean, report = clean_laps(df)

    assert len(clean) == 1
    removed = dict(zip(report["rule"], report["removed"]))
    assert removed["pit-in or pit-out lap"] == 2
    assert removed["lap 1"] == 1
    assert removed["no lap time"] == 1
    assert removed["safety car, vsc or red flag"] == 1
    assert removed["rain"] == 1
    assert removed["not a slick compound"] == 2  # intermediate and the "None" string
    assert removed["no tyre age"] == 1
    assert report["remaining"].iloc[-1] == 1


def test_extreme_slow_laps_are_dropped_relative_to_their_own_race():
    same_race = [_row(lap_number=n, lap_time_s=95.0) for n in range(2, 8)]
    slow = _row(lap_number=9, lap_time_s=95.0 * 1.5)
    other_race_slow_track = [_row(round=2, lap_number=n, lap_time_s=150.0) for n in range(2, 8)]
    df = pd.DataFrame(same_race + [slow] + other_race_slow_track)[COLUMNS]
    df["tyre_age"] = df["tyre_age"].astype("Int64")

    clean, report = clean_laps(df)

    removed = dict(zip(report["rule"], report["removed"]))
    assert removed["extreme slow lap"] == 1  # a 150s lap is normal at a slower track
    assert len(clean) == 12
