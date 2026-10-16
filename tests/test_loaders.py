import pandas as pd

from f1lab import fastf1_source, openf1
from f1lab.schema import COLUMNS


def test_openf1_build_laps_assigns_stint_age_pit_and_weather():
    laps = pd.DataFrame(
        {
            "driver_number": [1, 1, 1],
            "lap_number": [1, 2, 3],
            "lap_duration": [95.0, 92.5, 100.0],
            "is_pit_out_lap": [False, False, True],
            "date_start": [
                "2024-03-02T15:00:00+00:00",
                "2024-03-02T15:01:35+00:00",
                "2024-03-02T15:03:07+00:00",
            ],
        }
    )
    stints = pd.DataFrame(
        {
            "driver_number": [1, 1],
            "stint_number": [1, 2],
            "lap_start": [1, 3],
            "lap_end": [2, 5],
            "compound": ["SOFT", "HARD"],
            "tyre_age_at_start": [0, 0],
        }
    )
    weather = pd.DataFrame(
        {
            "date": ["2024-03-02T14:59:00+00:00", "2024-03-02T15:02:00+00:00"],
            "track_temperature": [30.0, 31.0],
            "air_temperature": [20.0, 21.0],
            "rainfall": [0, 0],
        }
    )
    drivers = pd.DataFrame({"driver_number": [1], "name_acronym": ["VER"]})

    df = openf1.build_laps(
        laps, stints, weather, drivers, season=2024, round_=1, track="Sakhir"
    )

    assert list(df.columns) == COLUMNS
    assert df["driver"].tolist() == ["VER"] * 3
    assert df["compound"].tolist() == ["SOFT", "SOFT", "HARD"]
    assert df["tyre_age"].tolist() == [1, 2, 1]  # first lap on a new set is age 1
    assert df["is_pit_in"].tolist() == [False, True, False]
    assert df["is_pit_out"].tolist() == [False, False, True]
    assert df["track_temp"].tolist() == [30.0, 30.0, 31.0]  # latest reading before lap start


def test_fastf1_build_laps_infers_out_lap_and_converts_times():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER"],
            "LapNumber": [1.0, 2.0, 3.0],
            "LapTime": pd.to_timedelta([95.0, 92.5, 100.0], unit="s"),
            "Compound": ["SOFT", "SOFT", "HARD"],
            "TyreLife": [1.0, 2.0, 1.0],
            "Stint": [1.0, 1.0, 2.0],
            "TrackStatus": ["1", "124", "6"],
            "PitInTime": pd.to_timedelta([None, 190, None], unit="s"),
            "PitOutTime": [pd.NaT, pd.NaT, pd.NaT],  # FastF1 sometimes leaves this empty
        },
        index=[10, 11, 12],  # non-default index, weather must still line up by position
    )
    weather = pd.DataFrame(
        {
            "TrackTemp": [30.0, 30.5, 31.0],
            "AirTemp": [20.0, 20.5, 21.0],
            "Rainfall": [False, False, False],
        },
        index=[0, 1, 2],
    )

    df = fastf1_source.build_laps(laps, weather, season=2022, round_=1, track="Sakhir")

    assert list(df.columns) == COLUMNS
    assert df["lap_time_s"].tolist() == [95.0, 92.5, 100.0]
    assert df["is_pit_in"].tolist() == [False, True, False]
    assert df["is_pit_out"].tolist() == [False, False, True]  # inferred from previous pit-in
    assert df["track_temp"].tolist() == [30.0, 30.5, 31.0]
    assert df["track_status"].tolist() == ["green", "sc", "vsc"]  # worst status in the lap
    assert (df["source"] == "fastf1").all()


def _rc(rows):
    return pd.DataFrame(rows, columns=["date", "lap_number", "category", "flag", "message"])


def test_status_by_lap_covers_deploy_through_end_inclusive():
    rc = _rc(
        [
            ("2024-01-01T10:00", 10, "SafetyCar", None, "SAFETY CAR DEPLOYED"),
            ("2024-01-01T10:20", 13, "SafetyCar", None, "SAFETY CAR IN THIS LAP"),
            ("2024-01-01T10:50", 30, "SafetyCar", None, "VIRTUAL SAFETY CAR DEPLOYED"),
            ("2024-01-01T10:52", 31, "SafetyCar", None, "VIRTUAL SAFETY CAR ENDING"),
        ]
    )
    status = openf1.status_by_lap(rc)
    assert {n: status[n] for n in (10, 11, 12, 13)} == {10: "sc", 11: "sc", 12: "sc", 13: "sc"}
    assert 14 not in status and 9 not in status  # restart lap is not neutralised
    assert status[30] == "vsc" and status[31] == "vsc"


def test_status_by_lap_red_flag_wins_and_unclosed_sc_runs_to_red():
    rc = _rc(
        [
            ("2024-01-01T10:00", 5, "SafetyCar", None, "SAFETY CAR DEPLOYED"),
            ("2024-01-01T10:05", 7, "Flag", "RED", "RED FLAG"),
        ]
    )
    status = openf1.status_by_lap(rc)
    assert status == {5: "sc", 6: "sc", 7: "red"}


def test_status_by_lap_handles_empty_race_control():
    assert openf1.status_by_lap(pd.DataFrame()) == {}
