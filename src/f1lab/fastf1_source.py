"""FastF1 loader (https://docs.fastf1.dev), used for 2022.

FastF1 reads F1's timing archive. From this machine it only worked for 2021 and
2022, so newer seasons come from OpenF1 instead (see openf1.py).
"""

import warnings
from pathlib import Path

import fastf1
import pandas as pd

from .schema import finalize

CACHE_DIR = Path("data/fastf1_cache")


def build_laps(
    laps: pd.DataFrame, weather: pd.DataFrame, *, season: int, round_: int, track: str
) -> pd.DataFrame:
    """Normalize FastF1 `Laps` plus per-lap weather into the common schema (no network)."""
    laps = laps.reset_index(drop=True)
    weather = weather.reset_index(drop=True)  # aligned to laps by position, not index label

    # FastF1 does not always fill PitOutTime, so also infer out-laps from the lap before
    pit_in = laps["PitInTime"].notna()
    prev_pit_in = pit_in.groupby(laps["Driver"]).shift(1, fill_value=False)
    pit_out = laps["PitOutTime"].notna() | prev_pit_in

    out = pd.DataFrame(
        {
            "season": season,
            "round": round_,
            "track": track,
            "driver": laps["Driver"],
            "lap_number": laps["LapNumber"],
            "lap_time_s": laps["LapTime"].dt.total_seconds(),
            "compound": laps["Compound"],
            "tyre_age": laps["TyreLife"],
            "stint": laps["Stint"],
            "is_pit_in": pit_in,
            "is_pit_out": pit_out,
            "track_temp": weather["TrackTemp"],
            "air_temp": weather["AirTemp"],
            "rainfall": weather["Rainfall"].fillna(False),
            "source": "fastf1",
        }
    )
    return finalize(out)


def load_race_laps(year: int, round_: int) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, round_, "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=True, messages=False)
    laps = session.laps
    return build_laps(
        laps,
        laps.get_weather_data(),
        season=year,
        round_=round_,
        track=session.event["Location"],
    )
