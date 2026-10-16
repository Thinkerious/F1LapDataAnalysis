"""OpenF1 loader (https://openf1.org), used for 2023 onward.

Raw JSON responses are cached on disk so each endpoint is only fetched once.
"""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

from .schema import finalize

BASE = "https://api.openf1.org/v1"
CACHE_DIR = Path("data/openf1_cache")
MIN_GAP_S = 0.4  # be polite to a free API
_last_call = 0.0


class NoRaceData(ValueError):
    pass


def get(endpoint: str, **params) -> list[dict]:
    """GET an OpenF1 endpoint, cached by endpoint and params."""
    global _last_call
    key = hashlib.md5(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{endpoint}_{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    for attempt in range(5):
        wait = MIN_GAP_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        if r.status_code == 404:  # OpenF1 uses 404 for "no rows"
            data = []
            break
        if r.status_code in (429, 500, 502, 503):
            time.sleep(3 * 2**attempt)  # 3, 6, 12, 24, 48 s; 429s clear slowly
            continue
        r.raise_for_status()
        data = r.json()
        break
    else:
        raise RuntimeError(f"OpenF1 {endpoint} kept failing: {r.status_code}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def list_races(year: int) -> pd.DataFrame:
    """Race sessions for a season in calendar order, with a 1-based round number."""
    sessions = pd.DataFrame(get("sessions", year=year, session_name="Race"))
    sessions = sessions.sort_values("date_start").reset_index(drop=True)
    sessions["round"] = sessions.index + 1
    return sessions[["session_key", "round", "circuit_short_name", "year"]]


_NEUTRALISATIONS = {
    "sc": ("SAFETY CAR DEPLOYED", "SAFETY CAR IN THIS LAP"),
    "vsc": ("VIRTUAL SAFETY CAR DEPLOYED", "VIRTUAL SAFETY CAR ENDING"),
}


def status_by_lap(race_control: pd.DataFrame) -> dict[int, str]:
    """Map lap number to its worst track status (sc, vsc or red) from race control.

    A safety-car or VSC period covers the deploy lap through the end lap
    inclusive. Measured on 2023-2025 pace: deploy lap ~1.3x race median, middle
    ~1.5x, end lap ~1.5x (SC) or ~1.08x (VSC), and the lap after is back to
    ~1.03x, so the restart lap needs no special treatment. A deployment with no
    matching end (it ended under a red flag) runs to the next red flag lap, or
    covers just its deploy lap if there is none. Lap numbers are the race
    leader's, so this is approximate for lapped cars.
    """
    if race_control.empty:
        return {}
    rc = race_control.dropna(subset=["lap_number"]).sort_values("date")
    red_laps = sorted(
        int(n) for n in rc.loc[rc["flag"] == "RED", "lap_number"]
    )
    rank = {"vsc": 1, "sc": 2, "red": 3}
    status: dict[int, str] = {}

    def mark(lap: int, kind: str) -> None:
        if rank[kind] > rank.get(status.get(lap), 0):
            status[lap] = kind

    for kind, (deploy, end) in _NEUTRALISATIONS.items():
        start = None
        for message, lap in zip(rc["message"], rc["lap_number"].astype(int)):
            if message == deploy and start is None:
                start = lap
            elif message == end and start is not None:
                for n in range(start, lap + 1):
                    mark(n, kind)
                start = None
        if start is not None:  # never closed
            later_red = [n for n in red_laps if n >= start]
            for n in range(start, (later_red[0] if later_red else start) + 1):
                mark(n, kind)
    for lap in red_laps:
        mark(lap, "red")
    return status


def build_laps(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    weather: pd.DataFrame,
    drivers: pd.DataFrame,
    race_control: pd.DataFrame | None = None,
    *,
    season: int,
    round_: int,
    track: str,
) -> pd.DataFrame:
    """Join the raw OpenF1 tables into the common lap schema (no network)."""
    df = laps.copy()
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True, format="ISO8601")

    # stint and compound: a lap belongs to the stint whose lap range contains it
    merged = df.merge(stints, on="driver_number", how="left")
    in_stint = (merged["lap_number"] >= merged["lap_start"]) & (
        merged["lap_number"] <= merged["lap_end"]
    )
    merged = merged[in_stint | merged["stint_number"].isna()]
    merged = merged.drop_duplicates(["driver_number", "lap_number"])
    merged["tyre_age"] = (
        merged["tyre_age_at_start"] + (merged["lap_number"] - merged["lap_start"]) + 1
    )

    # Pit-in laps: the last lap of every stint except a driver's final one. The
    # `pit` endpoint is empty for early 2023, and this matched it exactly (43/43
    # stops) on 2024 round 1, so stints are the single source.
    last_stint = stints.groupby("driver_number")["stint_number"].transform("max")
    ended = stints[stints["stint_number"] < last_stint]
    pit_in = set(zip(ended["driver_number"], ended["lap_end"]))
    merged["is_pit_in"] = [
        (d, n) in pit_in for d, n in zip(merged["driver_number"], merged["lap_number"])
    ]
    merged["is_pit_out"] = merged["is_pit_out_lap"].fillna(False)

    # weather: latest reading at or before the lap started
    wx = weather.copy()
    wx["date"] = pd.to_datetime(wx["date"], utc=True, format="ISO8601")
    wx = wx.sort_values("date")
    merged = merged.dropna(subset=["date_start"]).sort_values("date_start")
    merged = pd.merge_asof(merged, wx, left_on="date_start", right_on="date", direction="backward")

    merged = merged.merge(drivers[["driver_number", "name_acronym"]], on="driver_number", how="left")

    lap_status = status_by_lap(race_control if race_control is not None else pd.DataFrame())

    out = pd.DataFrame(
        {
            "season": season,
            "round": round_,
            "track": track,
            "driver": merged["name_acronym"],
            "lap_number": merged["lap_number"],
            "lap_time_s": merged["lap_duration"],
            "compound": merged["compound"],
            "tyre_age": merged["tyre_age"],
            "stint": merged["stint_number"],
            "is_pit_in": merged["is_pit_in"],
            "is_pit_out": merged["is_pit_out"],
            "track_temp": merged["track_temperature"],
            "air_temp": merged["air_temperature"],
            "rainfall": merged["rainfall"].fillna(0) > 0,
            "track_status": merged["lap_number"].map(lap_status).fillna("green"),
            "source": "openf1",
        }
    )
    return finalize(out)


def load_race_laps(year: int, round_: int) -> pd.DataFrame:
    races = list_races(year)
    row = races[races["round"] == round_].iloc[0]
    sk = int(row["session_key"])
    frames = {
        name: pd.DataFrame(get(name, session_key=sk))
        for name in ("laps", "stints", "weather", "drivers", "race_control")
    }
    if frames["laps"].empty:
        # e.g. 2023 Imola: listed as a race but cancelled, so every endpoint is empty
        raise NoRaceData(f"{year} round {round_} ({row['circuit_short_name']}) has no lap data")
    return build_laps(
        frames["laps"],
        frames["stints"],
        frames["weather"],
        frames["drivers"],
        frames["race_control"],
        season=year,
        round_=round_,
        track=row["circuit_short_name"],
    )
