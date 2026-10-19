"""Replay real races through the simulator and score the predictions.

For every driver who completed the full race distance with every lap timed, rebuild
what actually happened (pit laps, compounds, how old each set was, the observed
safety car timeline) and ask the simulator for the total race time. Compare with the
real total.

Limits worth knowing: base pace is the driver's own median clean lap from that same
race, which is only known in hindsight, and the safety car timeline is the observed
one. So this tests the lap-time model, pit costs and neutralised-lap costs, not the
ability to forecast pace or to predict safety cars.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import drying_races, wet_races
from .models import COMPOUNDS
from .safety_car import RANK, lap_status
from .simulate import NoModel, Simulator, Stint

KEY = ["season", "round"]


@dataclass
class DriverRace:
    season: int
    round: int
    driver: str
    track: str
    race_laps: int
    base_pace: float
    track_temp: float
    air_temp: float
    strategy: list
    status: np.ndarray
    actual_total: float


def driver_races(raw: pd.DataFrame, clean: pd.DataFrame, seasons) -> list[DriverRace]:
    skip = drying_races(clean) | wet_races(raw)
    pace = clean.groupby(KEY + ["driver"])["lap_time_s"].agg(["median", "size"])
    out = []
    for (season, rnd), race in raw[raw["season"].isin(seasons)].groupby(KEY):
        if (season, rnd) in skip:
            continue
        race_laps = int(race["lap_number"].max())
        timeline = lap_status(race).set_index("lap_number")["rank"]
        timeline = timeline.reindex(range(1, race_laps + 1)).fillna(0)
        if (timeline == RANK["red"]).any():
            continue  # stopped races break the lap-by-lap bookkeeping
        status = np.minimum(timeline.to_numpy(dtype=int), 2)
        temps = race[["track_temp", "air_temp"]].mean()
        for driver, laps in race.groupby("driver"):
            laps = laps.sort_values("lap_number")
            if len(laps) != race_laps or laps["lap_time_s"].isna().any():
                continue
            if (season, rnd, driver) not in pace.index or pace.loc[(season, rnd, driver), "size"] < 10:
                continue
            stops = laps.loc[laps["is_pit_in"] & (laps["lap_number"] < race_laps), "lap_number"].tolist()
            edges = [0, *stops, race_laps]
            stints = []
            for a, b in zip(edges, edges[1:]):
                chunk = laps[(laps["lap_number"] > a) & (laps["lap_number"] <= b)]
                compound = chunk["compound"].mode().iloc[0]
                if compound not in COMPOUNDS:
                    stints = None
                    break
                age = chunk["tyre_age"].iloc[0]
                stints.append(Stint(compound, b - a, int(age) if pd.notna(age) else 1))
            if not stints or any(s.laps < 1 for s in stints):
                continue
            out.append(DriverRace(
                int(season), int(rnd), driver, laps["track"].iloc[0], race_laps,
                float(pace.loc[(season, rnd, driver), "median"]), float(temps["track_temp"]),
                float(temps["air_temp"]), stints, status, float(laps["lap_time_s"].sum()),
            ))
    return out


def replay(sim: Simulator, plans: list[DriverRace], n: int = 300, seed: int = 0, level: float = 0.8) -> pd.DataFrame:
    """Simulated total, interval and error for each driver-race (observed safety car timeline)."""
    rows = []
    for i, p in enumerate(plans):
        r = sim.simulate(p.track, p.race_laps, p.base_pace, p.strategy, p.track_temp, p.air_temp,
                         n=n, seed=seed + i, status=p.status)
        s = r.summary(level)
        rows.append({"season": p.season, "round": p.round, "driver": p.driver, "track": p.track,
                     "stops": len(p.strategy) - 1, "actual": p.actual_total, "predicted": s["mean"],
                     "lo": s["lo"], "hi": s["hi"]})
    out = pd.DataFrame(rows)
    out["error"] = out["actual"] - out["predicted"]
    out["inside"] = (out["actual"] >= out["lo"]) & (out["actual"] <= out["hi"])
    return out


def constant_pace_version(sim: Simulator) -> Simulator:
    """Same costs and noise, but no tyre or fuel effect on green laps."""
    return Simulator(NoModel(), sim.tracks, sim.pit_loss, sim.hazard, sim.pools)


def score(result: pd.DataFrame) -> dict:
    e = result["error"]
    return {"median abs error (s)": float(e.abs().median()), "mean error (s)": float(e.mean()),
            "RMSE (s)": float((e ** 2).mean() ** 0.5), "inside 80% interval": float(result["inside"].mean())}


def rank_agreement(result: pd.DataFrame) -> float:
    """Mean over races of the rank correlation between predicted and actual totals."""
    per_race = result.groupby(KEY).apply(
        lambda g: g["predicted"].corr(g["actual"], method="spearman") if len(g) >= 8 else np.nan,
        include_groups=False,
    )
    return float(per_race.mean())
