"""Safety car and virtual safety car events, and a simple model of when they start.

A "neutralisation" here is a safety car or VSC period. The hazard model answers
one question: given the race is running under green flags, how likely is a
neutralisation to start on this lap?

There are only about 80 events in the dry races of four seasons, so the model is
deliberately small: a separate rate for the first two laps, a flat rate after
that, and a per-track multiplier that is pulled hard toward the average.
"""

import numpy as np
import pandas as pd

RANK = {"green": 0, "vsc": 1, "sc": 2, "red": 3}
EARLY_LAPS = 2
MIN_CARS = 5  # ignore race laps where fewer cars have a status (the end of the race)


def lap_status(raw: pd.DataFrame, skip: set[tuple[int, int]] = frozenset()) -> pd.DataFrame:
    """Worst track status over all drivers for every lap of every race."""
    d = raw.assign(rank=raw["track_status"].map(RANK))
    lap = (
        d.groupby(["season", "round", "lap_number"])
        .agg(rank=("rank", "max"), cars=("rank", "size"), track=("track", "first"))
        .reset_index()
    )
    lap = lap[lap["cars"] >= MIN_CARS]
    keep = [(s, r) not in skip for s, r in zip(lap["season"], lap["round"])]
    return lap[keep].sort_values(["season", "round", "lap_number"]).reset_index(drop=True)


def neutralisation_events(lap: pd.DataFrame) -> pd.DataFrame:
    """One row per safety car / VSC period: start lap, length in laps, and kind."""
    rows = []
    for (season, rnd), g in lap.groupby(["season", "round"]):
        race_laps, track = int(g["lap_number"].max()), g["track"].iloc[0]
        prev, start, worst = 0, None, 0
        for lap_no, rank in zip(g["lap_number"], g["rank"]):
            under = rank in (1, 2)
            if under and start is None and prev == 0:
                start, worst = lap_no, rank
            elif under and start is not None:
                worst = max(worst, rank)
            elif not under and start is not None:
                rows.append((season, rnd, track, race_laps, start, lap_no - start, "sc" if worst == 2 else "vsc"))
                start = None
            prev = rank
        if start is not None:
            rows.append((season, rnd, track, race_laps, start, race_laps - start + 1, "sc" if worst == 2 else "vsc"))
    return pd.DataFrame(rows, columns=["season", "round", "track", "race_laps", "start", "length", "kind"])


def hazard_table(lap: pd.DataFrame) -> pd.DataFrame:
    """One row per lap that was at risk (race under green the lap before)."""
    rows = []
    for (season, rnd), g in lap.groupby(["season", "round"]):
        race_laps, track = int(g["lap_number"].max()), g["track"].iloc[0]
        prev = 0
        for lap_no, rank in zip(g["lap_number"], g["rank"]):
            if prev == 0:
                rows.append((season, rnd, track, lap_no, race_laps, int(rank in (1, 2))))
            prev = rank
    out = pd.DataFrame(rows, columns=["season", "round", "track", "lap", "race_laps", "event"])
    out["early"] = out["lap"] <= EARLY_LAPS
    return out


class Hazard:
    """P(neutralisation starts on a lap | race under green the lap before).

    `k` is the strength of the pull toward the average track: a track's multiplier
    is (events + k) / (expected events + k), so a track with little history stays
    close to 1.
    """

    def __init__(self, k: float = 2.0):
        self.k = k

    def fit(self, table: pd.DataFrame) -> "Hazard":
        early = table[table["early"]]
        late = table[~table["early"]]
        self.rate_early_ = float(early["event"].mean())
        self.rate_late_ = float(late["event"].mean())
        expected = late.groupby("track")["event"].apply(lambda s: len(s) * self.rate_late_)
        observed = late.groupby("track")["event"].sum()
        self.track_multiplier_ = ((observed + self.k) / (expected + self.k)).to_dict()
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        mult = table["track"].map(self.track_multiplier_).fillna(1.0).to_numpy()
        return np.where(table["early"], self.rate_early_, self.rate_late_ * mult)

    def rate(self, lap: int, track: str) -> float:
        if lap <= EARLY_LAPS:
            return self.rate_early_
        return self.rate_late_ * self.track_multiplier_.get(track, 1.0)


def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
