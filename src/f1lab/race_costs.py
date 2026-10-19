"""Costs a lap-time model leaves out: pit stops, safety-car laps and the first lap.

Everything here is estimated from the lap table as empirical samples, so the
simulator can draw from the real distributions (including their long tails)
instead of using one number.
"""

import numpy as np
import pandas as pd

KEY = ["season", "round"]
STATUS_ORDER = ["green", "vsc", "sc", "red"]


def pit_stops(raw: pd.DataFrame) -> pd.DataFrame:
    """Time lost at each pit stop, in seconds.

    Loss = (in-lap + out-lap) minus what the rest of the field did on those same
    laps, using the median over cars that were not themselves in the pits on or
    just after that lap. `status` is the worst track status over the two laps.
    """
    d = raw.sort_values(KEY + ["driver", "lap_number"]).copy()
    d["after_pit"] = d.groupby(KEY + ["driver"])["is_pit_in"].shift(1, fill_value=False)
    free = d[~d["is_pit_in"] & ~d["after_pit"] & ~d["is_pit_out"] & d["lap_time_s"].notna()]
    ref = free.groupby(KEY + ["lap_number"])["lap_time_s"].median().rename("ref")
    d = d.join(ref, on=KEY + ["lap_number"])

    rows = []
    for (season, rnd, driver), g in d.groupby(KEY + ["driver"]):
        g = g.set_index("lap_number")
        for lap in g.index[g["is_pit_in"].to_numpy()]:
            if lap + 1 not in g.index:
                continue
            a, b = g.loc[lap], g.loc[lap + 1]
            if pd.isna([a["lap_time_s"], b["lap_time_s"], a["ref"], b["ref"]]).any():
                continue
            status = max(a["track_status"], b["track_status"], key=STATUS_ORDER.index)
            loss = (a["lap_time_s"] + b["lap_time_s"]) - (a["ref"] + b["ref"])
            rows.append((season, rnd, a["track"], driver, lap, status, loss))
    return pd.DataFrame(rows, columns=KEY + ["track", "driver", "lap", "status", "loss"])


class PitLoss:
    """Pit stop loss: a per-track median (pulled toward the overall median) plus noise.

    Under a safety car or VSC the loss depends mostly on when the field slows
    relative to the stop, so those stops are sampled from the pooled history
    instead of being modelled per track.
    """

    GREEN_RANGE = (10.0, 45.0)  # drops red-flag and stop-and-go outliers from the green pool

    def __init__(self, k: float = 10.0):
        self.k = k

    def fit(self, stops: pd.DataFrame) -> "PitLoss":
        green = stops[(stops["status"] == "green") & stops["loss"].between(*self.GREEN_RANGE)]
        self.overall_ = float(green["loss"].median())
        by = green.groupby("track")["loss"].agg(["median", "size"])
        # shrink: a track with few stops stays close to the overall median
        self.track_median_ = ((by["median"] * by["size"] + self.overall_ * self.k) / (by["size"] + self.k)).to_dict()
        self.green_noise_ = (green["loss"] - green["track"].map(self.track_median_)).to_numpy()
        neutral = stops[stops["status"].isin(["sc", "vsc"])]
        self.neutral_pool_ = neutral["loss"].clip(0, 60).to_numpy()
        return self

    def median(self, track: str) -> float:
        return self.track_median_.get(track, self.overall_)

    def sample(self, rng: np.random.Generator, track: str, neutralised: bool, size: int = 1) -> np.ndarray:
        if neutralised:
            return rng.choice(self.neutral_pool_, size=size)
        return self.median(track) + rng.choice(self.green_noise_, size=size)


def lap_multipliers(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, np.ndarray]:
    """Lap time as a multiple of the driver's own median clean lap, for the laps the
    lap-time model does not cover: safety car, VSC and lap 1 (no pit laps)."""
    pace = clean.groupby(KEY + ["driver"])["lap_time_s"].median().rename("pace")
    d = raw.join(pace, on=KEY + ["driver"])
    d = d[d["pace"].notna() & d["lap_time_s"].notna() & ~d["is_pit_in"] & ~d["is_pit_out"]]
    ratio = d["lap_time_s"] / d["pace"]
    return {
        "sc": ratio[d["track_status"] == "sc"].to_numpy(),
        "vsc": ratio[d["track_status"] == "vsc"].to_numpy(),
        "lap1": ratio[d["lap_number"] == 1].to_numpy(),
    }
