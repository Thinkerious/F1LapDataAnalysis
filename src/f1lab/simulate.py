"""Monte Carlo race simulator for one driver.

Each run builds a race lap by lap:
  - a safety car / VSC timeline drawn from the hazard model (or an observed one),
  - green laps = base pace + lap-time model + stint-level shift + lap noise,
  - neutralised laps = base pace x a multiplier drawn once per safety car period, plus
    small lap-to-lap noise; lap 1 = base pace x a multiplier drawn from history,
  - pit stops add a loss drawn from history and reset tyre age.

Noise comes from out-of-fold residuals of the lap-time model, split into a shift
shared by a whole stint and noise that changes every lap, because errors inside a
stint are strongly correlated (a driver is simply faster or slower that day).
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import drying_races, prepare, wet_races
from .models import COMPOUNDS, gbm_features, make_gbm
from .race_costs import PitLoss, lap_multipliers, pit_stops
from .safety_car import Hazard, hazard_table, lap_status, neutralisation_events

MAX_AGE = 80
GBM_LEAVES, GBM_TREES = 8, 150  # picked by grouped CV on the 2022-2023 seasons (notebook 05)
REACT_WINDOW = 12      # pit early under a safety car if the planned stop is this close
REACT_MIN_AGE = 6      # ...and the tyres have done at least this many laps


@dataclass(frozen=True)
class Stint:
    compound: str
    laps: int
    start_age: int = 1   # laps already on the set when the stint starts (1 = a new set)


class NoModel:
    """Stand-in lap-time model that predicts no change from base pace. Used as the
    'constant pace' baseline: same costs and noise, no tyre or fuel effect."""

    def predict(self, X):
        return np.zeros(len(X))


@dataclass
class SimResult:
    totals: np.ndarray          # total race time per run, seconds
    neutralisations: np.ndarray  # safety car / VSC starts per run
    pit_laps: np.ndarray        # n x (stints - 1) lap of each stop (reflects early stops)

    def summary(self, level: float = 0.8) -> dict:
        lo, hi = np.quantile(self.totals, [(1 - level) / 2, 1 - (1 - level) / 2])
        return {"mean": float(self.totals.mean()), "median": float(np.median(self.totals)),
                "lo": float(lo), "hi": float(hi), "sd": float(self.totals.std())}


class Simulator:
    def __init__(self, lap_model, tracks, pit_loss: PitLoss, hazard: Hazard, pools: dict):
        self.lap_model, self.tracks = lap_model, tracks
        self.pit_loss, self.hazard, self.pools = pit_loss, hazard, pools

    # ---- pieces -----------------------------------------------------------------
    def _delta_table(self, track, race_laps, track_temp, air_temp) -> np.ndarray:
        """Predicted pace delta for every (compound, tyre age, lap), shape (3, MAX_AGE+1, laps+1).

        Cached per race setup: a strategy search runs thousands of plans on one race."""
        key = (track, race_laps, round(float(track_temp), 1), round(float(air_temp), 1))
        cache = self.__dict__.setdefault("_table_cache", {})
        if key not in cache:
            cache[key] = self._build_delta_table(track, race_laps, track_temp, air_temp)
        return cache[key]

    def _build_delta_table(self, track, race_laps, track_temp, air_temp) -> np.ndarray:
        comp, age, lap = np.meshgrid(range(3), range(1, MAX_AGE + 1), range(1, race_laps + 1), indexing="ij")
        frame = pd.DataFrame(
            {
                "compound": np.array(COMPOUNDS)[comp.ravel()], "tyre_age": age.ravel().astype(float),
                "lap_number": lap.ravel(), "race_laps": race_laps, "laps_remaining": race_laps - lap.ravel(),
                "track_temp": track_temp, "air_temp": air_temp, "track": track,
            }
        )
        pred = self.lap_model.predict(gbm_features(frame, self.tracks))
        table = np.zeros((3, MAX_AGE + 1, race_laps + 1))
        table[comp.ravel(), age.ravel(), lap.ravel()] = pred
        return table

    def _status_timeline(self, rng, track, race_laps, n) -> np.ndarray:
        """n x laps matrix: 0 green, 1 VSC, 2 safety car."""
        p, status = self.pools, np.zeros((n, race_laps), dtype=np.int8)
        remaining, kind = np.zeros(n, dtype=int), np.zeros(n, dtype=np.int8)
        for lap in range(1, race_laps + 1):
            start = (remaining == 0) & (rng.random(n) < self.hazard.rate(lap, track))
            p_sc = p["p_sc_early"] if lap <= 2 else p["p_sc_late"]
            new_kind = np.where(rng.random(n) < p_sc, 2, 1).astype(np.int8)
            length = np.where(new_kind == 2, rng.choice(p["len_sc"], n), rng.choice(p["len_vsc"], n))
            running = remaining > 0
            kind = np.where(start, new_kind, kind)
            remaining = np.where(start, length, remaining)
            status[:, lap - 1] = np.where(running | start, kind, 0)
            remaining = np.where(remaining > 0, remaining - 1, 0)
        return status

    # ---- main -------------------------------------------------------------------
    def simulate(self, track, race_laps, base_pace, strategy, track_temp, air_temp, n=1000, seed=0,
                 react=False, status=None) -> SimResult:
        """Total race time for `strategy` over n runs.

        `status` (length race_laps, 0/1/2) replays an observed safety car timeline; when
        omitted one is drawn per run from the hazard model. `react=True` pulls a planned
        stop forward to a safety car lap when it is close enough to be worth it.
        """
        if sum(s.laps for s in strategy) != race_laps:
            raise ValueError("stint lengths must add up to race_laps")
        rng = np.random.default_rng(seed)
        stints = len(strategy)
        comp_idx = np.array([COMPOUNDS.index(s.compound) for s in strategy])
        planned_end = np.tile(np.cumsum([s.laps for s in strategy]), (n, 1))
        delta = self._delta_table(track, race_laps, track_temp, air_temp)
        if status is None:
            timeline = self._status_timeline(rng, track, race_laps, n)
        else:
            timeline = np.tile(np.asarray(status, dtype=np.int8), (n, 1))
        stint_shift = rng.choice(self.pools["stint_shift"], size=(n, stints))
        prev = np.concatenate([np.zeros((n, 1), dtype=timeline.dtype), timeline[:, :-1]], axis=1)
        period = np.cumsum((timeline > 0) & (timeline != prev), axis=1)   # id of the period each lap is in
        sc_shared = rng.choice(self.pools["sc_period"], size=(n, int(period.max()) + 1))
        vsc_shared = rng.choice(self.pools["vsc_period"], size=(n, int(period.max()) + 1))
        rows = np.arange(n)

        start_ages = np.array([s.start_age for s in strategy])
        total, age, stint = np.zeros(n), np.full(n, start_ages[0]), np.zeros(n, dtype=int)
        pit_laps = np.zeros((n, max(stints - 1, 1)), dtype=int)
        for lap in range(1, race_laps + 1):
            st = timeline[:, lap - 1]
            c = comp_idx[stint]
            if lap == 1:
                t = base_pace * rng.choice(self.pools["lap1"], n)
            else:
                green = base_pace + delta[c, np.minimum(age, MAX_AGE), lap] + stint_shift[rows, stint] \
                    + rng.choice(self.pools["lap_noise"], n)
                pid = period[:, lap - 1]
                sc_mult = sc_shared[rows, pid] + rng.choice(self.pools["sc_lap"], n)
                vsc_mult = vsc_shared[rows, pid] + rng.choice(self.pools["vsc_lap"], n)
                t = np.where(st == 2, base_pace * sc_mult, np.where(st == 1, base_pace * vsc_mult, green))
            has_stop = stint < stints - 1
            pit = has_stop & (planned_end[rows, stint] == lap)
            if react and lap > 1:
                next_end = planned_end[rows, np.minimum(stint, stints - 1)]
                early = (has_stop & ~pit & (st > 0) & (age >= REACT_MIN_AGE)
                         & (next_end - lap <= REACT_WINDOW) & (next_end > lap))
                planned_end[rows[early], stint[early]] = lap
                pit = pit | early
            neutral = st > 0
            loss = np.where(neutral, rng.choice(self.pit_loss.neutral_pool_, n),
                            self.pit_loss.median(track) + rng.choice(self.pit_loss.green_noise_, n))
            t = t + np.where(pit, loss, 0.0)
            if stints > 1:
                pit_laps[rows[pit], stint[pit]] = lap
            total += t
            age = np.where(pit, start_ages[np.minimum(stint + 1, stints - 1)], age + 1)
            stint = stint + pit
        starts = ((timeline[:, 1:] > 0) & (timeline[:, :-1] == 0)).sum(axis=1) + (timeline[:, 0] > 0)
        return SimResult(total, starts, pit_laps)


# ---- strategies -----------------------------------------------------------------------
def candidate_strategies(race_laps: int, max_stops: int = 3, min_stint: int = 8, step: int = 4,
                         compounds=COMPOUNDS) -> list[list[Stint]]:
    """All plans with 1..max_stops stops, at least two different compounds (the dry-race
    rule), stints of at least `min_stint` laps, stop laps on a grid of `step` laps."""
    plans = []
    for stops in range(1, max_stops + 1):
        grid = range(min_stint, race_laps - min_stint + 1, step)
        for cuts in product(grid, repeat=stops):
            if any(b - a < min_stint for a, b in zip(cuts, cuts[1:])):
                continue
            if list(cuts) != sorted(cuts) or race_laps - cuts[-1] < min_stint:
                continue
            lengths = np.diff([0, *cuts, race_laps])
            for seq in product(compounds, repeat=stops + 1):
                if len(set(seq)) >= 2:
                    plans.append([Stint(c, int(n)) for c, n in zip(seq, lengths)])
    return plans


def describe(strategy: list[Stint]) -> str:
    return " > ".join(f"{s.compound[0]}{s.laps}" for s in strategy)


# ---- building a simulator from data -------------------------------------------------------
def build_simulator(raw: pd.DataFrame, clean: pd.DataFrame, seasons, seed: int = 0) -> Simulator:
    """Fit every part from the given seasons only."""
    skip = drying_races(clean) | wet_races(raw)
    in_seasons = raw["season"].isin(seasons)
    raw_s = raw[in_seasons]
    clean_s = clean[clean["season"].isin(seasons)]

    train = prepare(raw, clean)
    train = train[train["season"].isin(seasons)].reset_index(drop=True)
    tracks = sorted(train["track"].unique())
    X, y = gbm_features(train, tracks), train["delta_s"]
    groups = (train["season"].astype(str) + "-" + train["round"].astype(str)).to_numpy()

    oof = np.zeros(len(y))
    for fit_idx, test_idx in GroupKFold(5).split(X, y, groups):
        oof[test_idx] = make_gbm(GBM_LEAVES, GBM_TREES, seed).fit(X.iloc[fit_idx], y.iloc[fit_idx]).predict(X.iloc[test_idx])
    resid = (y.to_numpy() - oof)
    stint_key = train[["season", "round", "driver", "stint"]].astype(str).agg("-".join, axis=1)
    frame = pd.DataFrame({"stint": stint_key, "r": resid})
    stint_mean = frame.groupby("stint")["r"].transform("mean")
    stint_size = frame.groupby("stint")["r"].transform("size")
    big = stint_size >= 8
    pools = {
        "stint_shift": frame.loc[big, "r"].groupby(frame.loc[big, "stint"]).mean().to_numpy(),
        "lap_noise": (frame.loc[big, "r"] - stint_mean[big]).to_numpy(),
    }
    # Safety car lap pace and period lengths come from OpenF1 seasons only. FastF1 (2022)
    # marks a lap as safety car if any part of it was, which dilutes the multiplier (2022
    # averages 1.33x against 1.45-1.47x in every OpenF1 season). Hazard starts are counts,
    # so they use every season.
    dry = raw_s[[(s, r) not in skip for s, r in zip(raw_s["season"], raw_s["round"])]]
    openf1 = dry["source"] == "openf1"
    pools.update(lap_multipliers(dry[openf1], clean_s[clean_s["source"] == "openf1"]))

    lap = lap_status(raw_s, skip)
    table = hazard_table(lap)
    events = neutralisation_events(lap_status(raw_s[raw_s["source"] == "openf1"], skip))
    pools["len_sc"] = events.loc[events["kind"] == "sc", "length"].to_numpy()
    pools["len_vsc"] = events.loc[events["kind"] == "vsc", "length"].to_numpy()
    early = events["start"] <= 2
    pools["p_sc_early"] = float((events.loc[early, "kind"] == "sc").mean())
    pools["p_sc_late"] = float((events.loc[~early, "kind"] == "sc").mean())

    stops = pit_stops(raw_s[[(s, r) not in skip for s, r in zip(raw_s["season"], raw_s["round"])]])
    return Simulator(
        make_gbm(GBM_LEAVES, GBM_TREES, seed).fit(X, y), tracks, PitLoss().fit(stops), Hazard().fit(table), pools
    )


def rank_strategies(sim: Simulator, track: str, race_laps: int, base_pace: float, track_temp: float,
                    air_temp: float, plans: list | None = None, n: int = 200, seed: int = 0,
                    react: bool = False, **search) -> pd.DataFrame:
    """Simulate every plan against the same random draws and sort by mean race time.

    Using the same draws for every plan makes the differences between plans far less
    noisy than the totals themselves (a safety car helps or hurts every plan together).
    `p_beats_best` is the share of runs in which the plan finished ahead of the best one.
    """
    plans = plans if plans is not None else candidate_strategies(race_laps, **search)
    totals = np.array([
        sim.simulate(track, race_laps, base_pace, plan, track_temp, air_temp, n=n, seed=seed, react=react).totals
        for plan in plans
    ])
    means = totals.mean(axis=1)
    best = means.argmin()
    out = pd.DataFrame({
        "plan": [describe(p) for p in plans], "stops": [len(p) - 1 for p in plans],
        "mean_s": means, "sd_s": totals.std(axis=1),
        "behind_best_s": means - means[best], "p_beats_best": (totals < totals[best]).mean(axis=1),
    })
    return out.sort_values("mean_s").reset_index(drop=True)
