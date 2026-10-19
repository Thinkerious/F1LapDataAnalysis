"""The full evaluation, written once so it can be run on either split.

    stage "dev":   train 2022-2023, test 2024  (used to catch bugs and make decisions)
    stage "final": train 2022-2024, test 2025  (run once, after every choice was frozen)

Nothing here picks settings from the test season. Model sizes and noise definitions are
the ones fixed during development.
"""

import numpy as np
import pandas as pd

from .backtest import constant_pace_version, driver_races, rank_agreement, replay, score
from .conformal import CQR, coverage, mean_width
from .features import drying_races, mae, prepare, rmse, split, wet_races
from .models import gbm_features, linear_features, make_gbm, make_linear
from .safety_car import Hazard, hazard_table, lap_status, log_loss, neutralisation_events
from .simulate import GBM_LEAVES, GBM_TREES, build_simulator

SEASONS = {"dev": ((2022, 2023), 2024), "final": ((2022, 2023, 2024), 2025)}


def lap_time_results(raw: pd.DataFrame, clean: pd.DataFrame, stage: str, seeds: int = 5) -> dict:
    train, test = split(prepare(raw, clean), stage)
    tracks = sorted(train["track"].unique())
    y = test["delta_s"]
    preds = {"predict no change": pd.Series(0.0, index=test.index)}

    by_compound = train.groupby("compound")["delta_s"].mean()
    preds["mean by compound"] = test["compound"].map(by_compound)
    bins = [0, 5, 10, 15, 20, 30, 100]
    by_age = train.assign(b=pd.cut(train["tyre_age"], bins)).groupby(["compound", "b"], observed=True)["delta_s"].mean()
    tb = test.assign(b=pd.cut(test["tyre_age"], bins))
    age_pred = pd.Series([by_age.get((c, b), np.nan) for c, b in zip(tb["compound"], tb["b"])], index=test.index)
    preds["mean by compound and tyre age"] = age_pred.fillna(test["compound"].map(by_compound))

    lin = make_linear().fit(linear_features(train), train["delta_s"])
    preds["linear (robust)"] = pd.Series(lin.predict(linear_features(test)), index=test.index)

    X, Xt = gbm_features(train, tracks), gbm_features(test, tracks)
    gbm_runs = [pd.Series(make_gbm(GBM_LEAVES, GBM_TREES, s).fit(X, train["delta_s"]).predict(Xt), index=test.index)
                for s in range(seeds)]
    preds["gradient boosting"] = sum(gbm_runs) / seeds   # average of the seeds
    metrics = pd.DataFrame({name: {"MAE (s)": mae(y, p), "RMSE (s)": rmse(y, p)} for name, p in preds.items()}).T
    metrics.loc["gradient boosting", "MAE (s)"] = float(np.mean([mae(y, p) for p in gbm_runs]))   # per-seed mean, as in notebook 05

    seen = test["track"].isin(train["track"].unique())
    by_seen = pd.DataFrame({
        name: {"seen tracks": mae(y[seen], p[seen]), "unseen tracks": mae(y[~seen], p[~seen]) if (~seen).any() else np.nan}
        for name, p in preds.items() if name in ("linear (robust)", "gradient boosting")
    }).T

    groups = (train["season"].astype(str) + "-" + train["round"].astype(str)).to_numpy()
    intervals = {}
    for level in (0.5, 0.8, 0.9):
        lo, hi = CQR(1 - level).fit(X, train["delta_s"], groups).predict(Xt)
        hit = ((y.to_numpy() >= lo) & (y.to_numpy() <= hi))
        intervals[level] = {"coverage": float(hit.mean()), "mean width (s)": mean_width(lo, hi),
                            "coverage, seen tracks": float(hit[seen.to_numpy()].mean()),
                            "coverage, unseen tracks": float(hit[~seen.to_numpy()].mean()) if (~seen).any() else np.nan}
    return {"metrics": metrics, "by_seen": by_seen, "intervals": pd.DataFrame(intervals).T,
            "n_laps": len(test), "n_races": test.groupby(["season", "round"]).ngroups,
            "unseen_tracks": sorted(test.loc[~seen, "track"].unique())}


def hazard_results(raw: pd.DataFrame, clean: pd.DataFrame, stage: str) -> dict:
    train_seasons, test_season = SEASONS[stage]
    skip = drying_races(clean) | wet_races(raw)
    lap = lap_status(raw, skip)
    train_t = hazard_table(lap[lap["season"].isin(train_seasons)])
    test_lap = lap[lap["season"] == test_season]
    test_t = hazard_table(test_lap)
    hz = Hazard().fit(train_t)
    constant = np.full(len(test_t), train_t["event"].mean())
    out = {"log loss, one rate for every lap": log_loss(test_t["event"], constant),
           "log loss, laps 1-2 split": log_loss(test_t["event"], hz.predict(test_t))}
    races = test_lap.groupby(["season", "round"]).agg(race_laps=("lap_number", "max")).reset_index()
    events = neutralisation_events(test_lap)
    actual = events.groupby(["season", "round"]).size().rename("actual").reset_index()
    races = races.merge(actual, how="left", on=["season", "round"]).fillna({"actual": 0})
    races["expected"] = [hz.rate_early_ * 2 + hz.rate_late_ * max(n - 2, 0) for n in races["race_laps"]]
    out["expected events per race"] = float(races["expected"].mean())
    out["actual events per race"] = float(races["actual"].mean())
    out["races"] = len(races)
    out["events"] = int(races["actual"].sum())
    return {"summary": pd.Series(out), "races": races, "rates": (hz.rate_early_, hz.rate_late_)}


def simulator_results(raw: pd.DataFrame, clean: pd.DataFrame, stage: str, n: int = 400) -> dict:
    train_seasons, test_season = SEASONS[stage]
    sim = build_simulator(raw, clean, seasons=train_seasons)
    plans = driver_races(raw, clean, seasons=(test_season,))
    full = replay(sim, plans, n=n)
    flat = replay(constant_pace_version(sim), plans, n=n)
    naive = pd.Series([p.actual_total - p.base_pace * p.race_laps for p in plans])
    neutral = {(p.season, p.round): int((p.status > 0).sum()) for p in plans}
    full["neutral laps"] = [neutral[(s, r)] for s, r in zip(full["season"], full["round"])]
    levels = {f"{lvl:.0%}": replay(sim, plans, n=n, level=lvl)["inside"].mean() for lvl in (0.5, 0.8, 0.9)}
    table = pd.DataFrame({"simulator": score(full), "constant pace": score(flat)})
    table.loc["rank agreement within races"] = [rank_agreement(full), rank_agreement(flat)]
    return {"table": table, "replay": full, "coverage": pd.Series(levels), "naive": naive,
            "driver_races": len(plans), "races": len({(p.season, p.round) for p in plans})}
