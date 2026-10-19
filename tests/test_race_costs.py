import numpy as np
import pandas as pd

from f1lab.race_costs import PitLoss, lap_multipliers, pit_stops


def _race(rows):
    cols = ["season", "round", "track", "driver", "lap_number", "lap_time_s", "is_pit_in", "is_pit_out", "track_status"]
    return pd.DataFrame(rows, columns=cols)


def test_pit_stop_loss_is_in_plus_out_lap_minus_the_field_on_those_laps():
    rows = []
    for driver in ("AAA", "BBB", "CCC"):  # three cars on 90 s laps, five laps
        for lap in range(1, 6):
            rows.append((2024, 1, "T", driver, lap, 90.0, False, False, "green"))
    rows = [r for r in rows if r[3] != "AAA"]
    # AAA pits at the end of lap 3: in-lap 100 s, out-lap 98 s -> 198 s against 180 s, a loss of 18 s
    rows += [(2024, 1, "T", "AAA", lap, t, lap == 3, lap == 4, "green")
             for lap, t in zip(range(1, 6), (90, 90, 100, 98, 90))]
    stops = pit_stops(_race(rows))
    assert len(stops) == 1
    assert stops["loss"].iloc[0] == 18.0 and stops["status"].iloc[0] == "green"


def test_pit_stop_status_is_the_worst_of_the_two_laps():
    rows = []
    for driver in ("BBB", "CCC", "DDD"):
        for lap in range(1, 6):
            rows.append((2024, 1, "T", driver, lap, 90.0, False, False, "green"))
    rows += [(2024, 1, "T", "AAA", lap, 90.0 + (12 if lap in (3, 4) else 0), lap == 3, lap == 4,
              "green" if lap != 4 else "sc") for lap in range(1, 6)]
    stops = pit_stops(_race(rows))
    assert stops["status"].tolist() == ["sc"]


def test_pit_loss_shrinks_a_sparse_track_toward_the_overall_median():
    stops = pd.DataFrame(
        {
            "track": ["A"] * 200 + ["B"] * 2,
            "status": "green",
            "loss": [20.0] * 200 + [30.0] * 2,
        }
    )
    model = PitLoss(k=10).fit(stops)
    assert model.overall_ == 20.0
    assert 20.0 < model.median("B") < 22.0   # two stops should barely move it
    assert model.median("never raced here") == 20.0


def test_pit_loss_samples_neutralised_stops_from_their_own_pool():
    stops = pd.DataFrame(
        {"track": ["A"] * 50 + ["A"] * 4, "status": ["green"] * 50 + ["sc"] * 4,
         "loss": [22.0] * 50 + [5.0, 6.0, 7.0, 8.0]}
    )
    model = PitLoss().fit(stops)
    rng = np.random.default_rng(0)
    assert model.sample(rng, "A", neutralised=True, size=200).max() <= 8.0
    assert abs(model.sample(rng, "A", neutralised=False, size=200).mean() - 22.0) < 1e-6


def test_lap_multipliers_use_the_drivers_own_clean_pace_and_skip_pit_laps():
    clean = pd.DataFrame({"season": 2024, "round": 1, "driver": "AAA", "lap_time_s": [90.0, 90.0, 90.0]})
    raw = _race(
        [
            (2024, 1, "T", "AAA", 1, 99.0, False, False, "green"),
            (2024, 1, "T", "AAA", 2, 135.0, False, False, "sc"),
            (2024, 1, "T", "AAA", 3, 108.0, False, False, "vsc"),
            (2024, 1, "T", "AAA", 4, 200.0, True, False, "sc"),   # pit lap, must be skipped
        ]
    )
    m = lap_multipliers(raw, clean)
    assert m["lap1"].tolist() == [1.1] and m["sc"].tolist() == [1.5] and m["vsc"].tolist() == [1.2]
