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


def test_lap_multipliers_use_own_pace_skip_pit_laps_and_split_period_from_lap_noise():
    cars = ["A", "B", "C", "D", "E", "F"]
    clean = pd.DataFrame({"season": 2024, "round": 1, "driver": cars, "lap_time_s": 90.0})
    rows = []
    for car in cars:
        for lap in range(1, 11):
            status = "sc" if 4 <= lap <= 8 else "green"
            t = 99.0 if lap == 1 else (135.0 if status == "sc" else 90.0)
            if car == "A" and lap == 6:
                t = 144.0                                  # one slow lap inside the period
            rows.append((2024, 1, "T", car, lap, t, False, False, status))
    rows.append((2024, 1, "T", "A", 11, 500.0, True, False, "sc"))   # pit lap, must be skipped
    m = lap_multipliers(_race(rows), clean)
    assert set(np.round(m["lap1"], 3)) == {1.1}
    assert len(m["sc_period"]) == 5 and np.allclose(m["sc_period"], 1.5 + 0.1 / 30)   # one entry per lap; mean includes the slow lap
    assert abs(m["sc_lap"].max() - (0.1 - 0.1 / 30)) < 1e-9 and abs(m["sc_lap"].mean()) < 1e-9   # deviations centre on the period mean
    assert len(m["vsc_period"]) == 0
