import numpy as np
import pandas as pd

from f1lab.safety_car import Hazard, hazard_table, lap_status, log_loss, neutralisation_events


def _lap_table(ranks, track="T", season=2024, rnd=1):
    return pd.DataFrame(
        {
            "season": season, "round": rnd, "lap_number": range(1, len(ranks) + 1),
            "rank": ranks, "cars": 20, "track": track,
        }
    )


def test_events_have_start_length_and_kind_and_vsc_upgrades_to_sc():
    #            laps: 1  2  3  4  5  6  7  8  9 10
    lap = _lap_table([0, 0, 1, 1, 0, 0, 1, 2, 2, 0])
    ev = neutralisation_events(lap)
    assert ev[["start", "length", "kind"]].values.tolist() == [[3, 2, "vsc"], [7, 3, "sc"]]


def test_hazard_table_skips_laps_under_neutralisation_and_marks_event_laps():
    lap = _lap_table([0, 0, 1, 1, 0, 0])
    h = hazard_table(lap)
    # lap 3 starts the event (at risk, event=1). Lap 4 is under neutralisation, not at risk.
    # Lap 5 follows a neutralised lap, so it is also not at risk. Lap 6 is.
    assert h["lap"].tolist() == [1, 2, 3, 6]
    assert h["event"].tolist() == [0, 0, 1, 0]
    assert h["early"].tolist() == [True, True, False, False]


def test_lap_status_drops_excluded_races_and_laps_with_few_cars():
    raw = pd.DataFrame(
        {
            "season": 2024, "round": [1, 1, 2], "lap_number": [1, 2, 1], "track": "T",
            "track_status": ["green", "sc", "green"],
        }
    )
    raw = pd.concat([raw] * 6, ignore_index=True)  # 6 cars per row so MIN_CARS is met
    out = lap_status(raw, skip={(2024, 2)})
    assert set(out["round"]) == {1}
    assert out.sort_values("lap_number")["rank"].tolist() == [0, 2]


def test_hazard_separates_early_laps_and_pulls_track_effect_toward_one():
    rng = np.random.default_rng(0)
    rows = []
    for track, late_rate in (("A", 0.01), ("B", 0.05)):
        for rnd in range(30):
            for lap in range(1, 51):
                rate = 0.15 if lap <= 2 else late_rate
                rows.append((2024, rnd, track, lap, 50, int(rng.random() < rate)))
    table = pd.DataFrame(rows, columns=["season", "round", "track", "lap", "race_laps", "event"])
    table["early"] = table["lap"] <= 2

    tight = Hazard(k=1000).fit(table)  # huge k: track barely matters
    loose = Hazard(k=0.1).fit(table)
    assert tight.rate_early_ > 5 * tight.rate_late_
    assert abs(tight.track_multiplier_["A"] - 1) < abs(loose.track_multiplier_["A"] - 1)
    assert loose.rate(10, "B") > loose.rate(10, "A")
    assert loose.rate(1, "A") == loose.rate_early_ == loose.rate(2, "B")


def test_log_loss_rewards_confident_correct_predictions():
    assert log_loss([1, 0], [0.9, 0.1]) < log_loss([1, 0], [0.5, 0.5])


def test_hazard_without_track_effect_ignores_the_track():
    table = pd.DataFrame(
        {
            "season": 2024, "round": 1, "track": ["A"] * 6 + ["B"] * 6, "lap": [1, 2, 3, 4, 5, 6] * 2,
            "race_laps": 6, "event": [0, 0, 1, 0, 0, 0] + [0, 0, 0, 0, 0, 0],
        }
    )
    table["early"] = table["lap"] <= 2
    h = Hazard().fit(table)
    assert h.rate(5, "A") == h.rate(5, "B") == h.rate_late_
