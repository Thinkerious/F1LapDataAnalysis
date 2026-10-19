import numpy as np
import pytest

from f1lab.race_costs import PitLoss
from f1lab.safety_car import Hazard
from f1lab.simulate import Simulator, Stint, candidate_strategies, describe


class ZeroModel:
    """Lap-time model that predicts no change from base pace."""

    def predict(self, X):
        return np.zeros(len(X))


def make_sim(rate_late=0.0, rate_early=0.0):
    pit = PitLoss()
    pit.overall_, pit.track_median_ = 22.0, {"T": 22.0}
    pit.green_noise_, pit.neutral_pool_ = np.array([0.0]), np.array([10.0])
    hz = Hazard()
    hz.rate_early_, hz.rate_late_, hz.track_multiplier_ = rate_early, rate_late, {}
    pools = {
        "stint_shift": np.array([0.0]), "lap_noise": np.array([0.0]),
        "sc_period": np.array([1.5]), "sc_lap": np.array([0.0]),
        "vsc_period": np.array([1.2]), "vsc_lap": np.array([0.0]), "lap1": np.array([1.1]), "len_sc": np.array([3]),
        "len_vsc": np.array([2]), "p_sc_early": 1.0, "p_sc_late": 1.0,
    }
    return Simulator(ZeroModel(), ["T"], pit, hz, pools)


ONE_STOP = [Stint("SOFT", 20), Stint("HARD", 30)]


def run(sim, **kw):
    return sim.simulate("T", 50, 100.0, ONE_STOP, 30.0, 20.0, n=4, seed=0, **kw)


def test_without_noise_or_safety_cars_total_is_pace_plus_lap1_plus_one_pit_stop():
    r = run(make_sim())
    # lap 1 at 1.1x, 49 laps at base pace, one stop costing 22 s
    assert np.allclose(r.totals, 100 * 1.1 + 49 * 100 + 22)
    assert (r.pit_laps[:, 0] == 20).all() and (r.neutralisations == 0).all()


def test_replayed_safety_car_laps_cost_the_multiplier():
    status = np.zeros(50, dtype=int)
    status[9:12] = 2  # laps 10, 11, 12
    r = run(make_sim(), status=status)
    assert np.allclose(r.totals, 100 * 1.1 + 49 * 100 + 22 + 3 * 0.5 * 100)
    assert (r.neutralisations == 1).all()


def test_stop_under_safety_car_uses_the_cheaper_neutralised_loss():
    status = np.zeros(50, dtype=int)
    status[19] = 2  # the planned stop lap is itself a safety car lap
    r = run(make_sim(), status=status)
    assert np.allclose(r.totals, 100 * 1.1 + 48 * 100 + 150 + 10)


def test_react_pulls_the_stop_forward_to_a_nearby_safety_car():
    status = np.zeros(50, dtype=int)
    status[14] = 2  # lap 15: five laps before the planned stop on lap 20
    plain = run(make_sim(), status=status)
    react = run(make_sim(), status=status, react=True)
    assert (plain.pit_laps[:, 0] == 20).all() and (react.pit_laps[:, 0] == 15).all()
    assert np.allclose(plain.totals - react.totals, 22.0 - 10.0)


def test_react_ignores_a_safety_car_that_is_too_far_from_the_stop():
    status = np.zeros(50, dtype=int)
    status[2] = 2  # lap 3: stop is 17 laps away and tyres are fresh
    react = run(make_sim(), status=status, react=True)
    assert (react.pit_laps[:, 0] == 20).all()


def test_hazard_draws_safety_cars_only_when_the_rate_is_positive():
    assert run(make_sim(rate_late=0.0)).neutralisations.sum() == 0
    many = make_sim(rate_late=0.05).simulate("T", 50, 100.0, ONE_STOP, 30.0, 20.0, n=400, seed=1)
    assert 0.5 < many.neutralisations.mean() < 3.0


def test_stint_lengths_must_cover_the_race():
    with pytest.raises(ValueError):
        make_sim().simulate("T", 50, 100.0, [Stint("SOFT", 20), Stint("HARD", 20)], 30.0, 20.0, n=2)


def test_candidate_strategies_obey_the_dry_race_rules():
    plans = candidate_strategies(57, max_stops=2, min_stint=10, step=5)
    assert plans
    for plan in plans:
        assert sum(s.laps for s in plan) == 57
        assert len({s.compound for s in plan}) >= 2
        assert min(s.laps for s in plan) >= 10
        assert 2 <= len(plan) <= 3
    assert describe([Stint("SOFT", 16), Stint("HARD", 41)]) == "S16 > H41"


def test_used_sets_start_older_and_stint_age_resets_to_the_next_sets_age():
    class AgeCounter:
        """Charges 0.1 s per lap of tyre age, so totals reveal the ages that were used."""

        def predict(self, X):
            return 0.1 * X["tyre_age"].to_numpy()

    sim = make_sim()
    sim.lap_model = AgeCounter()
    new = [Stint("SOFT", 20), Stint("HARD", 30)]
    used = [Stint("SOFT", 20, start_age=5), Stint("HARD", 30, start_age=8)]
    a = sim.simulate("T", 50, 100.0, new, 30.0, 20.0, n=2, seed=0)
    b = sim.simulate("T", 50, 100.0, used, 30.0, 20.0, n=2, seed=0)
    # lap 1 is not modelled. Stint one: ages 5..24 vs 1..20 -> laps 2-20 are 4 older each (19 laps).
    # Stint two: 8.. vs 1.. -> 30 laps, each 7 older.
    assert np.allclose(b.totals - a.totals, 0.1 * (4 * 19 + 7 * 30))


def test_each_safety_car_period_gets_one_shared_multiplier_across_its_laps():
    sim = make_sim()
    sim.pools["sc_period"] = np.array([1.3, 1.7])   # two possible multipliers per period
    status = np.zeros(50, dtype=int)
    status[9:14] = 2      # one 5-lap period
    r = sim.simulate("T", 50, 100.0, ONE_STOP, 30.0, 20.0, n=500, seed=3, status=status)
    base = 100 * 1.1 + 49 * 100 + 22
    extra = (r.totals - base) / 100 / 5          # (multiplier - 1) summed over 5 laps, per lap
    # shared draw: every run is all-1.3 or all-1.7, never a mix of the two
    assert set(np.round(extra, 6)) == {0.3, 0.7}


def test_rank_strategies_uses_common_draws_and_orders_by_mean():
    from f1lab.simulate import rank_strategies

    class FavourHard:
        """HARD laps are 0.5 s faster than SOFT, so a longer HARD stint should win."""

        def predict(self, X):
            return np.where(X["compound"].astype(str) == "HARD", -0.5, 0.0)

    sim = make_sim()
    sim.pools["lap_noise"] = np.array([-1.0, 0.0, 1.0])   # runs differ, but plans share each run's draws
    sim.lap_model = FavourHard()
    plans = [[Stint("SOFT", 40), Stint("HARD", 10)], [Stint("SOFT", 10), Stint("HARD", 40)]]
    out = rank_strategies(sim, "T", 50, 100.0, 30.0, 20.0, plans=plans, n=300, seed=2)
    assert out["plan"].tolist() == ["S10 > H40", "S40 > H10"]
    assert out["behind_best_s"].iloc[0] == 0 and abs(out["behind_best_s"].iloc[1] - 15.0) < 1e-6
    assert out["p_beats_best"].iloc[1] == 0.0   # same draws everywhere, so the gap never flips
    assert out["sd_s"].iloc[0] > 1.0           # while the totals themselves do vary run to run


def test_cli_ranks_plans_and_prints_a_table(capsys):
    from f1lab import cli

    out = cli.main(["--track", "T", "--laps", "40", "--pace", "100", "--max-stops", "1",
                    "--min-stint", "10", "--step", "10", "--runs", "20", "--top", "3"], simulator=make_sim())
    text = capsys.readouterr().out
    assert "plans" in text and "behind" in text and "beats_best" in text
    assert out["mean_s"].is_monotonic_increasing and len(out) > 3
    assert cli.clock(5415.3) == "1:30:15.3"


def test_cli_warns_when_the_track_has_no_training_data(capsys):
    from f1lab import cli

    cli.main(["--track", "Nowhere", "--laps", "30", "--pace", "90", "--max-stops", "1", "--min-stint", "10",
              "--step", "10", "--runs", "10"], simulator=make_sim())
    assert "no training data" in capsys.readouterr().out
