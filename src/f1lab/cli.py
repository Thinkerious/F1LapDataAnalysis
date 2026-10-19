"""Rank pit strategies for a race from the command line.

    python -m f1lab.cli --track Sakhir --laps 57 --pace 95.5 --track-temp 30 --air-temp 25

`--pace` is the driver's typical clean lap time in seconds (from practice, say). Needs
data/laps.parquet and data/laps_clean.parquet (see the README for how to build them).
"""

import argparse
from pathlib import Path

import pandas as pd

from .simulate import Simulator, build_simulator, rank_strategies


def clock(seconds: float) -> str:
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}"


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m f1lab.cli", description=__doc__.split("\n\n")[0])
    p.add_argument("--track", required=True, help="circuit name as used in the data, e.g. Sakhir")
    p.add_argument("--laps", type=int, required=True, help="race distance in laps")
    p.add_argument("--pace", type=float, required=True, help="driver's typical clean lap time, seconds")
    p.add_argument("--track-temp", type=float, default=30.0)
    p.add_argument("--air-temp", type=float, default=25.0)
    p.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024], help="seasons to learn from")
    p.add_argument("--max-stops", type=int, default=3)
    p.add_argument("--min-stint", type=int, default=10)
    p.add_argument("--step", type=int, default=5, help="pit lap grid spacing, in laps")
    p.add_argument("--runs", type=int, default=200, help="simulated races per plan")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--no-react", action="store_true", help="do not pit early under a safety car")
    p.add_argument("--data", type=Path, default=Path("data"))
    return p.parse_args(argv)


def main(argv=None, simulator: Simulator | None = None) -> pd.DataFrame:
    args = parse(argv)
    if simulator is None:
        raw = pd.read_parquet(args.data / "laps.parquet")
        clean = pd.read_parquet(args.data / "laps_clean.parquet")
        simulator = build_simulator(raw, clean, seasons=tuple(args.seasons))
    if args.track not in simulator.tracks:
        print(f"note: {args.track!r} has no training data, so track effects are ignored.")
    ranked = rank_strategies(
        simulator, args.track, args.laps, args.pace, args.track_temp, args.air_temp,
        n=args.runs, react=not args.no_react, max_stops=args.max_stops, min_stint=args.min_stint, step=args.step,
    )
    shown = ranked.head(args.top).assign(
        mean=lambda d: d["mean_s"].map(clock), behind=lambda d: d["behind_best_s"].map(lambda v: f"+{v:.1f}s"),
        sd=lambda d: d["sd_s"].map(lambda v: f"{v:.0f}s"), beats_best=lambda d: d["p_beats_best"].map(lambda v: f"{v:.0%}"),
    )
    print(f"{len(ranked)} plans, {args.runs} simulated races each. Best first:\n")
    print(shown[["plan", "stops", "mean", "behind", "sd", "beats_best"]].to_string(index=False))
    print("\nPlans within about a second of each other are effectively tied. A single race varies by "
          "minutes from run to run (sd), so read the order as a guide, not a prediction.")
    return ranked


if __name__ == "__main__":
    main()
