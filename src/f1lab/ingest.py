"""Pull every race for the chosen seasons into one lap table.

    python -m f1lab.ingest

Season 2022 comes from FastF1, 2023 onward from OpenF1 (see the module
docstrings for why). Races that fail are reported, not fatal, so one bad
session does not lose the rest of the pull.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from . import fastf1_source, openf1

OUT = Path("data/laps.parquet")
FAILED = Path("data/failed_races.json")
SEASONS = {2022: "fastf1", 2023: "openf1", 2024: "openf1", 2025: "openf1"}


def rounds_for(year: int, source: str) -> list[int]:
    if source == "openf1":
        return openf1.list_races(year)["round"].tolist()
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["RoundNumber"].astype(int).tolist()


def main() -> None:
    frames, failed = [], []
    for year, source in SEASONS.items():
        loader = fastf1_source.load_race_laps if source == "fastf1" else openf1.load_race_laps
        for rnd in rounds_for(year, source):
            try:
                df = loader(year, rnd)
                if df.empty:
                    raise ValueError("no laps returned")
                frames.append(df)
                print(f"ok   {year} r{rnd:02d} {source:6s} {len(df)} laps", flush=True)
            except Exception as e:  # noqa: BLE001 - record and continue
                failed.append({"year": year, "round": rnd, "source": source, "error": repr(e)[:200]})
                print(f"FAIL {year} r{rnd:02d} {source:6s} {type(e).__name__}", flush=True)

    laps = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    laps.to_parquet(OUT, index=False)
    FAILED.write_text(json.dumps(failed, indent=2))
    print(f"\nwrote {len(laps)} laps from {laps.groupby(['season', 'round']).ngroups} races to {OUT}")
    print(f"{len(failed)} races failed (see {FAILED})")
    sys.exit(0)


if __name__ == "__main__":
    main()
