"""Cleaning rules that turn the raw lap table into modelling data.

    python -m f1lab.clean

Each rule removes laps that are slow or unusual for reasons unrelated to tyre
and fuel pace. Rules run in order and `clean_laps` reports how many laps each
one removed, so the effect of every decision is visible.
"""

from pathlib import Path

import pandas as pd

SLICKS = ["SOFT", "MEDIUM", "HARD"]

# Applied after every other rule: a lap this far over its race's median pace is
# not real slick racing pace. Mostly neutralised or wet laps the labels missed (60 of the
# 169 such laps were in 2025 Spa, where the rain flag lagged the conditions).
EXTREME_SLOW = 1.3


def _extreme_slow(d: pd.DataFrame) -> pd.Series:
    median = d.groupby(["season", "round"])["lap_time_s"].transform("median")
    return d["lap_time_s"] > EXTREME_SLOW * median


# (name, mask of rows to DROP). Order matters only for the reported counts.
RULES = [
    ("pit-in or pit-out lap", lambda d: d["is_pit_in"] | d["is_pit_out"]),
    ("lap 1", lambda d: d["lap_number"] == 1),
    ("no lap time", lambda d: d["lap_time_s"].isna()),
    ("safety car, vsc or red flag", lambda d: d["track_status"] != "green"),
    ("rain", lambda d: d["rainfall"]),
    ("not a slick compound", lambda d: ~d["compound"].isin(SLICKS)),
    ("no tyre age", lambda d: d["tyre_age"].isna()),
    ("extreme slow lap", _extreme_slow),  # last: needs the other rules' laps for its median
]


def clean_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean laps, report of laps removed by each rule in order)."""
    kept, report = df, [{"rule": "start", "removed": 0, "remaining": len(df)}]
    for name, drop in RULES:
        mask = drop(kept)
        kept = kept[~mask]
        report.append({"rule": name, "removed": int(mask.sum()), "remaining": len(kept)})
    return kept.reset_index(drop=True), pd.DataFrame(report)


def main() -> None:
    raw = pd.read_parquet("data/laps.parquet")
    clean, report = clean_laps(raw)
    print(report.to_string(index=False))
    out = Path("data/laps_clean.parquet")
    clean.to_parquet(out, index=False)
    print(f"\nwrote {len(clean)} of {len(raw)} laps ({len(clean) / len(raw):.1%}) to {out}")


if __name__ == "__main__":
    main()
