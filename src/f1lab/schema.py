"""Common lap table both data sources are normalized into.

One row per driver per lap of a race. Times are seconds, temperatures are
degrees C. `tyre_age` counts laps on the current set including the current lap.
"""

import pandas as pd

COLUMNS = [
    "season",
    "round",
    "track",
    "driver",
    "lap_number",
    "lap_time_s",
    "compound",
    "tyre_age",
    "stint",
    "is_pit_in",
    "is_pit_out",
    "track_temp",
    "air_temp",
    "rainfall",
    "track_status",
    "source",
]

# Worst status seen during a lap. Yellow flags are local, so they stay "green".
TRACK_STATUSES = ["green", "vsc", "sc", "red"]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns and enforce dtypes so both sources produce identical tables."""
    out = df[COLUMNS].copy()
    out["season"] = out["season"].astype(int)
    out["round"] = out["round"].astype(int)
    out["lap_number"] = out["lap_number"].astype(int)
    out["tyre_age"] = out["tyre_age"].astype("Int64")  # nullable: FastF1 gives floats
    out["stint"] = out["stint"].astype("Int64")
    out["is_pit_in"] = out["is_pit_in"].astype(bool)
    out["is_pit_out"] = out["is_pit_out"].astype(bool)
    out["rainfall"] = out["rainfall"].astype(bool)
    return out.sort_values(["driver", "lap_number"]).reset_index(drop=True)
