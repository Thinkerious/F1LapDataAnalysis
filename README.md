# F1 Race Strategy Lab

Notebook-driven ML and simulation on Formula 1 telemetry: tire-degradation
lap-time model, safety-car hazard model, and a Monte Carlo race simulator that
evaluates pit strategies, backtested on held-out seasons.

The goal is calibrated uncertainty: not just a predicted lap time, but an
interval that is checked against held-out races to confirm it covers the truth
as often as it claims.

## Status

Work in progress. No results yet; this README is filled in as each step lands.

## Setup

On macOS, LightGBM needs OpenMP: `brew install libomp`.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m ipykernel install --user --name f1lab

## Layout

- `notebooks/`: numbered notebooks, run top to bottom
- `src/f1lab/`: reusable code the notebooks import
- `data/`: local cache of raw pulls (git-ignored)
- `reports/`: figures and result tables exported by the notebooks
