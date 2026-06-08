# Quick Start

Minimal steps to run the HIV hotspot pipeline. For the full picture see
[README.md](../README.md).

## 1. Install

```bash
pip install -r requirements.txt
```

A C compiler is required for PyMC/PyTensor (the run warns at startup if one
is missing).

## 2. Check the data

Confirm `data/` contains:

- `input_data.xlsx` with the `hiv_cases` sheet (columns `test_date`,
  `type`, `longitude`, `latitude`) and, optionally, a `testing_sites` sheet.
  This file is not in the repository — place your local copy here (see
  [../data/README.md](../data/README.md)).
- the H3 geometry layer `h3_hexagons_res4.geojson` (and `res3` if you use it)
- the `Ukraine_Adm*.geojson` boundary layers (used only to label hexagons
  with place names and to draw the fallback map — not an analysis mode)

## 3. Smoke test

```bash
python run_hotspots.py --test
```

Runs the whole pipeline on the built-in default config (~3 min), without the
wizard. It should finish with `TEST MODE COMPLETED!` and write maps + Excel
reports under `output/<timestamp>/`.

## 4. Real run

```bash
python run_hotspots.py config.json
```

This **always** launches the interactive wizard, which asks:
- **analysis type** — standard (single window) or iterative (sliding windows);
- **levels** — any combination of `res3`, `res4`, `adm1` (oblasts);
- **analysis window** — iterative: 3/6/9/12 months; standard: you enter the
  period (≤ 12 months). The **baseline is derived** (1-6 m → 12, 7-9 m → 18,
  10-12 m → 24) and never starts before 2023-01-01;
- (standard) parametrization and model.

The period, baseline and levels come from the wizard, not the config.
Non-interactive / scripted runs use `validation/service_run.py` with a
fully-specified config.

Useful flags:

```bash
python run_hotspots.py config.json --use-loo-ic     # LOO-IC model selection
python run_hotspots.py config.json --use-hurdle     # sparse-data hurdle branch
python run_hotspots.py config.json --log-level DEBUG
```

## 5. Read the results

In `output/<timestamp>/`:

- `bayesian/<level>/Report_*.xlsx` — per-unit classification, reliability and
  the watch-list columns (start here)
- `bayesian/<level>/Map_*_EN.png` / `_ua.png` — anomaly maps
- `bayesian/<level>/Watchlist_Map_*` — burden + rate triage map
- `bayesian/<level>/Diagnostics_*.xlsx` — convergence / quality metrics
- `summary/Dashboard_*.png` — one-page overview

Read two things together for each unit: its **classification**
(Established / Emerging hotspot, Stable high-burden, …) with its
**reliability rating** (HIGH / MODERATE / LOW, or UNRELIABLE when the fit
failed) — these are the rigorous result. The **watch-list** columns
(`on_watchlist`, `watch_reason`, `watch_rank`) are a separate triage ranking
that also surfaces high-burden places the rate axis alone would miss; see the
README *Classification* and *Watch-list* sections for what each label means.
