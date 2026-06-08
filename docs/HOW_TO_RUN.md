# How to Run

The pipeline is driven by a single entry point, `run_hotspots.py`. All
logic lives in the `pipeline/` package.

## Commands

```bash
# Smoke test on the built-in default config (no JSON required)
python run_hotspots.py --test

# Full run on a config file
python run_hotspots.py config.json
```

With a config file the **interactive wizard always runs** (there is no
"use defaults?" shortcut and no config-driven auto-start). It asks: analysis
type (standard / iterative) → levels (any combination of `res3`, `res4`,
`adm1`) → analysis window → (standard) parametrization & model. `--test`
skips the wizard entirely. Scripted runs use `validation/service_run.py`.

The **analysis window** is 3/6/9/12 months in iterative mode, or a period of
≤ 12 months in standard mode; the **baseline length is derived** from it
(1-6 m → 12, 7-9 m → 18, 10-12 m → 24) and never starts before **2023-01-01**.
A period/range whose baseline would cross that floor is rejected (asking for a
later start); iterative additionally requires ≥ 2 windows to fit.

## Options

| Flag | Effect |
|------|--------|
| `--test` | Use the built-in `DEFAULT_CONFIG`; a config path is optional. |
| `--use-loo-ic` | LOO-IC model selection instead of the heuristic score. |
| `--use-hurdle` | Enable the Truncated-Binomial branch for sparse data. |
| `--hurdle-threshold N` | Structural-zero % that triggers the hurdle suggestion (default 70). |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Log verbosity (default INFO). |
| `--no-log-stdout` / `--no-log-file` | Turn off console or file logging. |

## Run modes

- **Single window** — one analysis period. Produces one set of reports and
  maps per selected level (res3 / res4 / adm1).
- **Iterative** — a rolling sequence of analysis windows (3/6/9/12 months)
  stepping back 1 month at a time, each with its derived baseline. The sweep
  runs once **per selected level**; per-window results are combined into a
  rolling hotspot report under `output/<timestamp>/iterative/`
  (`Iterative_Hotspots_Report_<level>_<ts>.xlsx`).

  > **Read each window on its own.** False-alarm (FDR) control is applied
  > *within* each window, not across the whole sequence. A hexagon flagged
  > in, say, 8 of 30 windows is **not** thereby stronger evidence — the
  > combined report is a stack of independent snapshots, not a
  > multiplicity-controlled set. Genuine time-trend / space-time
  > interaction is analysed separately (SaTScan), not here.

## Output

Everything lands in a timestamped folder `output/<YYYYMMDDhhmmss>/`
(`bayesian/`, `bayesian_covariates/`, `summary/`, `pipeline.log`). Per level
you get:

- `Report_*.xlsx` — per-unit results: counts, classification, posterior SMR
  mean/median + CI, reliability, and the watch-list columns
  (`on_watchlist`, `watch_reason`, `watch_rank`, `burden_rank`, `rate_rank`).
- `Map_*_EN.png` / `_ua.png` — anomaly choropleth (rigorous classification).
- `Reliability_Map_*` — HIGH / MODERATE / LOW reliability.
- `Watchlist_Map_*` — burden + rate triage map (units coloured by reason,
  labelled by priority rank).
- `Diagnostics_*.xlsx` and the `*_PPC` / `*_Forest` / `*_Pairs` plots.

See the README *Output* and *Watch-list* sections for the file-by-file
layout and how to read the watch-list alongside the classification.

## Verifying a change

```bash
python run_hotspots.py --test                   # end-to-end, should exit cleanly
python validation/test_convergence_gate.py      # convergence-gate smoke test
python validation/simulation_validation.py      # synthetic FDR / sensitivity / specificity
```
