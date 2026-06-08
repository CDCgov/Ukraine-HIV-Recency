# -*- coding: utf-8 -*-
"""
Multi-seed stability check for the hotspot classification.

Purpose
-------
The pipeline fixes ``cfg['random_seed']`` so a run is reproducible, but a
reviewer will rightly ask: how much does the *conclusion* (which territories
are flagged, and as what) depend on that one seed? MCMC is stochastic, and on
small samples a borderline territory can flip category between seeds. This
script refits the real Bayesian model on the real data under several seeds and
reports how stable the per-territory ``classification_smr_sir`` label is.

What it does
------------
For each seed in ``--seeds`` it runs one analysis window (the config's
``analysis_period``) at one H3 resolution through the actual
:class:`~pipeline.analyzers.bayesian.BayesianAnalyzer` -- same aggregation,
same model, same FDR-controlled SIR/SMR taxonomy as production -- and records
each territory's label. It then reports, per territory, the modal label and the
fraction of seeds that agree with it, plus an overall summary (mean agreement,
how many territories were perfectly stable, and how many ever entered the
hotspot set). Territories that are never in any hotspot category under any seed
are omitted from the per-territory table to keep it short; they are counted in
the summary.

This is intentionally a *separate* offline diagnostic: it is slow (one full
MCMC fit per seed) and is not part of the routine pipeline.

Run:
    python multiseed_stability.py                       # default config, seeds 42..46
    python multiseed_stability.py config.json --seeds 1 2 3 4 5 --resolution 4
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path so this script runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.bootstrap  # noqa: F401 -- env / warning setup before heavy imports

import argparse
import json
import logging
from collections import Counter
from typing import Dict, List

import pandas as pd

from pipeline.analyzers import BayesianAnalyzer
from pipeline.aggregation.periods import get_periods
from pipeline.classification import HOTSPOT_LABELS
from pipeline.constants import DEFAULT_CONFIG
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _territory_key_column(gdf) -> str:
    """Pick a stable per-territory id column for aligning results across seeds."""
    for col in ('h3_id', 'h3index', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN', 'territory_id'):
        if col in gdf.columns:
            return col
    # Fall back to the frame index rendered as a string.
    return '__index__'


def _labels_for_seed(config: Dict, seed: int, resolution: int) -> Dict[str, str]:
    """Fit the model once under ``seed`` and return ``{territory_id: label}``.

    Mirrors the single-window fit path used by the orchestrator: load cases,
    compute the national baseline over the baseline window, load and aggregate
    the geometry for the requested resolution, then run the Bayesian model.
    Only the seed differs between calls.
    """
    config = dict(config)
    config['random_seed'] = seed

    start, end, b_start, b_end = get_periods(config)

    analyzer = BayesianAnalyzer(config, 'hex', orchestrator=None)
    gdf_cases = analyzer.load_cases()
    national_rate, national_se = analyzer.calculate_national_baseline(gdf_cases, b_start, b_end)
    gdf = analyzer.load_geodata(resolution)
    gdf = analyzer.aggregate_stats(gdf, gdf_cases, start, end, b_start, b_end)

    gdf_result, diagnostics = analyzer.run_model(
        gdf, f"Hex_Res{resolution}", national_rate, national_se)

    if diagnostics is None:
        logger.warning(f"seed={seed}: model returned no diagnostics; skipping")
        return {}

    key_col = _territory_key_column(gdf_result)
    active = gdf_result[gdf_result['all_tested_curr'] > 0]
    labels: Dict[str, str] = {}
    for idx, row in active.iterrows():
        key = str(row[key_col]) if key_col != '__index__' else str(idx)
        labels[key] = str(row.get('classification_smr_sir', 'No Data'))
    return labels


def summarise(per_seed_labels: Dict[int, Dict[str, str]]) -> None:
    """Print the per-territory agreement table and the overall stability summary."""
    seeds = sorted(per_seed_labels)
    n_seeds = len(seeds)
    if n_seeds == 0:
        logger.error("No successful fits — nothing to summarise.")
        return

    territories = sorted({t for labels in per_seed_labels.values() for t in labels})

    rows = []
    perfectly_stable = 0
    ever_hotspot = 0
    agreements = []
    for t in territories:
        labels = [per_seed_labels[s].get(t, 'No Data') for s in seeds]
        counts = Counter(labels)
        modal_label, modal_n = counts.most_common(1)[0]
        agreement = modal_n / n_seeds
        agreements.append(agreement)
        if modal_n == n_seeds:
            perfectly_stable += 1
        if any(lbl in HOTSPOT_LABELS for lbl in labels):
            ever_hotspot += 1
            rows.append((t, modal_label, agreement, counts))

    print("=" * 100)
    print(f"Multi-seed classification stability  |  {n_seeds} seeds: {seeds}")
    print("=" * 100)
    print(f"Territories analysed (active):       {len(territories)}")
    print(f"Mean per-territory agreement:        {sum(agreements) / len(agreements):.3f}")
    print(f"Perfectly stable (all seeds agree):  {perfectly_stable}/{len(territories)} "
          f"({perfectly_stable / len(territories) * 100:.1f}%)")
    print(f"Ever flagged as a hotspot:           {ever_hotspot}")
    print("-" * 100)

    if rows:
        print("Territories that entered a hotspot category under at least one seed:")
        print(f"{'territory':<28}{'modal label':<28}{'agree':>7}   label spread")
        print("-" * 100)
        for t, modal_label, agreement, counts in sorted(rows, key=lambda r: r[2]):
            spread = ", ".join(f"{lbl}×{n}" for lbl, n in counts.most_common())
            print(f"{t[:27]:<28}{modal_label[:27]:<28}{agreement:>7.2f}   {spread}")
    else:
        print("No territory was flagged as a hotspot under any seed.")
    print("-" * 100)
    print("Reading the table: agreement = fraction of seeds giving the modal label. "
          "Low agreement on a hotspot territory means its call is seed-sensitive "
          "(usually a small-sample borderline case) and should be reported with that caveat.")


def main() -> None:
    """Parse arguments, fit under each seed, and print the stability summary."""
    parser = argparse.ArgumentParser(description="Multi-seed hotspot classification stability check")
    parser.add_argument('config', nargs='?', default=None, help='Config JSON (defaults to DEFAULT_CONFIG)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                        help='Random seeds to compare (default: 42 43 44 45 46)')
    parser.add_argument('--resolution', type=int, default=4, help='H3 resolution to fit (default: 4)')
    args = parser.parse_args()

    setup_logging(log_to_stdout=True, log_to_file=False, log_level='WARNING')

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = dict(DEFAULT_CONFIG)

    logger.warning(f"Fitting {len(args.seeds)} seeds at H3 res{args.resolution}; "
                   "this runs one full MCMC fit per seed and may take several minutes.")

    per_seed_labels: Dict[int, Dict[str, str]] = {}
    for seed in args.seeds:
        logger.warning(f"--- seed {seed} ---")
        labels = _labels_for_seed(config, seed, args.resolution)
        if labels:
            per_seed_labels[seed] = labels

    summarise(per_seed_labels)


if __name__ == '__main__':
    main()
