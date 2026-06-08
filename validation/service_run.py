# -*- coding: utf-8 -*-
"""
Non-interactive pipeline runner (validation / scripted use).

In normal use the wizard always runs and collects the analysis type, window,
period and levels interactively. This helper bypasses the wizard: pass a
config JSON that already specifies ``analysis_type`` and the mode-specific
fields, and it builds the orchestrator, sets ``period_str``, and runs the full
pipeline. It is the "service file" used for background validation runs.

The config must contain, in addition to the usual paths/bayesian block:
  * standard mode:  "analysis_type": "standard", "analysis_period": {...},
                    "hex_resolutions": [..].
  * iterative mode: "analysis_type": "iterative", "iterative_date_range": {...},
                    "iterative_analysis_months": N, "iterative_resolution": R.

Run:
    python validation/service_run.py <config.json> [--use-hurdle] [--use-loo-ic]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path so this runs from the validation/ subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.bootstrap  # noqa: F401 -- env/warning setup before heavy imports

import argparse
from datetime import datetime

import pandas as pd

from pipeline.logging_setup import setup_logging
from pipeline.orchestrator import PipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-interactive pipeline runner")
    parser.add_argument('config', help='Config JSON with analysis_type + mode fields set')
    parser.add_argument('--use-hurdle', action='store_true', default=False)
    parser.add_argument('--use-loo-ic', action='store_true', default=False)
    parser.add_argument('--hurdle-threshold', type=float, default=70.0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_base = Path('output') / timestamp
    output_base.mkdir(parents=True, exist_ok=True)
    setup_logging(log_to_stdout=True, log_to_file=True,
                  log_file=output_base / 'pipeline.log', log_level='INFO')

    orch = PipelineOrchestrator(args.config, run_timestamp=timestamp, output_base=output_base,
                                use_loo_ic=args.use_loo_ic, use_hurdle=args.use_hurdle,
                                hurdle_threshold=args.hurdle_threshold)
    cfg = orch.config
    cfg.setdefault('data_type', 'facility_based')

    analysis_type = cfg.get('analysis_type', 'standard')
    if analysis_type == 'iterative':
        orch.period_str = 'iterative'
    else:
        orch.period_str = pd.to_datetime(cfg['analysis_period']['start']).strftime('%Y%m')

    orch.run_full_pipeline()
    print("SERVICE_RUN_DONE")


if __name__ == '__main__':
    main()
