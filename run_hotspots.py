"""
HIV Hotspot Detection Pipeline -- entry point.

The pipeline implementation lives in the ``pipeline`` package
(``pipeline.orchestrator``, ``pipeline.analyzers``, etc.). This
file is the thin CLI shell that parses arguments, configures
logging, builds the orchestrator, and runs it.

Importing :mod:`pipeline.bootstrap` is the very first thing the
script does -- it sets ``KMP_DUPLICATE_LIB_OK`` and the warning
filters that PyMC/PyTensor/ArviZ would otherwise emit. It MUST
run before any heavy import (NumPy/PyMC/etc.) so the OpenMP and
warning configuration is in place when those libraries load.

Two compatibility re-exports are kept (``DEFAULT_CONFIG`` and
``BayesianAnalyzer``) because ``validation/test_convergence_gate.py``
imports them off this module via ``import run_hotspots as m``; once
that test is migrated those lines can go too.
"""

import pipeline.bootstrap  # noqa: F401 -- side-effect import

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from pipeline.analyzers import BayesianAnalyzer  # noqa: F401 -- re-exported for test_convergence_gate
from pipeline.config import InteractiveConfig
from pipeline.constants import DEFAULT_CONFIG
from pipeline.logging_setup import check_compiler_availability, setup_logging
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HIV Hotspot Detection Pipeline")
    parser.add_argument('--test', action='store_true', help='Run in test mode with default config')
    parser.add_argument('config', nargs='?', type=str, default=None, help='Path to config file')

    # Model selection arguments
    parser.add_argument('--use-loo-ic', action='store_true', default=False,
                       help='Use LOO-IC for model selection instead of heuristic scoring')
    parser.add_argument('--use-hurdle', action='store_true', default=False,
                       help='Use Hurdle Binomial model for sparse data with structural zeros')
    parser.add_argument('--hurdle-threshold', type=float, default=70.0,
                       help='Percentage of structural zeros to trigger Hurdle model (default: 70.0)')

    # Logging configuration arguments
    parser.add_argument('--log-stdout', action='store_true', default=True,
                       help='Enable logging to console (default: True)')
    parser.add_argument('--no-log-stdout', action='store_false', dest='log_stdout',
                       help='Disable logging to console')
    parser.add_argument('--log-file', action='store_true', default=True,
                       help='Enable logging to file (default: True)')
    parser.add_argument('--no-log-file', action='store_false', dest='log_file',
                       help='Disable logging to file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_base = Path(args.output_dir) if hasattr(args, 'output_dir') else Path('output')
    log_dir = output_base / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pipeline.log' if args.log_file else None
    setup_logging(log_to_stdout=args.log_stdout, log_to_file=args.log_file,
                 log_file=log_file, log_level=args.log_level)

    logger.info("="*80)
    logger.info("HIV HOTSPOT DETECTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info("")

    check_compiler_availability()

    output_base = log_dir

    if args.test:
        config_path = 'config_universal.json' if os.path.exists('config_universal.json') else None
        orchestrator = PipelineOrchestrator(config_path, run_timestamp=timestamp,
                                           output_base=output_base, use_loo_ic=args.use_loo_ic,
                                           use_hurdle=args.use_hurdle, hurdle_threshold=args.hurdle_threshold)
        if not config_path:
            orchestrator.config = DEFAULT_CONFIG.copy()
        orchestrator.run_full_pipeline()
        logger.info("\nTEST MODE COMPLETED!")
        return

    config_path = args.config or (sys.argv[1] if len(sys.argv) > 1 else None)
    orchestrator = PipelineOrchestrator(config_path, run_timestamp=timestamp,
                                       output_base=output_base, use_loo_ic=args.use_loo_ic,
                                       use_hurdle=args.use_hurdle, hurdle_threshold=args.hurdle_threshold)

    orchestrator.run_interactive_setup()
    if config_path and orchestrator.config.get('run_mode') != 'iterative':
        if InteractiveConfig.ask_overwrite_config(config_path):
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(orchestrator.config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")

    orchestrator.run_full_pipeline()

    logger.info("\nPIPELINE COMPLETED!")


if __name__ == "__main__":
    main()
else:
    # Default logging setup when imported as module (e.g. by tests).
    # Guard: skip in multiprocessing worker processes (Windows spawn).
    import multiprocessing as _mp
    if _mp.current_process().name == 'MainProcess':
        setup_logging(log_to_stdout=True, log_to_file=True,
                      log_file='pipeline.log', log_level='INFO')
