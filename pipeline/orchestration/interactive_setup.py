"""
Interactive configuration entry-point.

Branches on ``run_mode``/``analysis_type`` and prompts (via
:class:`pipeline.config.interactive.InteractiveConfig`) for any missing
fields. Mutates ``config`` in place and returns the ``period_str``
derived from the chosen analysis period (or ``'iterative'`` for the
iterative mode).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from pipeline.config import InteractiveConfig

logger = logging.getLogger(__name__)


def run_interactive_setup(config: Dict[str, Any]) -> str:
    """Run the interactive wizard. Mutates ``config``; returns ``period_str``.

    The wizard always runs (no "use defaults?" shortcut and no config-driven
    auto-start). It asks: analysis type (standard / iterative), then the
    analysis window and period (baseline length is derived from the window),
    then the levels. Non-interactive / scripted runs bypass this entirely via
    ``validation/service_run.py``.
    """
    print("\n" + "=" * 60)
    print("HIV HOTSPOT DETECTION PIPELINE")
    print("=" * 60)

    config['data_type'] = 'facility_based'
    logger.info("Data type: Facility-based → Exchangeable model")

    analysis_type = InteractiveConfig.choose_analysis_type()
    config['analysis_type'] = analysis_type

    if analysis_type == 'iterative':
        # Iterative mode: one or more levels (res3/res4/adm1), each swept
        # separately, plus the analysis-window length (baseline derived).
        levels = InteractiveConfig.choose_levels()
        iter_am = InteractiveConfig.choose_iterative_analysis_window()
        config['iterative_levels'] = levels
        config['iterative_resolution'] = levels[0]   # current level (loop overwrites)
        config['iterative_analysis_months'] = iter_am
        config['analysis_mode'] = 'h3_hexagons'
        config['admin_levels'] = ['Oblast'] if 'Oblast' in levels else []
        config['hex_resolutions'] = [lv for lv in levels if isinstance(lv, int)]
        config['bayesian_parametrization'] = 'non_centered'

        start, end = InteractiveConfig.choose_iterative_date_range()
        config['iterative_date_range'] = {
            'start': start.strftime('%Y-%m-%d'),
            'end': end.strftime('%Y-%m-%d')
        }

        logger.info(f"Iterative mode: data {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        logger.info("  - Model: Bayesian non-centered only")
        logger.info(f"  - Levels: {levels}")
        logger.info(f"  - Window: {iter_am}-month analysis (baseline derived), 1-month step")
        return 'iterative'

    # Standard (single-window) mode: choose one or more levels (res3/res4/adm1).
    levels = InteractiveConfig.choose_levels()
    config['analysis_levels'] = levels
    config['hex_resolutions'] = [lv for lv in levels if isinstance(lv, int)]
    config['admin_levels'] = ['Oblast'] if 'Oblast' in levels else []
    config['analysis_mode'] = 'h3_hexagons'

    start, end = InteractiveConfig.choose_analysis_period()
    config['analysis_period'] = {
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d')
    }

    parametrization = InteractiveConfig.choose_parametrization()
    config['bayesian_parametrization'] = parametrization

    model_selection = InteractiveConfig.choose_model_selection()
    config['manual_model_selection'] = model_selection
    if model_selection != 'auto':
        _labels = {
            'bayesian': 'Bayesian only',
            'bayesian_covariates': 'Bayesian with Covariates only',
        }
        logger.info(f"[OK] Manual model selection: {_labels[model_selection]}")

    return start.strftime('%Y%m')
