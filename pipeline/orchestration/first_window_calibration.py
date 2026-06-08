"""
Calibration: run once on the first iterative window.

The iterative analyser runs many sliding-window fits. Running a full
multiplier-grid on every window would dominate the wall-clock
time and is unnecessary -- the priors don't change shape across
windows. So we calibrate ``sigma_hyperprior`` multipliers and the
``min_tests`` threshold on the first window and stamp the result back
into the module-level :data:`pipeline.constants.ANALYSIS_CONSTANTS`,
where every subsequent window will pick them up.

The analyser class is passed in (rather than imported) to keep the
dependency direction one-way and avoid a circular import.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Type

from pipeline.constants import ANALYSIS_CONSTANTS
from pipeline.orchestration.calibrate_min_tests import calibrate_min_tests
from pipeline.orchestration.calibrate_sigma_multipliers import calibrate_sigma_multipliers

logger = logging.getLogger(__name__)


def run_first_window_calibration(config: Dict[str, Any],
                        first_window: Dict[str, Any],
                        iterative_dir: Path,
                        analyzer_cls: Type,
                        orchestrator: Any,
                        random_seed: int = 42) -> None:
    """Calibrate sigma multipliers and min_tests on ``first_window``; stamp into ANALYSIS_CONSTANTS."""
    try:
        # Temporarily set config for first window
        original_period = config.get('analysis_period')
        config['analysis_period'] = {
            'start': first_window['analysis_start'].strftime('%Y-%m-%d'),
            'end': first_window['analysis_end'].strftime('%Y-%m-%d')
        }

        iter_lvl = config.get('iterative_resolution', 4)
        level_name = f'Hex_Res{iter_lvl}' if isinstance(iter_lvl, int) else str(iter_lvl)
        bayesian_cal = analyzer_cls(config, 'hex', orchestrator=orchestrator)
        gdf_cases_cal = bayesian_cal.load_cases()
        national_rate_cal, national_se_cal = bayesian_cal.calculate_national_baseline(
            gdf_cases_cal, first_window['baseline_start'], first_window['baseline_end'])
        gdf_cal = bayesian_cal.load_geodata(iter_lvl)
        gdf_cal = bayesian_cal.aggregate_stats(
            gdf_cal, gdf_cases_cal,
            first_window['analysis_start'], first_window['analysis_end'],
            first_window['baseline_start'], first_window['baseline_end'])

        # Restore original config
        if original_period:
            config['analysis_period'] = original_period

        opt_sigma = calibrate_sigma_multipliers(
            gdf_cal, level_name,
            national_rate_cal, national_se_cal,
            first_window['analysis_start'], first_window['analysis_end'],
            first_window['baseline_start'], first_window['baseline_end'],
            iterative_dir,
            random_seed=random_seed,
        )
        # Apply optimal multipliers to CONSTANTS
        ANALYSIS_CONSTANTS['sigma_hyperprior_small_sample_mult']['value'] = opt_sigma['small_sample_mult']
        ANALYSIS_CONSTANTS['sigma_hyperprior_local_density_mult']['value'] = opt_sigma['local_density_mult']
        ANALYSIS_CONSTANTS['sigma_hyperprior_se_high_mult']['value'] = opt_sigma['se_high_mult']
        ANALYSIS_CONSTANTS['sigma_hyperprior_se_moderate_mult']['value'] = opt_sigma['se_moderate_mult']
        logger.info(f"  [OK] Optimal sigma multipliers applied to all subsequent iterations")

        opt_min = calibrate_min_tests(gdf_cal, iterative_dir)
        ANALYSIS_CONSTANTS['min_tests_territory']['value'] = opt_min
        ANALYSIS_CONSTANTS['small_sample_threshold']['value'] = opt_min
        logger.info(f"  [OK] Optimal min_tests={opt_min} applied")

    except Exception as e:
        logger.warning(f"Calibration failed: {e} — using defaults")
    finally:
        # Bug fix: Clear PyTensor/PyMC compiled cache after calibration.
        # Calibration creates many models which can leave stale compiled
        # functions with fixed shapes that conflict with later windows.
        gc.collect()
