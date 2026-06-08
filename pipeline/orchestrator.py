"""
:class:`PipelineOrchestrator` -- the top-level coordinator that runs
the full hotspot detection pipeline.

Owns the configuration, the output-directory layout (session +
per-model subtrees), the analyzer instances, the audit-trail
registry, and the cross-stage state shared between the per-mode
analysis runs (run_for_mode), the iterative sliding-window scan
(run_iterative_analysis), and the post-pipeline reports.

Almost every actual stage is delegated to a free function in
:mod:`pipeline.orchestration`; this class is the integration
layer that gives those free functions the orchestrator state they
need (output paths, current audit trail, analyzer instances,
historical comparison handle).

Path resolution: ``Path(__file__).resolve().parent.parent`` gives
the repo root from this module's location (``pipeline/orchestrator.py``).
The same trick is used in :mod:`pipeline.analyzers.base` so config
paths declared relative to the repo root keep working from any
analyzer or orchestrator instance.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import pymc as pm
from dateutil.relativedelta import relativedelta

from pipeline.analyzers import BayesianAnalyzer, BayesianCovariatesAnalyzer
from pipeline.classification import (
    HOTSPOT_LABELS,
    is_hotspot as _is_hotspot,
)
from pipeline.config import InteractiveConfig, ModelConfigurationWizard
from pipeline.config.models import BayesianConfig
from pipeline.constants import ANALYSIS_CONSTANTS, DEFAULT_CONFIG, PIPELINE_VERSION
from pipeline.diagnostics import (
    DataQualityChecker,
    DiagnosticInterpreter,
    DiagnosticPlotter,
    ReliabilityScoreCalculator,
)
from pipeline.history import HistoricalComparison
from pipeline.orchestration import (
    analyze_specification as _orch_analyze_specification,
    assess_data_quality as _orch_assess_data_quality,
    build_model_comparison_entry as _orch_build_model_comparison_entry,
    calibrate_min_tests as _orch_calibrate_min_tests,
    calibrate_sigma_multipliers as _orch_calibrate_sigma_multipliers,
    check_diagnostics as _orch_check_diagnostics,
    collect_current_results as _orch_collect_current_results,
    collect_dashboard_data as _orch_collect_dashboard_data,
    create_bayesian_covariates_plots as _orch_create_bayesian_covariates_plots,
    create_bayesian_plots as _orch_create_bayesian_plots,
    create_summary_dashboard as _orch_create_summary_dashboard,
    finalize_model_choice as _orch_finalize_model_choice,
    generate_audit_trail_reports as _orch_generate_audit_trail_reports,
    generate_comparison_report as _orch_generate_comparison_report,
    generate_iterative_hotspots_report as _orch_generate_iterative_hotspots_report,
    generate_model_comparison as _orch_generate_model_comparison,
    generate_recommendations as _orch_generate_recommendations,
    interpret_bayesian_covariates_diagnostics as _orch_interpret_bayesian_covariates_diagnostics,
    interpret_bayesian_diagnostics as _orch_interpret_bayesian_diagnostics,
    prepare_iterative_windows as _orch_prepare_iterative_windows,
    resolve_paths as _orch_resolve_paths,
    run_bayesian_covariates_dispatch as _orch_run_bayesian_covariates_dispatch,
    run_bayesian_dispatch as _orch_run_bayesian_dispatch,
    run_historical_comparison as _orch_run_historical_comparison,
    run_interactive_setup as _orch_run_interactive_setup,
    run_iterative_loop as _orch_run_iterative_loop,
    run_first_window_calibration as _orch_run_first_window_calibration,
    run_wizard_and_record_decisions as _orch_run_wizard_and_record_decisions,
    validate_config as _orch_validate_config,
    write_skipped_windows_table as _orch_write_skipped_windows_table,
)
from pipeline.reporting import SummaryDashboard
from pipeline.spec import AutoSpecificationSystem

# Decision Audit Trail (optional). Same try/except shape as the main
# script kept here so the orchestrator works whether or not the
# optional ``decision_audit_trail`` package is installed.
try:
    from decision_audit_trail import DecisionAuditTrail
    AUDIT_TRAIL_AVAILABLE = True
except ImportError:
    AUDIT_TRAIL_AVAILABLE = False

    class DecisionAuditTrail:
        """No-op fallback used when the real audit-trail module is absent.

        Mirrors the public API of the real class so call sites work
        unchanged; every method simply does nothing (or returns ``self``
        for the stage context helpers), so no audit artefacts are written.
        """

        def __init__(self, *args, **kwargs):
            """Accept any arguments and start with no open stage."""
            self.current_stage = None

        def start_stage(self, *args, **kwargs):
            """No-op stage open; returns self for call chaining."""
            return self

        def end_stage(self, *args, **kwargs):
            """No-op stage close; clears the current stage."""
            self.current_stage = None

        def start_substage(self, *args, **kwargs):
            """No-op substage open; returns self for call chaining."""
            return self

        def end_substage(self, *args, **kwargs):
            """No-op substage close; returns self for call chaining."""
            return self

        def add_decision(self, *args, **kwargs):
            """No-op: a recorded decision is discarded."""
            pass

        def set_metadata(self, *args, **kwargs):
            """No-op: stage metadata is discarded."""
            pass

        def generate_markdown(self, *args, **kwargs):
            """No-op: no Markdown audit report is produced."""
            pass

        def generate_html(self, *args, **kwargs):
            """No-op: no HTML audit report is produced."""
            pass

        def to_json(self, *args, **kwargs):
            """No-op: no JSON audit report is produced."""
            pass


logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete analysis pipeline."""

    def __init__(self, config_path: Optional[str] = None, run_timestamp: Optional[str] = None,
                 output_base: Optional[Path] = None, use_loo_ic: bool = False,
                 use_hurdle: bool = False, hurdle_threshold: float = 70.0):
        """Build the orchestrator and fix the output-directory layout.

        Args:
            config_path: JSON config to load; ``None`` uses ``DEFAULT_CONFIG``.
            run_timestamp: shared ``YYYYMMDDhhmmss`` stamp from ``main()`` so
                every artefact of this run lands under one folder.
            output_base: pre-made timestamped session directory; when given it
                is used verbatim (``main()`` already created it), otherwise the
                path is derived from ``config['output_dir']`` on first write.
            use_loo_ic: select models via LOO-IC instead of the heuristic score.
            use_hurdle: enable the Truncated-Binomial branch for sparse data.
            hurdle_threshold: structural-zero percentage that triggers it.
        """
        self.config = self._load_config(config_path)
        self.bayesian = None
        self.bayesian_cov = None
        self.results = {}
        self.period_str = None
        self.run_timestamp = run_timestamp  # Set from main() for consistent timestamping
        self.model_comparison_data = []  # Store model comparison data
        self.use_loo_ic = use_loo_ic  # Use LOO-IC for model selection
        self.use_hurdle = use_hurdle  # Use Hurdle model for sparse data
        self.hurdle_threshold = hurdle_threshold  # Threshold for structural zeros %

        # Decision Audit Trail - will be initialized per level
        self.audit_trails = {}  # Dictionary to store audit trail for each level

        # Use output_base from main() if provided, otherwise fall back to config
        if output_base is not None:
            # output_base is already the full path: output/20260429232156/
            # Don't use .parent - use it directly
            self.output_dir = Path('output')  # Base output directory
            self.session_dir = output_base  # Full session directory with timestamp
        else:
            self.output_dir = Path(self.config.get('output_dir', 'output'))
            self.session_dir = None  # Will be created in get_output_path()

    def get_output_path(self, model_type: str, level_name: str, filename: str,
                        is_hex: bool = False) -> Path:
        """
        Creates correct path for saving file in new structure.

        Args:
            model_type: "bayesian", "bayesian_covariates", or "summary"
            level_name: "Community", "District", "Oblast", "Hex_Res3", "Hex_Res4", "Hex_Res5"
            filename: File name (e.g., "Map_Community_202601.png")
            is_hex: True if this is hexagonal grid

        Returns:
            Path: Full path to file
        """
        # Use timestamp instead of period (e.g., "20260428164352")
        if self.run_timestamp is None:
            self.run_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Use session_dir if available (from main()), otherwise create from output_dir
        if self.session_dir is not None:
            period_dir = self.session_dir
        else:
            period_dir = self.output_dir / self.run_timestamp

        if model_type == "summary":
            # General files
            output_path = period_dir / "summary"
        else:
            # Model files
            territory_type = "hex" if is_hex else "admin"

            # For hex: level_name = "Hex_Res4" -> "res4"
            # For admin: level_name = "Community" -> "Community"
            if is_hex:
                level_dir = level_name.replace("Hex_Res", "res").lower()
            else:
                level_dir = level_name

            output_path = period_dir / model_type / territory_type / level_dir

        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        return output_path / filename

    def _run_loo_ic_model_selection(self, df_analysis: pd.DataFrame, gdf: gpd.GeoDataFrame,
                                     level_name: str, national_rate: float,
                                     national_se: float) -> Dict[str, Any]:
        """Compare Binomial vs Beta-Binomial by LOO-IC on the active territories.

        Fits both likelihoods on the same exchangeable mean structure and uses
        leave-one-out cross-validation to test whether the data actually
        support the extra-binomial variation, or whether the overdispersion
        parameter is essentially unidentified on these small counts (audit
        C3-B / Mo2). These are dedicated diagnostic fits, separate from the
        production fit, so they do not affect the classification or maps.

        The heuristic specification summary is still produced (it drives the
        ``Specification_Analysis`` report and surfaces zero-inflation /
        outlier flags); the LOO verdict is merged into it.

        Args:
            df_analysis: active territories (``all_tested_curr > 0``).
            gdf: full GeoDataFrame (unused here; kept for signature stability).
            level_name: level being analysed (for logging).
            national_rate / national_se: national baseline summaries.

        Returns:
            The specification-analysis dict, augmented with the LOO comparison
            fields and a data-driven ``recommended_model`` string.
        """
        from pipeline.diagnostics.overdispersion import compare_binomial_betabinomial

        analysis = AutoSpecificationSystem.recommend_specification(
            df_analysis,
            y_col='recent_count_curr',
            n_col='all_tested_curr'
        )

        try:
            loo_result = compare_binomial_betabinomial(df_analysis, national_rate, self.config)
            analysis.setdefault('data_analysis', {}).update(loo_result)

            best = loo_result.get('loo_best_model')
            if best is None:
                analysis.setdefault('warnings', []).append(
                    f"LOO comparison unavailable: {loo_result.get('error', 'unknown error')}")
            elif loo_result.get('overdispersion_supported'):
                analysis['recommended_model'] = (
                    'Beta-Binomial (overdispersion supported by LOO)')
            else:
                analysis['recommended_model'] = (
                    'Beta-Binomial (Binomial not rejected by LOO; overdispersion weak — '
                    'Beta-Binomial nests it harmlessly)')
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.error(f"LOO-IC model comparison failed: {e}")
            analysis.setdefault('warnings', []).append(f"LOO-IC comparison failed: {e}")

        return analysis

    def _validate_config(self, config: Dict) -> Dict:
        """Thin wrapper around :func:`pipeline.orchestration.validate_config`."""
        return _orch_validate_config(config, bayesian_config_cls=BayesianConfig)

    def _resolve_paths(self, config: Dict) -> Dict:
        """Thin wrapper around :func:`pipeline.orchestration.resolve_paths`."""
        return _orch_resolve_paths(config, Path(__file__).resolve().parent.parent)

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            config = self._validate_config(config)
            config = self._resolve_paths(config)
            return config
        return DEFAULT_CONFIG.copy()

    def run_interactive_setup(self):
        """Thin wrapper around :func:`pipeline.orchestration.run_interactive_setup`."""
        self.period_str = _orch_run_interactive_setup(self.config)

    def check_diagnostics(self, diagnostics: dict) -> bool:
        """Thin wrapper around :func:`pipeline.orchestration.check_diagnostics`."""
        return _orch_check_diagnostics(diagnostics)

    def run_for_mode(self, mode: str, levels: List):
        """Run analysis for specified mode and levels."""
        # DiagnosticInterpreter already imported from improvements_v1 at top
        # model_specification and diagnostic_plots modules are optional

        logger.info("\n" + "=" * 60)
        logger.info(f"ANALYZING: {mode.upper()}")
        logger.info("=" * 60)

        mode_suffix = 'hex'
        period_str = self.period_str

        # Initialize analyzers with orchestrator
        bayesian = BayesianAnalyzer(self.config, mode_suffix, orchestrator=self)
        bayesian_cov = BayesianCovariatesAnalyzer(self.config, mode_suffix, orchestrator=self)

        # Store current audit trail in orchestrator for access by analyzers
        self.current_audit_trail = None

        # Initialize DiagnosticPlotter
        plotter = DiagnosticPlotter()

        # Get periods
        start, end, b_start, b_end = bayesian.get_periods()

        # Load data once
        gdf_cases = bayesian.load_cases()
        national_rate, national_se = bayesian.calculate_national_baseline(gdf_cases, b_start, b_end)

        # Load testing sites data (optional - for network analysis)
        df_sites = bayesian.load_testing_sites(self.config['excel_path'])
        if df_sites is not None:
            logger.info("[OK] Testing sites data loaded - network analysis enabled")
        else:
            logger.info("[WARN] Testing sites data not available - network analysis disabled")

        all_summaries = []

        # Initialize interpretation variables to avoid locals() issues
        bayesian_interpretation = None
        bayesian_cov_interpretation = None

        for level in levels:
            logger.info("\n" + "-" * 40)
            logger.info(f"LEVEL: {level}")
            logger.info("-" * 40)

            level_name = level if isinstance(level, str) else f'Hex_Res{level}'

            # Initialize Decision Audit Trail for this level
            audit_trail = DecisionAuditTrail(
                level_name=level_name,
                run_timestamp=datetime.now()
            )
            self.audit_trails[level_name] = audit_trail

            # Make audit trail accessible to analyzers via orchestrator
            self.current_audit_trail = audit_trail

            logger.info(f"Decision Audit Trail initialized for {level_name}")

            # Initialize tracking variables for this level
            final_gdf = None
            model_used = None
            final_diag = None

            # Load geographic data
            gdf = bayesian.load_geodata(level)

            # CRITICAL WARNING for large Bayesian models (H3 Res 4+)
            if isinstance(level, int) and level >= 4:
                n_territories = len(gdf)
                if n_territories > 5000:
                    logger.warning("=" * 80)
                    logger.warning("[WARN] CRITICAL: VERY LARGE GRID DETECTED")
                    logger.warning(f"[WARN] Territories: {n_territories}")
                    logger.warning(f"[WARN] Estimated time for Bayesian models: {n_territories/100:.0f}-{n_territories/50:.0f} minutes")
                    logger.warning("[WARN] Recommendation: Use H3 Res 4 instead")
                    logger.warning("=" * 80)

                    # Interactive confirmation if running in terminal
                    import sys
                    if sys.stdin.isatty():
                        try:
                            response = input("Continue with Bayesian models? (y/n): ").lower().strip()
                            if response != 'y':
                                logger.info("User chose to skip Bayesian models for this level")
                                bayesian = None
                                bayesian_cov = None
                        except (EOFError, KeyboardInterrupt):
                            logger.info("Skipping Bayesian models for this level")
                            bayesian = None
                            bayesian_cov = None
                    else:
                        logger.warning("Non-interactive mode: proceeding with Bayesian models (may take hours)")

            # Aggregate stats
            gdf = bayesian.aggregate_stats(gdf, gdf_cases, start, end, b_start, b_end)

            _orch_assess_data_quality(audit_trail, gdf)

            # Cache original CLI args on first level; reuse on subsequent levels
            if not hasattr(self, '_original_cli_use_hurdle'):
                self._original_cli_use_hurdle = self.use_hurdle
                self._original_cli_hurdle_threshold = self.hurdle_threshold
                self._original_cli_use_loo_ic = self.use_loo_ic

            cli_args = {
                'use_hurdle': self._original_cli_use_hurdle,
                'hurdle_threshold': self._original_cli_hurdle_threshold,
                'use_loo_ic': self._original_cli_use_loo_ic
            }

            _wiz = _orch_run_wizard_and_record_decisions(
                audit_trail, gdf, level_name, cli_args, self.config
            )
            level_use_hurdle = _wiz['use_hurdle']
            level_hurdle_threshold = _wiz['hurdle_threshold']
            level_use_loo_ic = _wiz['use_loo_ic']
            pct_structural_zeros = _wiz['pct_structural_zeros']

            # Manual model selection flags
            _manual_model = self.config.get('manual_model_selection', 'auto')
            _force_bayesian_only  = (_manual_model == 'bayesian')
            _force_bayes_cov_only = (_manual_model == 'bayesian_covariates')

            spec_analysis, recommended_model = _orch_analyze_specification(
                self, gdf, level_name, period_str,
                national_rate, national_se, level_use_loo_ic, audit_trail,
            )

            # === Apply manual model selection override ===
            if _force_bayesian_only or _force_bayes_cov_only:
                _label = 'Bayesian only' if _force_bayesian_only else 'Bayesian Covariates only'
                logger.info(f"MANUAL MODEL SELECTION: {_label}")

            # === Bayesian Analysis ===
            gdf_bayes, diag_bayes = _orch_run_bayesian_dispatch(
                bayesian, gdf, level_name, national_rate, national_se,
                level_use_hurdle, level_hurdle_threshold,
                _force_bayes_cov_only, self.config,
            )

            if diag_bayes:
                bayesian.diagnostics.append(diag_bayes)

            # === Interpret Bayesian diagnostics ===
            bayesian_interpretation = _orch_interpret_bayesian_diagnostics(
                self, diag_bayes, level_name, period_str,
            )

            # === Visualizations for Bayesian ===
            _orch_create_bayesian_plots(self, diag_bayes, level_name, period_str, plotter)

            if not _force_bayes_cov_only and diag_bayes is not None:
                gdf_bayes = ReliabilityScoreCalculator.calculate_territory_scores(gdf_bayes, diag_bayes, self.config)
                bayesian.plot_map(gdf_bayes, level_name, start, end, b_start, b_end, "Bayesian",
                                  diagnostics=diag_bayes)
                bayesian.save_report(gdf_bayes, level_name, period_str, diagnostics=diag_bayes)
                bayesian.plot_reliability_map(gdf_bayes, level_name, start, end, "Bayesian")
                bayesian.plot_watchlist_map(gdf_bayes, level_name, start, end, "Bayesian")

            if not _force_bayes_cov_only:
                model_used, final_diag, final_gdf = _orch_finalize_model_choice(
                    diag_bayes, gdf_bayes, level_name, 'Bayesian',
                    model_used, final_diag, final_gdf, all_summaries,
                    report_kept_previous=False,
                )

            gdf_bayes_cov, diag_bayes_cov = _orch_run_bayesian_covariates_dispatch(
                bayesian, bayesian_cov, gdf, gdf_cases, level, level_name,
                start, end, b_start, b_end, national_rate, national_se,
                _force_bayesian_only, self.config,
            )

            bayesian_cov_interpretation = _orch_interpret_bayesian_covariates_diagnostics(
                self, diag_bayes_cov, level_name, period_str,
            )

            _orch_create_bayesian_covariates_plots(self, diag_bayes_cov, level_name, period_str, plotter)

            if diag_bayes_cov is not None:
                gdf_bayes_cov = ReliabilityScoreCalculator.calculate_territory_scores(gdf_bayes_cov, diag_bayes_cov, self.config)
                bayesian_cov.plot_map(gdf_bayes_cov, level_name, start, end, b_start, b_end,
                                      "Bayesian with Covariates", diagnostics=diag_bayes_cov)
                bayesian_cov.save_report(gdf_bayes_cov, level_name, period_str, diagnostics=diag_bayes_cov)
                bayesian_cov.plot_reliability_map(gdf_bayes_cov, level_name, start, end, "Bayesian with Covariates")
                bayesian_cov.plot_watchlist_map(gdf_bayes_cov, level_name, start, end, "Bayesian with Covariates")

            model_used, final_diag, final_gdf = _orch_finalize_model_choice(
                diag_bayes_cov, gdf_bayes_cov, level_name, 'Bayesian with Covariates',
                model_used, final_diag, final_gdf, all_summaries,
                report_kept_previous=True,
            )

            _orch_generate_comparison_report(
                self, level_name, period_str,
                bayesian_interpretation, bayesian_cov_interpretation,
                final_diag, final_gdf,
            )

            # Store comparison data
            self._add_model_comparison(level_name, model_used, final_diag)

            # Store results for dashboard
            if final_gdf is not None and model_used is not None:
                self.results[level_name] = {
                    'gdf': final_gdf,
                    'diagnostics': final_diag,
                    'model_used': model_used
                }
                logger.debug(f"Stored results for {level_name}: model={model_used}, gdf_shape={final_gdf.shape}")
            else:
                logger.warning(f"No valid model results for {level_name} - skipping dashboard data")

        # Save diagnostics
        if bayesian.diagnostics:
            bayesian.save_diagnostics(level_name, period_str)
        if bayesian_cov.diagnostics:
            bayesian_cov.save_diagnostics(level_name, period_str)

        _orch_generate_audit_trail_reports(self, period_str)

        return '\n'.join(all_summaries)

    def run_full_pipeline(self):
        """Run the whole analysis end to end.

        Dispatches to the iterative sliding-window driver when
        ``run_mode``/``analysis_type`` requests it, otherwise runs a single
        analysis window. Each configured level (hex resolution or admin
        level) is aggregated, fitted, diagnosed, classified and reported;
        the run finishes with the summary dashboard, historical comparison
        and audit-trail reports.
        """
        # Set timestamp at the beginning of analysis
        self.run_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        logger.info(f"Starting analysis run: {self.run_timestamp}")

        # Check if iterative mode
        analysis_type = self.config.get('analysis_type', 'standard')

        if analysis_type == 'iterative':
            # Run iterative hotspot search
            self.run_iterative_analysis()
            return

        # Standard mode — run each selected level (H3 resolutions and/or the
        # ADM1 oblast level). ``analysis_levels`` is a mix of ints (hex
        # resolutions) and the string 'Oblast'; falls back to the hex
        # resolutions for older configs.
        levels = self.config.get('analysis_levels') or self.config.get('hex_resolutions', [4])
        self.run_for_mode('analysis', levels)

        # Generate model comparison
        self.generate_model_comparison()

        # Generate recommendations
        self.generate_recommendations()

        _orch_create_summary_dashboard(self, SummaryDashboard)
        _orch_run_historical_comparison(self, HistoricalComparison, PIPELINE_VERSION)

    def _collect_dashboard_data(self) -> dict:
        """Thin wrapper around :func:`pipeline.orchestration.collect_dashboard_data`."""
        return _orch_collect_dashboard_data(self.results, _is_hotspot)

    def _collect_current_results(self) -> dict:
        """Thin wrapper around :func:`pipeline.orchestration.collect_current_results`."""
        return _orch_collect_current_results(self.results, self.period_str,
                                              PIPELINE_VERSION, _is_hotspot)

    def _add_model_comparison(self, level_name: str, model_used: str, final_diag: dict):
        """Thin wrapper around :func:`pipeline.orchestration.build_model_comparison_entry`."""
        self.model_comparison_data.append(
            _orch_build_model_comparison_entry(level_name, model_used, final_diag)
        )

    def run_iterative_analysis(self):
        """
        Run iterative hotspot search with sliding window.

        - Analysis period: 3 months
        - Baseline period: 12 months
        - Step: 1 month backward
        - Model: Bayesian non-centered only
        - Level: Res4 hexagons only
        """
        logger.info("\n" + "=" * 60)
        logger.info("ITERATIVE HOTSPOT SEARCH")
        logger.info("=" * 60)

        # Load cases, attach H3 res-4 IDs, filter, and build sliding windows
        df_cases, windows = _orch_prepare_iterative_windows(self.config)

        # Create output directory for iterative results
        # Use session_dir if available, otherwise create from output_dir
        if self.session_dir is not None:
            iterative_dir = self.session_dir / "iterative"
        else:
            iterative_dir = self.output_dir / self.run_timestamp / "iterative"
        iterative_dir.mkdir(parents=True, exist_ok=True)

        # Run the full sliding-window sweep once per selected level (each level
        # — an H3 resolution int or 'Oblast' — is independent and gets its own
        # calibration, per-window maps and aggregated report).
        levels = self.config.get('iterative_levels') or [self.config.get('iterative_resolution', 4)]

        for lvl in levels:
            self.config['iterative_resolution'] = lvl
            level_name = f'Hex_Res{lvl}' if isinstance(lvl, int) else str(lvl)
            logger.info("\n" + "=" * 60)
            logger.info(f"ITERATIVE LEVEL: {level_name}")
            logger.info("=" * 60)

            # Calibrate sigma multipliers / min_tests on the first window.
            if len(windows) > 0:
                _orch_run_first_window_calibration(
                    self.config, windows[0], iterative_dir,
                    analyzer_cls=BayesianAnalyzer,
                    orchestrator=self,
                    random_seed=self.config.get('random_seed', 42),
                )

            # Windows whose posterior was flagged convergence_fatal, recorded
            # so the aggregated report shows them explicitly. Reset per level.
            self._iterative_skipped_windows: List[Dict[str, Any]] = []

            all_hotspots = _orch_run_iterative_loop(self, windows, _is_hotspot)

            # Each iteration is independent — no cross-iteration FDR.
            if all_hotspots:
                df_all_hotspots = pd.concat(all_hotspots, ignore_index=True)
                logger.info(f"  [{level_name}] {len(df_all_hotspots)} hotspot detection(s) "
                            f"across {len(all_hotspots)} iteration(s)")
                self._generate_iterative_hotspots_report(all_hotspots, iterative_dir, level=lvl)
                _orch_write_skipped_windows_table(
                    self._iterative_skipped_windows, iterative_dir, level_name=level_name)
            else:
                logger.warning(f"No hotspots found in any iteration for {level_name}")
        else:
            logger.warning("No hotspots found in any iteration")

    def _run_bayesian_for_window(self, window: Dict) -> Optional[gpd.GeoDataFrame]:
        """
        Run Bayesian non-centered model for a specific time window.

        Args:
            window: Dictionary with analysis_start, analysis_end, baseline_start, baseline_end

        Returns:
            GeoDataFrame with results or None if failed
        """
        # Temporarily set config for this window
        original_period = self.config.get('analysis_period')

        self.config['analysis_period'] = {
            'start': window['analysis_start'].strftime('%Y-%m-%d'),
            'end': window['analysis_end'].strftime('%Y-%m-%d')
        }

        try:
            # Initialize Bayesian analyzer
            bayesian = BayesianAnalyzer(self.config, 'hex', orchestrator=self)

            # Get periods
            start = window['analysis_start']
            end = window['analysis_end']
            b_start = window['baseline_start']
            b_end = window['baseline_end']

            # Load data
            gdf_cases = bayesian.load_cases()
            national_rate, national_se = bayesian.calculate_national_baseline(gdf_cases, b_start, b_end)

            # Load geometry at the configured iterative level: an H3 resolution
            # int (res3 / res4) or the string 'Oblast' (ADM1).
            lvl = self.config.get('iterative_resolution', 4)
            level_name = f'Hex_Res{lvl}' if isinstance(lvl, int) else str(lvl)
            gdf = bayesian.load_geodata(lvl)

            # Aggregate stats
            gdf = bayesian.aggregate_stats(gdf, gdf_cases, start, end, b_start, b_end)

            # Use original CLI parameters for iterative mode
            iter_use_hurdle = self._original_cli_use_hurdle if hasattr(self, '_original_cli_use_hurdle') else self.use_hurdle
            iter_hurdle_threshold = self._original_cli_hurdle_threshold if hasattr(self, '_original_cli_hurdle_threshold') else self.hurdle_threshold

            # Run Bayesian model (with Hurdle option)
            if iter_use_hurdle and 'site_present' in gdf.columns:
                n_total = len(gdf)
                n_structural_zeros = (~gdf['site_present']).sum()
                pct_structural = (n_structural_zeros / n_total) * 100

                if pct_structural >= iter_hurdle_threshold:
                    logger.info(f"Iteration {window['iteration']}: Using Truncated Binomial (active sites) ({pct_structural:.1f}% structural zeros)")
                    gdf_result, diagnostics = bayesian.run_hurdle_model(gdf, level_name, national_rate)
                else:
                    logger.info(f"Iteration {window['iteration']}: Using standard Bayesian ({pct_structural:.1f}% structural zeros)")
                    gdf_result, diagnostics = bayesian.run_model(
                        gdf, level_name, national_rate, national_se, parametrization='non_centered'
                    )
            else:
                if not iter_use_hurdle:
                    logger.info(f"Iteration {window['iteration']}: Using standard Bayesian (Hurdle disabled)")
                elif 'site_present' not in gdf.columns:
                    logger.warning(f"Iteration {window['iteration']}: site_present missing - using standard Bayesian")

                gdf_result, diagnostics = bayesian.run_model(
                    gdf, level_name, national_rate, national_se, parametrization='non_centered'
                )

            # Calculate reliability scores FOR EACH TERRITORY
            if diagnostics:
                gdf_result = ReliabilityScoreCalculator.calculate_territory_scores(gdf_result, diagnostics, self.config)

                # Debug: log reliability scores for this iteration
                scores = gdf_result['reliability_score'].values
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    logger.info(f"Iteration {window['iteration']} reliability scores: "
                              f"min={valid_scores.min():.1f}, max={valid_scores.max():.1f}, "
                              f"mean={valid_scores.mean():.1f}, median={np.median(valid_scores):.1f}")
                else:
                    logger.info(f"Iteration {window['iteration']} reliability scores: all NaN (no active territories with data)")

                # Log distribution
                high_count = (gdf_result['reliability_category'] == 'HIGH').sum()
                mod_count = (gdf_result['reliability_category'] == 'MODERATE').sum()
                low_count = (gdf_result['reliability_category'] == 'LOW').sum()
                logger.info(f"Reliability distribution: HIGH={high_count}, MODERATE={mod_count}, LOW={low_count}")
            else:
                logger.warning(f"No diagnostics for iteration {window['iteration']} - cannot calculate reliability scores")

            # Generate map for this iteration
            iteration_num = window['iteration']
            period_str = f"{start.strftime('%Y%m')}-{end.strftime('%Y%m')}"

            # record the window in the skipped table when the posterior
            # is unhealthy. plot_map will draw the boundary-only UNRELIABLE
            # map via its own gate; the table is the textual counterpart.
            if diagnostics and diagnostics.get('convergence_fatal', False):
                self._iterative_skipped_windows.append({
                    'iteration': window['iteration'],
                    'analysis_start': window['analysis_start'],
                    'analysis_end': window['analysis_end'],
                    'pct_divergences': diagnostics.get('pct_divergences'),
                    'rhat_max': diagnostics.get('rhat_max'),
                    'ess_min': diagnostics.get('ess_alpha_min',
                                               diagnostics.get('min_ess_bulk')),
                    'reason': 'convergence_fatal',
                })

            # Use the same plotting logic as standard analysis
            # plot_map signature: (gdf, level_name, start, end, b_start, b_end, model_name, diagnostics)
            if gdf_result is not None and 'classification' in gdf_result.columns:
                bayesian.plot_map(gdf_result, level_name, start, end, b_start, b_end,
                                  model_name='Bayesian', diagnostics=diagnostics)
            else:
                logger.warning(f"Iteration {window['iteration']}: skipping map — no classification column")

            return gdf_result

        except Exception as e:
            logger.error(f"Error running Bayesian model for window: {e}")
            logger.error(traceback.format_exc())
            return None

        finally:
            # Restore original config
            if original_period:
                self.config['analysis_period'] = original_period

    def _generate_iterative_hotspots_report(self, all_hotspots: List[gpd.GeoDataFrame],
                                             output_dir: Path, level=None):
        """Thin wrapper around :func:`pipeline.orchestration.generate_iterative_hotspots_report`."""
        _orch_generate_iterative_hotspots_report(
            all_hotspots, output_dir, self.config, self.run_timestamp, level=level,
        )

    def _calibrate_sigma_multipliers(self, gdf: pd.DataFrame, level_name: str,
                                      national_rate: float, national_se: float,
                                      start, end, b_start, b_end,
                                      output_dir: Path) -> Dict[str, float]:
        """Thin wrapper around :func:`pipeline.orchestration.calibrate_sigma_multipliers`."""
        optimal = _orch_calibrate_sigma_multipliers(
            gdf, level_name, national_rate, national_se,
            start, end, b_start, b_end, output_dir,
            random_seed=self.config.get('random_seed', 42),
        )
        self._optimal_sigma_multipliers = optimal
        return optimal

    def _calibrate_min_tests(self, gdf: pd.DataFrame, output_dir: Path) -> int:
        """Thin wrapper around :func:`pipeline.orchestration.calibrate_min_tests`."""
        optimal_min = _orch_calibrate_min_tests(gdf, output_dir)
        self._optimal_min_tests = optimal_min
        return optimal_min

    # _calibrate_reliability_weights was the calibration step for the
    # legacy 40/30/30 reliability-weighting scheme. With the CV-based
    # reliability (ReliabilityScoreCalculator.calculate_territory_scores)
    # there are no weights to calibrate, so the function was removed.

    def generate_model_comparison(self):
        """Thin wrapper around :func:`pipeline.orchestration.generate_model_comparison`."""
        comparison_path = self.get_output_path(
            "summary", "All_Levels",
            f'Model_Comparison_{self.period_str}.xlsx',
            is_hex=False,
        )
        _orch_generate_model_comparison(self.model_comparison_data, comparison_path)

    def generate_recommendations(self):
        """Thin wrapper around :func:`pipeline.orchestration.generate_recommendations`."""
        output_dir = Path(__file__).resolve().parent.parent / self.config['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_path = self.get_output_path(
            "summary", "Recommendations",
            f'RECOMMENDATIONS_{self.period_str}.txt',
            is_hex=False,
        )
        _orch_generate_recommendations(self.config, output_dir, rec_path, _is_hotspot)

# =============================================================================
# MAIN
