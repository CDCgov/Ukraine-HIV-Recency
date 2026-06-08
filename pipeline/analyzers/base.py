"""
:class:`BaseHotspotAnalyzer` -- the shared facade used by both the
standard Bayesian and the covariates analysers.

Almost every public method here is a thin delegation to a
``pipeline.*`` helper; the class exists so existing call sites
(``analyzer.load_geodata(...)``, ``analyzer.save_report(...)``,
``analyzer.plot_map(...)`` and so on) share a single implementation.
The two pieces of real local state
are the IO caches (``_cached_cases`` / ``_cached_geodata``) with
file-mtime invalidation and the per-instance ``diagnostics`` log
that downstream code appends to after each fit.

Subclass identity is exposed via the ``MODEL_TYPE`` class
attribute (``'bayesian'`` / ``'bayesian_covariates'``) so this
module can route to the right output subtree without taking an
import on the subclasses (avoids the circular import that
``isinstance(self, BayesianAnalyzer)`` would force once the
classes live in their own modules).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.aggregation import (
    aggregate_covariates as _aggregate_covariates,
    aggregate_stats as _aggregate_stats,
    aggregate_stats_hard_stratified as _aggregate_stats_hard_stratified,
    aggregate_stats_stratified as _aggregate_stats_stratified,
    analyze_network_change as _analyze_network_change,
    analyze_site_profile as _analyze_site_profile,
    calculate_national_baseline as _calculate_national_baseline,
    calculate_testing_intensity as _calculate_testing_intensity,
    classify_network_stability as _classify_network_stability,
    detect_outbreak_and_artifact as _detect_outbreak_and_artifact,
    ensure_crs_match as _ensure_crs_match,
    generate_network_explanation as _generate_network_explanation,
    get_periods as _get_periods,
    soft_fallback_result as _soft_fallback_result,
)
from pipeline.classification import (
    classify_with_exceedance as _classify_with_exceedance,
    classify_with_smr_sir as _classify_with_smr_sir,
    add_watchlist as _add_watchlist,
)
from pipeline.diagnostics import ReliabilityScoreCalculator
from pipeline.io import (
    load_cases_from_disk as _load_cases_from_disk,
    load_geodata_from_disk as _load_geodata_from_disk,
    load_testing_sites as _load_testing_sites,
)
from pipeline.reporting import (
    add_admin_territory_info as _add_admin_territory_info_fn,
    add_disclaimer_to_metadata as _add_disclaimer_to_metadata,
    add_hex_territory_info as _add_hex_territory_info_fn,
    add_oblast_labels as _add_oblast_labels,
    get_facility_based_disclaimer as _get_facility_based_disclaimer,
    print_summary as _print_summary,
    render_anomaly_map as _render_anomaly_map,
    render_boundary_only_map as _render_boundary_only_map,
    render_reliability_map as _render_reliability_map_fn,
    render_watchlist_map as _render_watchlist_map_fn,
    save_diagnostics as _save_diagnostics,
    write_report as _write_report,
)
from pipeline.standardization import bayesian_fdr_threshold
from pipeline.standardization.smr_sir import compute_smr_sir, eb_baseline_rate
from pipeline.standardization.z_scores import calculate_z_scores as _calculate_z_scores

logger = logging.getLogger(__name__)


class BaseHotspotAnalyzer:
    """Base class with common functionality for all analyzers."""

    LEVEL_NAMES = {
        'Community': 'Community',
        'District': 'District',
        'Oblast': 'Oblast',
        3: 'Hex_Res3',
        4: 'Hex_Res4'
    }

    # Subclasses set this so get_output_path() can route to the right
    # output subtree without importing the subclasses (avoids a circular
    # import once each analyzer lives in its own module).
    MODEL_TYPE: Optional[str] = None

    def __init__(self, config: Dict[str, Any], mode_suffix: str = 'admin', orchestrator=None):
        """Store config and set up the per-instance IO caches.

        Args:
            config: the resolved pipeline config dict.
            mode_suffix: ``'admin'`` or ``'hex'`` — selects which geometry
                block and output subtree the analyzer uses.
            orchestrator: optional back-reference so the analyzer can reach
                the shared output-path policy and audit trail; ``None`` in
                standalone use (e.g. tests, calibration).
        """
        self.cfg = config
        self.mode_suffix = mode_suffix
        self.orchestrator = orchestrator
        self.national_baseline_rate = None
        self.national_baseline_se = None
        self.diagnostics = []

        # Cache for loaded data (IO optimization)
        self._cached_cases = None
        self._cached_geodata = {}
        self._testing_sites = None  # testing sites with activation/deactivation dates

        # Base output directory.
        # NOTE: __file__ here points at pipeline/analyzers/base.py — the
        # repo-root resolution that was previously done by the in-script
        # version is preserved by walking up two levels to the script root.
        self.base_out_dir = Path(__file__).resolve().parent.parent.parent / config['output_dir']

        # NOTE: Directory creation is now handled by PipelineOrchestrator.get_output_path()
        # Legacy paths kept for backward compatibility when orchestrator is None
        if orchestrator is None:
            # Only create directories if running standalone (no orchestrator)
            self.bayesian_out_dir = self.base_out_dir / 'bayesian' / mode_suffix
            self.bayesian_cov_out_dir = self.base_out_dir / 'bayesian_covariates' / mode_suffix

            self.bayesian_out_dir.mkdir(parents=True, exist_ok=True)
            self.bayesian_cov_out_dir.mkdir(parents=True, exist_ok=True)
        else:
            # When orchestrator exists, paths are managed dynamically
            # No need to create directories upfront
            self.bayesian_out_dir = None
            self.bayesian_cov_out_dir = None

    @staticmethod
    def _ensure_crs_match(gdf_left: gpd.GeoDataFrame, gdf_right: gpd.GeoDataFrame,
                          operation: str = "spatial join") -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Thin wrapper around :func:`pipeline.aggregation.ensure_crs_match`."""
        return _ensure_crs_match(gdf_left, gdf_right, operation=operation)

    def get_output_dir(self) -> Path:
        """Get the correct output directory based on analyzer type."""
        # Only used when orchestrator is None (backward compatibility)
        if self.orchestrator:
            raise RuntimeError("get_output_dir() should not be called when orchestrator exists. Use get_output_path() instead.")

        if self.MODEL_TYPE == "bayesian_covariates":
            return self.bayesian_cov_out_dir
        if self.MODEL_TYPE == "bayesian":
            return self.bayesian_out_dir
        return None

    def get_output_path(self, level_name: str, filename: str) -> Path:
        """Get output path using orchestrator's new structure if available."""
        if self.orchestrator:
            model_type = self.MODEL_TYPE or "bayesian"
            is_hex = level_name.startswith("Hex_Res")
            return self.orchestrator.get_output_path(model_type, level_name, filename, is_hex=is_hex)
        # Fallback to old structure
        return self.get_output_dir() / filename

    @staticmethod
    def get_facility_based_disclaimer() -> Dict[str, str]:
        """Thin wrapper around :func:`pipeline.reporting.get_facility_based_disclaimer`."""
        return _get_facility_based_disclaimer()

    def add_disclaimer_to_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.reporting.add_disclaimer_to_metadata`."""
        return _add_disclaimer_to_metadata(metadata)

    def _validate_path(self, path_str: str, label: str) -> Path:
        """Validate that a file path exists, resolving relative paths from script root."""
        p = Path(path_str)
        # Resolve relative paths from the script root (two levels up from this module)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / p
        if not p.exists():
            raise FileNotFoundError(f"{label}: Path {p} does not exist")
        if p.is_dir():
            raise IsADirectoryError(f"{label}: Expected a file, got directory: {p}")
        return p

    def get_periods(self, extend_period: bool = False) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """Thin wrapper around :func:`pipeline.aggregation.get_periods`."""
        return _get_periods(self.cfg, extend_period=extend_period)

    def load_geodata(self, level: str, level_key: str = None, use_cache: bool = True) -> gpd.GeoDataFrame:
        """Load geographic data for specified level with caching support."""
        if level_key is None:
            level_key = level

        cache_key = str(level)

        cache_enabled = self.cfg.get('io_optimization', {}).get('cache_geodata', True)
        if use_cache and cache_enabled and cache_key in self._cached_geodata:
            cached_entry = self._cached_geodata[cache_key]
            if isinstance(cached_entry, dict) and 'path' in cached_entry:
                cached_mtime = cached_entry['loaded_at']
                current_mtime = os.path.getmtime(cached_entry['path'])
                if current_mtime > cached_mtime:
                    logger.info(f"Cache invalidated for {cache_key} - file modified (cached: {cached_mtime}, current: {current_mtime})")
                    del self._cached_geodata[cache_key]
                else:
                    logger.debug(f"Using cached geodata for {cache_key}")
                    return cached_entry['data'].copy()
            else:
                logger.debug(f"Using cached geodata for {cache_key} (old format, no timestamp)")
                return cached_entry.copy()

        use_parquet = self.cfg.get('io_optimization', {}).get('use_parquet', False)

        if isinstance(level, int):
            hex_paths = self.cfg['h3_hexagons']
            path_key = f'res{level}_path'
            if path_key not in hex_paths:
                raise ValueError(f"H3 resolution {level} not configured")
            path = self._validate_path(hex_paths[path_key], f"H3 Res{level}")
        else:
            admin_paths = self.cfg['administrative_units']
            if level == 'Community':
                path = self._validate_path(admin_paths['adm3_path'], 'Community')
            elif level == 'District':
                path = self._validate_path(admin_paths['adm2_path'], 'District')
            else:  # Oblast
                path = self._validate_path(admin_paths['adm1_path'], 'Oblast')

        gdf = _load_geodata_from_disk(self.cfg, path, level, use_parquet=use_parquet)

        if use_cache and cache_enabled:
            self._cached_geodata[cache_key] = {
                'data': gdf.copy(),
                'loaded_at': os.path.getmtime(path),
                'path': path,
            }
            logger.debug(f"Cached geodata for {cache_key} (mtime: {os.path.getmtime(path)})")

        return gdf

    def load_cases(self, use_cache: bool = True) -> gpd.GeoDataFrame:
        """Load HIV case data from Excel with caching support."""
        if use_cache and self._cached_cases is not None:
            if isinstance(self._cached_cases, dict) and 'path' in self._cached_cases:
                cached_mtime = self._cached_cases['loaded_at']
                current_mtime = os.path.getmtime(self._cached_cases['path'])
                if current_mtime > cached_mtime:
                    logger.info(f"Cache invalidated for cases - file modified (cached: {cached_mtime}, current: {current_mtime})")
                    self._cached_cases = None
                else:
                    logger.debug("Using cached case data")
                    return self._cached_cases['data']
            else:
                logger.debug("Using cached case data (old format, no timestamp)")
                return self._cached_cases

        excel_path = self._validate_path(self.cfg['excel_path'], 'Excel Data')
        use_parquet = self.cfg.get('io_optimization', {}).get('use_parquet', False)

        gdf_cases = _load_cases_from_disk(excel_path, self.cfg['target_crs'],
                                          use_parquet=use_parquet)

        if use_cache:
            self._cached_cases = {
                'data': gdf_cases,
                'loaded_at': os.path.getmtime(excel_path),
                'path': excel_path,
            }
            logger.debug(f"Cached case data for future use (mtime: {os.path.getmtime(excel_path)})")

        return gdf_cases

    def load_testing_sites(self, excel_path: str) -> pd.DataFrame:
        """Thin wrapper around :func:`pipeline.io.load_testing_sites`."""
        return _load_testing_sites(excel_path)

    def analyze_site_profile(self, site_id: str, gdf_cases: gpd.GeoDataFrame, b_start: pd.Timestamp, b_end: pd.Timestamp) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.aggregation.analyze_site_profile`."""
        return _analyze_site_profile(site_id, gdf_cases, b_start, b_end)

    def calculate_testing_intensity(self, gdf_cases: gpd.GeoDataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.aggregation.calculate_testing_intensity`."""
        return _calculate_testing_intensity(gdf_cases, start_date, end_date)

    def classify_network_stability(self, intensity_curr: float, intensity_hist: float, all_intensities_curr: np.ndarray, all_intensities_hist: np.ndarray) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.aggregation.classify_network_stability`."""
        return _classify_network_stability(intensity_curr, intensity_hist, all_intensities_curr, all_intensities_hist)

    def analyze_network_change(self, territory_idx: int, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame, df_sites: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.aggregation.analyze_network_change`."""
        return _analyze_network_change(territory_idx, gdf_admin, gdf_cases, df_sites, start, end, b_start, b_end)

    def generate_network_explanation(self, network_analysis: Dict[str, Any], stability: Dict[str, Any]) -> str:
        """Thin wrapper around :func:`pipeline.aggregation.generate_network_explanation`."""
        return _generate_network_explanation(network_analysis, stability)

    def calculate_national_baseline(self, gdf_cases: gpd.GeoDataFrame, b_start: pd.Timestamp, b_end: pd.Timestamp) -> Tuple[float, float]:
        """Thin wrapper around :func:`pipeline.aggregation.calculate_national_baseline`."""
        rate, se = _calculate_national_baseline(self.cfg, gdf_cases, b_start, b_end)
        self.national_baseline_rate = rate
        self.national_baseline_se = se
        return rate, se

    @staticmethod
    def _eb_baseline_rate(recent_hist, tested_hist, national_rate_baseline: float):
        """Thin wrapper around :func:`pipeline.standardization.smr_sir.eb_baseline_rate`."""
        return eb_baseline_rate(recent_hist, tested_hist, national_rate_baseline)

    @staticmethod
    def _compute_smr_sir(p_samples, df: pd.DataFrame,
                         national_rate_baseline: float,
                         smr_threshold: float = 2.0,
                         sir_threshold: float = 1.5,
                         smr_low_threshold: float = 0.5,
                         sir_low_threshold: float = 1.0 / 1.5,
                         national_rate_curr_floor: float = 1e-3):
        """Thin wrapper around :func:`pipeline.standardization.smr_sir.compute_smr_sir`."""
        return compute_smr_sir(
            p_samples=p_samples,
            df=df,
            national_rate_baseline=national_rate_baseline,
            smr_threshold=smr_threshold,
            sir_threshold=sir_threshold,
            smr_low_threshold=smr_low_threshold,
            sir_low_threshold=sir_low_threshold,
            national_rate_curr_floor=national_rate_curr_floor,
        )

    def aggregate_stats(self, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame,
                       start: pd.Timestamp, end: pd.Timestamp,
                       b_start: pd.Timestamp, b_end: pd.Timestamp) -> gpd.GeoDataFrame:
        """Thin wrapper around :func:`pipeline.aggregation.aggregate_stats`."""
        if self._testing_sites is None:
            self._testing_sites = _load_testing_sites(self.cfg['excel_path'])
        return _aggregate_stats(self.cfg, self._testing_sites, gdf_admin,
                                gdf_cases, start, end, b_start, b_end)

    def aggregate_covariates(self, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame, start: pd.Timestamp, end: pd.Timestamp) -> gpd.GeoDataFrame:
        """Thin wrapper around :func:`pipeline.aggregation.aggregate_covariates`."""
        return _aggregate_covariates(gdf_admin, gdf_cases, start, end)

    def aggregate_stats_stratified(self, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame, start: pd.Timestamp, end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> pd.DataFrame:
        """Thin wrapper around :func:`pipeline.aggregation.aggregate_stats_stratified`."""
        return _aggregate_stats_stratified(gdf_admin, gdf_cases, start, end, b_start, b_end)

    def aggregate_stats_hard_stratified(self, gdf_admin: gpd.GeoDataFrame, gdf_cases: gpd.GeoDataFrame, start: pd.Timestamp, end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> pd.DataFrame:
        """Thin wrapper around :func:`pipeline.aggregation.aggregate_stats_hard_stratified`."""
        return _aggregate_stats_hard_stratified(gdf_admin, gdf_cases, start, end, b_start, b_end)

    def detect_outbreak_and_artifact(self, territory_idx: int, df_hard: pd.DataFrame, national_rate: float) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.aggregation.detect_outbreak_and_artifact`."""
        return _detect_outbreak_and_artifact(territory_idx, df_hard, national_rate)

    def _get_soft_fallback_result(self) -> Dict[str, Any]:
        """Thin wrapper around :func:`pipeline.aggregation.soft_fallback_result`."""
        return _soft_fallback_result()

    def calculate_z_scores(self, df: pd.DataFrame, national_rate: float) -> pd.DataFrame:
        """Thin wrapper around :func:`pipeline.standardization.calculate_z_scores`."""
        return _calculate_z_scores(df, national_rate)

    def classify_with_exceedance(self, row: pd.Series, threshold: float = 0.95) -> str:
        """Thin wrapper around :func:`pipeline.classification.classify_with_exceedance`."""
        return _classify_with_exceedance(row, threshold=threshold)

    @staticmethod
    def classify_with_smr_sir(row: pd.Series,
                              cutoff_smr_high: float,
                              cutoff_sir_high: float,
                              cutoff_smr_low: float,
                              cutoff_sir_low: float) -> str:
        """Thin wrapper around :func:`pipeline.classification.classify_with_smr_sir`."""
        return _classify_with_smr_sir(
            row,
            cutoff_smr_high=cutoff_smr_high,
            cutoff_sir_high=cutoff_sir_high,
            cutoff_smr_low=cutoff_smr_low,
            cutoff_sir_low=cutoff_sir_low,
        )

    @staticmethod
    def _auto_threshold(exc_probs: np.ndarray, max_fdr: float = 0.05,
                        start: float = 0.95, step: float = 0.005,
                        ceiling: float = 0.99, floor: float = 0.70) -> Tuple[float, float]:
        """Thin wrapper around :func:`pipeline.standardization.thresholds.bayesian_fdr_threshold`."""
        return bayesian_fdr_threshold(
            exc_probs, max_fdr=max_fdr, start=start, step=step,
            ceiling=ceiling, floor=floor,
        )

    def _finalize_classification(self, df: pd.DataFrame, national_rate: float) -> pd.DataFrame:
        """Shared deterministic post-fit step (audit M2).

        Given a per-territory frame that already carries ``exceedance_prob``
        and the four SMR/SIR exceedance columns, this computes the
        FDR-controlled cut-offs, the two-axis ``classification_smr_sir`` label
        (and its alias ``classification``), the new-site flag, the national
        baseline and the percent deviation. It was copied verbatim into the
        crude, hurdle and covariates fits; centralising it removes that
        duplication. It is pure post-processing -- no sampling, no RNG -- so
        the numbers are unchanged by where it lives.

        ``calculate_z_scores`` is intentionally left to the caller, since the
        three fits compute it at slightly different points; it is independent
        of everything here.
        """
        # FDR threshold on the single-axis exceedance. The SMR/SIR taxonomy
        # drives classification now (audit M4); this stays only as an
        # informative log of the discovery budget on the "above national" axis.
        exc_arr = df['exceedance_prob'].dropna().values
        auto_threshold, bayesian_fdr = self._auto_threshold(exc_arr)
        n_above = int((exc_arr > auto_threshold).sum())
        _moved = ("raised from 0.95" if auto_threshold > 0.95
                  else "relaxed from 0.95" if auto_threshold < 0.95 else "unchanged")
        logger.info(f"Bayesian FDR threshold: {auto_threshold:.3f} ({_moved}), "
                    f"FDR={bayesian_fdr:.1%}, {n_above} territories above threshold")

        # Each axis gets its own FDR-controlled cut-off so a call is made only
        # when the posterior evidence is strong on that specific dimension.
        cutoff_smr_high, _ = self._auto_threshold(df['exc_prob_smr'].dropna().values)
        cutoff_sir_high, _ = self._auto_threshold(df['exc_prob_sir'].dropna().values)
        cutoff_smr_low, _ = self._auto_threshold(df['exc_prob_smr_low'].dropna().values)
        cutoff_sir_low, _ = self._auto_threshold(df['exc_prob_sir_low'].dropna().values)
        df['classification_smr_sir'] = df.apply(
            lambda row: self.classify_with_smr_sir(
                row,
                cutoff_smr_high=cutoff_smr_high,
                cutoff_sir_high=cutoff_sir_high,
                cutoff_smr_low=cutoff_smr_low,
                cutoff_sir_low=cutoff_sir_low,
            ), axis=1)
        # Single label set: the legacy single-axis classifier is retired
        # (audit M4); `classification` aliases the taxonomy.
        df['classification'] = df['classification_smr_sir']
        # New sites (no historical testing) have an undefined trend axis; the
        # map marks them with an open circle rather than a colour class.
        df['is_new_site'] = (df['all_tested_hist'].fillna(0) == 0)
        logger.info(
            f"SIR/SMR cutoffs (FDR-controlled): smr_high={cutoff_smr_high:.3f}, "
            f"sir_high={cutoff_sir_high:.3f}, smr_low={cutoff_smr_low:.3f}, "
            f"sir_low={cutoff_sir_low:.3f}"
        )
        logger.info(
            "SIR/SMR taxonomy distribution: "
            f"{df['classification_smr_sir'].value_counts().to_dict()}"
        )

        df['national_baseline'] = national_rate
        # Percent deviation from the *current* national rate via the posterior
        # SMR (SMR = p / national_rate_curr, so SMR - 1 is the deviation).
        df['deviation_pct'] = (df['smr_mean'] - 1.0) * 100
        # Combined burden + rate watch-list (additive). The rate axis alone is
        # underpowered on sparse recency counts, so surface high-burden centres
        # (many recent infections, near-average rate) alongside a relative
        # top-percentile rate flag. Thresholds come from the optional
        # ``watchlist`` config block. See pipeline.classification.add_watchlist.
        _wcfg = (self.cfg or {}).get('watchlist', {}) if isinstance(self.cfg, dict) else {}
        df = _add_watchlist(
            df,
            top_burden_frac=float(_wcfg.get('burden_top_frac', 0.80)),
            rate_pctile=float(_wcfg.get('rate_percentile', 0.80)),
        )
        return df

    def save_diagnostics(self, level_name: str, period_str: str):
        """Thin wrapper around :func:`pipeline.reporting.save_diagnostics`."""
        if not self.diagnostics:
            return
        diag_path = self.get_output_path(level_name, f"Diagnostics_{period_str}.xlsx")
        _save_diagnostics(self.diagnostics, diag_path)

    def save_report(self, gdf_admin: gpd.GeoDataFrame, level_name: str, period_str: str,
                    diagnostics: Dict[str, Any] = None):
        """Save detailed report with parent territories and Ukrainian names.

        Skip the Excel write when the posterior is unhealthy
        (``convergence_fatal``) and drop a ``_SKIPPED.txt`` sentinel
        instead so the failure is recorded next to where the report
        would have been.
        """
        if diagnostics and diagnostics.get('convergence_fatal', False):
            sentinel_path = self.get_output_path(
                level_name, f"Report_{level_name}_{period_str}_SKIPPED.txt",
            )
            try:
                pct_div = diagnostics.get('pct_divergences')
                rhat_max = diagnostics.get('rhat_max')
                ess_min = diagnostics.get('ess_alpha_min', diagnostics.get('min_ess_bulk'))
                lines = [
                    "REPORT SKIPPED -- FATAL CONVERGENCE FAILURE",
                    "=" * 60,
                    f"Level: {level_name}",
                    f"Period: {period_str}",
                    f"Divergences: {pct_div}",
                    f"R-hat max: {rhat_max}",
                    f"Minimum ESS: {ess_min}",
                    "",
                    "Posterior geometry is unhealthy (convergence_fatal=True).",
                    "Per-territory SMR/SIR/exceedance values are not reliable",
                    "and the Excel report has been intentionally not written.",
                    "See Diagnostics_<level>_<period>.xlsx for the full failure record.",
                ]
                with open(sentinel_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            except (IOError, OSError) as e:
                logger.error(f"Could not write skip sentinel ({e})")
            logger.warning(
                f"save_report skipped for {level_name}: convergence_fatal=True"
            )
            return

        logger.debug(f"save_report called for {level_name}")
        logger.debug(f"gdf_admin columns: {list(gdf_admin.columns)}")
        logger.debug(f"high_outbreak in gdf_admin: {'high_outbreak' in gdf_admin.columns}")

        active = gdf_admin[gdf_admin['all_tested_curr'] > 0].copy()
        if active.empty:
            logger.warning(f"No data to save for {level_name}")
            return

        logger.debug(f"active columns after filter: {list(active.columns)}")
        logger.debug(f"high_outbreak in active: {'high_outbreak' in active.columns}")

        is_hex = 'Hex' in level_name

        if is_hex:
            active = self._add_hex_territory_info(active, level_name)
        else:
            active = self._add_admin_territory_info(active, level_name)

        if diagnostics:
            active = ReliabilityScoreCalculator.calculate_territory_scores(active, diagnostics, self.cfg)

            active['reliability_flag'] = active['reliability_category'].map({
                'HIGH': '[OK]',
                'MODERATE': '[WARN]',
                'LOW': '[WARN]'
            })

            scores = active['reliability_score'].values
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) > 0:
                logger.info(f"Reliability Scores: min={valid_scores.min():.1f}, max={valid_scores.max():.1f}, mean={valid_scores.mean():.1f}")
            else:
                logger.info("Reliability Scores: all NaN (no active territories with data)")

            high_count = (active['reliability_category'] == 'HIGH').sum()
            mod_count = (active['reliability_category'] == 'MODERATE').sum()
            low_count = (active['reliability_category'] == 'LOW').sum()
            logger.info(f"Distribution: HIGH={high_count}, MODERATE={mod_count}, LOW={low_count}")
        else:
            active['reliability_score'] = np.nan
            active['reliability_category'] = 'Unknown'
            active['reliability_flag'] = ''

        report_path = self.get_output_path(level_name, f"Report_{level_name}_{period_str}.xlsx")
        _write_report(self.cfg, active, level_name, period_str, report_path,
                      self.get_facility_based_disclaimer())

    def _add_admin_territory_info(self, gdf: gpd.GeoDataFrame, level_name: str) -> gpd.GeoDataFrame:
        """Thin wrapper around :func:`pipeline.reporting.add_admin_territory_info`."""
        return _add_admin_territory_info_fn(self.cfg, self.load_geodata, gdf, level_name)

    def _add_hex_territory_info(self, gdf: gpd.GeoDataFrame, level_name: str) -> gpd.GeoDataFrame:
        """Thin wrapper around :func:`pipeline.reporting.add_hex_territory_info`."""
        return _add_hex_territory_info_fn(self.cfg, self.load_geodata, gdf, level_name)

    def _add_oblast_labels(self, ax, gdf_oblast: gpd.GeoDataFrame, lang: str = 'en') -> None:
        """Thin wrapper around :func:`pipeline.reporting.add_oblast_labels`."""
        _add_oblast_labels(self.cfg, ax, gdf_oblast, lang=lang)

    def plot_map(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                 start: pd.Timestamp, end: pd.Timestamp,
                 b_start: pd.Timestamp = None, b_end: pd.Timestamp = None,
                 model_name: str = "Bayesian",
                 diagnostics: Optional[Dict[str, Any]] = None) -> Path:
        """Generate anomaly map with hatching -- saves EN and UA versions.

        When ``diagnostics['convergence_fatal']`` is set, render the EN
        map as oblast boundaries only with a WARN banner so a reader sees
        the failure visually; the UA copy is skipped.
        """
        if diagnostics and diagnostics.get('convergence_fatal', False):
            return self._render_boundary_only_map(level_name, start, end, model_name,
                                                  lang='en', diagnostics=diagnostics)
        en_path = self._render_map(gdf_admin, level_name, start, end,
                                   b_start, b_end, model_name, lang='en')
        self._render_map(gdf_admin, level_name, start, end,
                         b_start, b_end, model_name, lang='ua')
        return en_path

    def _render_boundary_only_map(self, level_name: str,
                                  start: pd.Timestamp, end: pd.Timestamp,
                                  model_name: str, lang: str,
                                  diagnostics: Dict[str, Any]) -> Optional[Path]:
        """Thin wrapper around :func:`pipeline.reporting.render_boundary_only_map`."""
        try:
            gdf_oblast = self.load_geodata('Oblast')
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.error(f"Cannot draw boundary-only map for {level_name}: {e}")
            return None

        lang_suffix = "_UA" if lang == 'ua' else "_EN"
        out_path = self.get_output_path(
            level_name,
            f"Map_{level_name}_{start.strftime('%Y%m')}{lang_suffix}_UNRELIABLE.png"
            if start is not None else f"Map_{level_name}_UNRELIABLE{lang_suffix}.png",
        )
        return _render_boundary_only_map(
            self.cfg, gdf_oblast, _add_oblast_labels,
            out_path, level_name, start, end, model_name, lang, diagnostics,
        )

    def _render_map(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                    start: pd.Timestamp, end: pd.Timestamp,
                    b_start: pd.Timestamp, b_end: pd.Timestamp,
                    model_name: str, lang: str) -> Path:
        """Thin wrapper around :func:`pipeline.reporting.render_anomaly_map`."""
        try:
            gdf_oblast = self.load_geodata('Oblast')
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"Could not load oblast layer for anomaly map: {e}")
            gdf_oblast = None
        lang_suffix = f"_{lang}" if lang != 'en' else "_EN"
        out_path = self.get_output_path(
            level_name,
            f"Map_{level_name}_{start.strftime('%Y%m')}{lang_suffix}.png",
        )
        return _render_anomaly_map(
            self.cfg, getattr(self, 'national_baseline_rate', None),
            gdf_oblast, out_path,
            gdf_admin, level_name, start, end, b_start, b_end, model_name, lang,
        )

    def plot_reliability_map(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                             start: pd.Timestamp, end: pd.Timestamp,
                             model_name: str = "Model") -> Path:
        """Generate reliability score map -- saves EN and UA versions."""
        en_path = self._render_reliability_map(gdf_admin, level_name, start, end, model_name, lang='en')
        self._render_reliability_map(gdf_admin, level_name, start, end, model_name, lang='ua')
        return en_path

    def _render_reliability_map(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                                start: pd.Timestamp, end: pd.Timestamp,
                                model_name: str, lang: str) -> Path:
        """Thin wrapper around :func:`pipeline.reporting.render_reliability_map`."""
        try:
            gdf_oblast = self.load_geodata('Oblast')
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"Could not load oblast layer for reliability map: {e}")
            gdf_oblast = None
        lang_suffix = f"_{lang}" if lang != 'en' else "_EN"
        out_path = self.get_output_path(
            level_name,
            f"Reliability_Map_{level_name}_{start.strftime('%Y%m')}{lang_suffix}.png",
        )
        return _render_reliability_map_fn(
            self.cfg, gdf_oblast, out_path,
            gdf_admin, level_name, start, end, model_name, lang,
        )

    def plot_watchlist_map(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                           start: pd.Timestamp, end: pd.Timestamp,
                           model_name: str = "Model") -> Path:
        """Generate the burden + rate watch-list map -- saves EN and UA versions."""
        en_path = self._render_watchlist_map(gdf_admin, level_name, start, end, model_name, lang='en')
        self._render_watchlist_map(gdf_admin, level_name, start, end, model_name, lang='ua')
        return en_path

    def _render_watchlist_map(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                              start: pd.Timestamp, end: pd.Timestamp,
                              model_name: str, lang: str) -> Path:
        """Thin wrapper around :func:`pipeline.reporting.render_watchlist_map`."""
        try:
            gdf_oblast = self.load_geodata('Oblast')
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.warning(f"Could not load oblast layer for watch-list map: {e}")
            gdf_oblast = None
        lang_suffix = f"_{lang}" if lang != 'en' else "_EN"
        out_path = self.get_output_path(
            level_name,
            f"Watchlist_Map_{level_name}_{start.strftime('%Y%m')}{lang_suffix}.png",
        )
        return _render_watchlist_map_fn(
            self.cfg, gdf_oblast, out_path,
            gdf_admin, level_name, start, end, model_name, lang,
        )

    def print_summary(self, gdf_admin: gpd.GeoDataFrame, level_name: str,
                     model_name: str, converged: bool, period_str: str,
                     rhat_max: float = None):
        """Thin wrapper around :func:`pipeline.reporting.print_summary`."""
        return _print_summary(gdf_admin, level_name, model_name, converged,
                              period_str, rhat_max=rhat_max)
