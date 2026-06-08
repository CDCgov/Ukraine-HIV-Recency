"""
Pre-flight data quality checks.

The :class:`DataQualityChecker` runs before any model is fitted and
produces a structured report with warnings, recommendations and a final
``proceed`` flag. Two routines:

* :meth:`check_data_quality` -- pure function over the territory frame;
  flags low-test territories, zero-event territories, extreme outliers
  (3*IQR on the empirical rate) and small total sample size.
* :meth:`print_quality_report` -- formats the result for the run log.

The thresholds in here are intentionally not pulled from
``ANALYSIS_CONSTANTS``: this checker is the very first thing that runs
on a new dataset, and the warnings are tuned to be loud rather than
quietly configurable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Pre-flight data quality checks before running models."""

    @staticmethod
    def check_data_quality(df: pd.DataFrame, min_tests: int = 3) -> Dict[str, Any]:
        """Report quality metrics and emit warnings / recommendations.

        Returns a dict with ``total_territories``, ``total_tests``,
        ``total_recent``, ``warnings``, ``recommendations`` and a final
        ``proceed`` flag. Additional keys (``n_low_test_territories``,
        ``n_zero_variance``, ``n_outliers``) are present only when those
        problems are detected.
        """
        results: Dict[str, Any] = {
            'total_territories': len(df),
            'total_tests': df['all_tested_curr'].sum(),
            'total_recent': df['recent_count_curr'].sum(),
            'warnings': [],
            'recommendations': [],
            'proceed': True,
        }

        low_test_territories = df[df['all_tested_curr'] < min_tests]
        n_low = len(low_test_territories)
        pct_low = n_low / len(df) * 100 if len(df) else 0.0

        if n_low > 0:
            results['n_low_test_territories'] = n_low
            results['pct_low_test_territories'] = pct_low
            results['warnings'].append(
                f"[WARN] {n_low} territories ({pct_low:.1f}%) have < {min_tests} tests"
            )

            if pct_low > 30:
                results['recommendations'].append("[FAIL] CRITICAL: >30% territories have insufficient data")
                results['recommendations'].append("   → Aggregate to higher administrative level")
                results['proceed'] = False
            elif pct_low > 15:
                results['recommendations'].append("[WARN] WARNING: >15% territories have insufficient data")
                results['recommendations'].append("   → Consider aggregating or using Bayesian model")

        zero_var = df[df['recent_count_curr'] == 0]
        n_zero = len(zero_var)
        pct_zero = n_zero / len(df) * 100 if len(df) else 0.0

        if n_zero > 0:
            results['n_zero_variance'] = n_zero
            results['pct_zero_variance'] = pct_zero
            results['warnings'].append(f"[WARN] {n_zero} territories ({pct_zero:.1f}%) have 0 recent infections")

            if pct_zero > 50:
                results['recommendations'].append("[WARN] Many territories with zero events - Bayesian model recommended")

        n_outliers = 0
        pct_outliers = 0

        if df['recent_count_curr'].max() > 0:
            rates = df['recent_count_curr'] / df['all_tested_curr']
            q75 = rates.quantile(0.75)
            q25 = rates.quantile(0.25)
            iqr = q75 - q25
            upper_fence = q75 + 3 * iqr

            outliers = df[rates > upper_fence]
            n_outliers = len(outliers)
            pct_outliers = n_outliers / len(df) * 100 if len(df) else 0.0

            if n_outliers > 0:
                results['n_outliers'] = n_outliers
                results['pct_outliers'] = pct_outliers
                results['warnings'].append(f"[WARN] {n_outliers} territories ({pct_outliers:.1f}%) are extreme outliers")

                if pct_outliers > 10:
                    results['recommendations'].append("[WARN] Many outliers detected - verify data quality")

        total_tests = df['all_tested_curr'].sum()
        total_recent = df['recent_count_curr'].sum()

        if total_tests < 100:
            results['warnings'].append(f"[WARN] Very small sample size: {total_tests} tests total")
            results['recommendations'].append("[FAIL] CRITICAL: Insufficient data for reliable analysis")
            results['proceed'] = False
        elif total_recent < 5:
            results['warnings'].append(f"[WARN] Very few events: {total_recent} recent infections total")
            results['recommendations'].append("[WARN] WARNING: Few events - results may be unstable")

        if not results['proceed']:
            results['recommended_model'] = 'INSUFFICIENT_DATA'
        elif pct_low > 15 or pct_zero > 30:
            results['recommended_model'] = 'Bayesian Hierarchical'
        elif n_outliers > len(df) * 0.1:
            results['recommended_model'] = 'Bayesian Hierarchical'
        else:
            results['recommended_model'] = 'Bayesian Hierarchical'

        return results

    @staticmethod
    def print_quality_report(results: Dict[str, Any]) -> None:
        """Pretty-print the quality report to the run log."""
        logger.info("\n" + "=" * 80)
        logger.info("DATA QUALITY PRE-CHECK")
        logger.info("=" * 80)
        logger.info(f"\nTerritories: {results['total_territories']}")
        logger.info(f"Total tests: {results['total_tests']:,}")
        logger.info(f"Recent infections: {results['total_recent']}")

        if results['warnings']:
            logger.info("\n[WARN]  WARNINGS:")
            for warning in results['warnings']:
                logger.info(f"  {warning}")

        if results['recommendations']:
            logger.info("\nRECOMMENDATIONS:")
            for rec in results['recommendations']:
                logger.info(f"  {rec}")

        logger.info(f"\nRecommended model: {results['recommended_model']}")
        logger.info(f"[OK] Proceed with analysis: {'YES' if results['proceed'] else 'NO'}")
        logger.info("=" * 80 + "\n")
