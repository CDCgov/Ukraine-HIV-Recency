"""
Human-readable rendering of the data-specification analysis.

The :class:`ModelSpecificationAnalyzer` formats the dict produced by
:class:`~pipeline.spec.auto.AutoSpecificationSystem.recommend_specification`
into the four-block prose report (zero inflation, overdispersion, sample
sizes, outliers) that goes into the run log and the analysis appendix.
"""

from __future__ import annotations

from typing import Any, Dict

from pipeline.constants import (
    ZERO_INFLATION_THRESHOLD_HIGH,
    ZERO_INFLATION_THRESHOLD_MODERATE,
)


class ModelSpecificationAnalyzer:
    """String formatter for the data-specification analysis."""

    @staticmethod
    def generate_specification_report(data_analysis: Dict[str, Any]) -> str:
        """Format the spec analysis as a four-block plaintext report."""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA SPECIFICATION ANALYSIS")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Territories: {data_analysis['n_territories']}")
        lines.append(f"Total events: {data_analysis['total_events']:,}")
        lines.append(f"Total observations: {data_analysis['total_n']:,}")
        lines.append(f"Overall rate: {data_analysis['overall_rate']:.4f} ({data_analysis['overall_rate']*100:.2f}%)")
        lines.append("")

        lines.append("ZERO INFLATION:")
        lines.append(f"  Territories with zero events: {data_analysis['n_zeros']} ({data_analysis['pct_zeros']:.1f}%)")
        if data_analysis['pct_zeros'] > ZERO_INFLATION_THRESHOLD_HIGH:
            lines.append("  [WARN]  HIGH zero inflation - Bayesian model recommended")
        elif data_analysis['pct_zeros'] > ZERO_INFLATION_THRESHOLD_MODERATE:
            lines.append("  [WARN]  MODERATE zero inflation")
        else:
            lines.append("  [OK] Low zero inflation")
        lines.append("")

        lines.append("OVERDISPERSION:")
        if 'loo_ic_diff' in data_analysis:
            lines.append(f"  Test method: {data_analysis['overdispersion_test']}")
            lines.append(f"  LOO-IC difference (Binomial - BetaBinomial): {data_analysis['loo_ic_diff']:.2f}")
            if data_analysis['overdispersion_detected']:
                lines.append("  [WARN]  Overdispersion detected - Beta-Binomial preferred, Bayesian recommended")
            else:
                lines.append("  [OK] No significant overdispersion - Binomial acceptable")
        else:
            lines.append(f"  Test method: {data_analysis.get('overdispersion_test', 'Variance heuristic')}")
            if data_analysis.get('overdispersion_detected', False):
                lines.append("  [WARN]  Overdispersion detected - Bayesian recommended")
            else:
                lines.append("  [OK] No significant overdispersion")
        lines.append("")

        lines.append("SAMPLE SIZES:")
        lines.append(f"  Territories with n < 30: {data_analysis['small_n']} ({data_analysis['pct_small_n']:.1f}%)")
        if data_analysis['pct_small_n'] > 50:
            lines.append("  [WARN]  Many small samples - Bayesian model recommended")
        elif data_analysis['pct_small_n'] > 30:
            lines.append("  [WARN]  Some small samples")
        else:
            lines.append("  [OK] Adequate sample sizes")
        lines.append("")

        lines.append("OUTLIERS:")
        lines.append(f"  Extreme outliers: {data_analysis['outliers']} ({data_analysis['pct_outliers']:.1f}%)")
        if data_analysis['pct_outliers'] > 10:
            lines.append("  [WARN]  Many outliers - Bayesian model more robust")
        elif data_analysis['pct_outliers'] > 5:
            lines.append("  [WARN]  Some outliers present")
        else:
            lines.append("  [OK] Few outliers")
        lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)
