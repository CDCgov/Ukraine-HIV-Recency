"""
Human-readable interpretation of Bayesian diagnostics.

The :class:`DiagnosticInterpreter` turns the raw diagnostics dict produced
by :class:`~pipeline.diagnostics.reliability.ReliabilityScoreCalculator`
(and the model-fitting machinery) into bulleted English strings that go
into ``RECOMMENDATIONS.txt`` and the comparison report. The logic here is
purely textual and has no PyMC dependency.

Two routines:

* :meth:`interpret_bayesian_diagnostics` -- per-model interpretation
  (one bullet block per quality metric: R-hat, ESS, divergences,
  credible-interval coverage) plus a closing "RECOMMENDATIONS" block.
* :meth:`generate_recommendations_report` -- side-by-side comparison of
  the crude and covariate models with an optional reliability summary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DiagnosticInterpreter:
    """String formatter for Bayesian diagnostics."""

    @staticmethod
    def interpret_bayesian_diagnostics(diagnostics: Dict[str, Any]) -> List[str]:
        """Bullet-style interpretation of one model's diagnostics dict."""
        interpretation: List[str] = []
        quality = diagnostics.get('overall_quality', 'UNKNOWN')
        interpretation.append(f"Overall Quality: {quality}")
        interpretation.append("")

        if diagnostics.get('convergence_ok') == 'Yes':
            interpretation.append("[OK] CONVERGENCE: All R-hat values < 1.01")
            interpretation.append("   → Chains have converged to the same distribution")
        else:
            interpretation.append("[FAIL] CONVERGENCE: Some R-hat values ≥ 1.01")
            interpretation.append("   → Chains have NOT converged properly")
            interpretation.append("   → SOLUTION: Increase tune iterations or check model specification")

        interpretation.append("")

        if diagnostics.get('ess_adequate') == 'Yes':
            interpretation.append("[OK] EFFECTIVE SAMPLE SIZE: All ESS > 100")
            interpretation.append("   → Sufficient independent samples for inference")
        else:
            interpretation.append("[FAIL] EFFECTIVE SAMPLE SIZE: Some ESS ≤ 100")
            interpretation.append("   → High autocorrelation in chains")
            interpretation.append("   → SOLUTION: Increase draws or improve parametrization")

        interpretation.append("")

        pct_div = diagnostics.get('pct_divergences', 0)
        if diagnostics.get('divergences_ok') == 'Yes':
            interpretation.append(f"[OK] DIVERGENCES: {pct_div:.1f}% (acceptable)")
            interpretation.append("   → Sampler is exploring the posterior well")
        else:
            interpretation.append(f"[FAIL] DIVERGENCES: {pct_div:.1f}% (too high)")
            interpretation.append("   → Sampler having difficulty with posterior geometry")
            interpretation.append("   → SOLUTION: Use non-centered parametrization or increase target_accept")

        interpretation.append("")

        if diagnostics.get('ci_coverage_ok') == 'Yes':
            interpretation.append("[OK] CREDIBLE INTERVAL COVERAGE: Within expected range")
            interpretation.append("   → Model is well-calibrated")
        else:
            interpretation.append("[WARN]  CREDIBLE INTERVAL COVERAGE: Outside expected range")
            interpretation.append("   → Model may be over/under-confident")
            interpretation.append("   → SOLUTION: Check model specification and priors")

        interpretation.append("")
        interpretation.append("RECOMMENDATIONS:")
        if quality == 'GOOD':
            interpretation.append("  [OK] Model is reliable - proceed with results")
        elif quality == 'ACCEPTABLE':
            interpretation.append("  [WARN]  Model is acceptable but verify key findings")
            interpretation.append("  → Check territories with extreme Z-scores manually")
        else:
            interpretation.append("  [FAIL] Model quality is poor - DO NOT use for decisions")
            interpretation.append("  → Consider:")
            interpretation.append("     1. Aggregating to higher level")
            interpretation.append("     2. Collecting more data")
            interpretation.append("     3. Using simpler model")

        return interpretation

    @staticmethod
    def generate_recommendations_report(bayesian_interp: List[str],
                                        bayesian_cov_interp: List[str],
                                        reliability_info: Optional[Dict[str, Any]] = None) -> str:
        """Side-by-side comparison report of crude vs covariate models."""
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("MODEL COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")

        qualities: Dict[str, str] = {}
        if bayesian_interp:
            for line in bayesian_interp:
                if line.startswith("Overall Quality:"):
                    qualities['Bayesian'] = line.split(":")[1].strip()
                    break

        if bayesian_cov_interp:
            for line in bayesian_cov_interp:
                if line.startswith("Overall Quality:"):
                    qualities['Bayesian Covariates'] = line.split(":")[1].strip()
                    break

        lines.append("MODEL QUALITY SUMMARY:")
        for model, quality in qualities.items():
            lines.append(f"  {model}: {quality}")
        lines.append("")

        if reliability_info:
            lines.append("=" * 80)
            lines.append("RELIABILITY ASSESSMENT")
            lines.append("=" * 80)
            lines.append("")
            lines.append(f"Overall Reliability Score: {reliability_info['overall_score']:.1f}/100")
            lines.append(f"Rating: {reliability_info['rating']} {reliability_info['flag']}")
            lines.append(f"Recommendation: {reliability_info['recommendation']}")
            lines.append("")
            lines.append("Component Scores:")
            for comp_name, comp_data in reliability_info['components'].items():
                comp_label = comp_name.replace('_', ' ').title()
                lines.append(f"  • {comp_label} ({comp_data['weight']}%): {comp_data['score']:.0f}/100")
                lines.append(f"    → {comp_data['interpretation']}")
            lines.append("")
            lines.append("Interpretation:")
            if reliability_info['rating'] == 'HIGH':
                lines.append("  [OK] HIGH reliability - Results are suitable for decision-making")
                lines.append("     Data quality, sample size, and model fit are all adequate")
            elif reliability_info['rating'] == 'MODERATE':
                lines.append("  [WARN]  MODERATE reliability - Use results with caution")
                lines.append("     Some limitations in data quality, sample size, or model fit")
                lines.append("     Consider collecting more data or aggregating to higher level")
            else:
                lines.append("  [WARN]  LOW reliability - Results have high uncertainty")
                lines.append("     Significant limitations in data quality, sample size, or model fit")
                lines.append("     Strongly recommend collecting more data or aggregating to higher level")
            lines.append("")
            lines.append("=" * 80)
            lines.append("")

        lines.append("RECOMMENDATIONS:")

        good_models = [m for m, q in qualities.items() if q == 'GOOD']
        acceptable_models = [m for m, q in qualities.items() if q == 'ACCEPTABLE']

        if good_models:
            lines.append(f"  [OK] Recommended models: {', '.join(good_models)}")
            lines.append(f"     → These models have passed all quality checks")
        elif acceptable_models:
            lines.append(f"  [WARN]  Acceptable models: {', '.join(acceptable_models)}")
            lines.append(f"     → Use with caution, verify key findings")
        else:
            lines.append(f"  [FAIL] No models passed quality checks")
            lines.append(f"     → Consider aggregating to higher level or collecting more data")

        lines.append("")

        if 'Bayesian Covariates' in qualities:
            lines.append("BAYESIAN COVARIATES NOTES:")
            lines.append("  • Accounts for risk group differences")
            lines.append("  • Can detect testing artifacts")
            lines.append("  • Most comprehensive analysis")
            lines.append("")

        if 'Bayesian' in qualities:
            lines.append("BAYESIAN NOTES:")
            lines.append("  • Robust to small sample sizes")
            lines.append("  • Handles zero counts well")
            lines.append("  • Provides uncertainty quantification")
            lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)
