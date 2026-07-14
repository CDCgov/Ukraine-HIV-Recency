"""
Human-readable interpretation of Bayesian diagnostics.

The :class:`DiagnosticInterpreter` turns the raw diagnostics dict produced
by :class:`~pipeline.diagnostics.reliability.ReliabilityScoreCalculator`
(and the model-fitting machinery) into bulleted English strings that go
into ``RECOMMENDATIONS.txt``. The logic here is purely textual and has no
PyMC dependency.

:meth:`interpret_bayesian_diagnostics` produces the per-model
interpretation: one bullet block per quality metric (R-hat, ESS,
divergences, credible-interval coverage) plus a closing "RECOMMENDATIONS"
block.
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

