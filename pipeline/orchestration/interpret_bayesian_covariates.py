"""
Render the per-level interpretation report for the Covariates model.

Mirrors :func:`interpret_bayesian_diagnostics` but writes into the
``bayesian_covariates`` output subtree and also emits the detailed
territory-level analysis when the diagnostics dict supplies one
(``detailed_analysis`` key from the stratified covariates fit).

The 5% divergent-transitions restart recommendation matches the
Bayesian path so a reader comparing the two report files sees the
same wording when both fits exceed the threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pipeline.diagnostics import DiagnosticInterpreter

logger = logging.getLogger(__name__)


def interpret_bayesian_covariates_diagnostics(orchestrator: Any,
                                              diag_bayes_cov: Optional[Dict[str, Any]],
                                              level_name: str,
                                              period_str: str) -> Optional[List[str]]:
    """Run covariates interpretation, write reports, return interpretation lines (or None)."""
    if not diag_bayes_cov:
        return None

    try:
        bayesian_cov_interpretation = DiagnosticInterpreter.interpret_bayesian_diagnostics(diag_bayes_cov)

        pct_divergences = diag_bayes_cov.get('pct_divergences', 0)
        current_param = orchestrator.config.get('bayesian_parametrization', 'centered')

        if pct_divergences > 5.0 and current_param == 'centered':
            logger.warning("\n" + "=" * 80)
            logger.warning("[WARN]  RECOMMENDATION: RESTART WITH NON-CENTERED PARAMETRIZATION")
            logger.warning("=" * 80)
            logger.warning(f"Bayesian Covariates - Divergences: {pct_divergences:.1f}% (threshold: 5%)")
            logger.warning("Non-centered parametrization will reduce divergences.")
            logger.warning("=" * 80 + "\n")

        # Save interpretation report
        is_hex = level_name.startswith("Hex_Res")
        interp_output_file = orchestrator.get_output_path(
            "bayesian_covariates", level_name,
            f"Interpretation_Bayesian_Covariates_{level_name}_{period_str}.txt",
            is_hex=is_hex,
        )
        with open(interp_output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(bayesian_cov_interpretation))

            if pct_divergences > 5.0 and current_param == 'centered':
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("[WARN]  CRITICAL RECOMMENDATION: RESTART WITH NON-CENTERED\n")
                f.write("=" * 80 + "\n")
                f.write(f"Divergences: {pct_divergences:.1f}% (threshold: 5%)\n")
                f.write("Non-centered parametrization will reduce divergences.\n")
                f.write("\nHow to restart:\n")
                f.write("1. Run the script again\n")
                f.write("2. Choose '2 - Non-centered' in menu\n")

        logger.info(f"[OK] Bayesian Covariates interpretation saved: {interp_output_file}")

        # Save detailed territory analysis if available
        if 'detailed_analysis' in diag_bayes_cov:
            detailed_output_file = orchestrator.get_output_path(
                "bayesian_covariates", level_name,
                f"Detailed_Territory_Analysis_{level_name}_{period_str}.txt",
                is_hex=is_hex,
            )
            with open(detailed_output_file, 'w', encoding='utf-8') as f:
                f.write(diag_bayes_cov['detailed_analysis'])
            logger.info(f"[OK] Detailed territory analysis saved: {detailed_output_file}")

        return bayesian_cov_interpretation
    except (IOError, KeyError, AttributeError) as e:
        logger.error(f"Failed to interpret Bayesian Covariates diagnostics: {e}")
        return None
