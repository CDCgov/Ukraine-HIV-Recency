"""
Render the per-level interpretation report for the standard Bayesian fit.

Wraps :meth:`DiagnosticInterpreter.interpret_bayesian_diagnostics`
to (a) write the textual interpretation to
``Interpretation_Bayesian_<level>_<period>.txt`` and (b) append a
"RESTART WITH NON-CENTERED PARAMETRIZATION" recommendation when the
posterior shows more than 5% divergences and the current run used
centered parametrisation. The 5% threshold mirrors Stan's
divergent-transitions guidance.

Returns the interpretation lines so the caller can add them to its
per-level ``interpretations`` dict for the comparison report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pipeline.diagnostics import DiagnosticInterpreter

logger = logging.getLogger(__name__)


def interpret_bayesian_diagnostics(orchestrator: Any,
                                   diag_bayes: Optional[Dict[str, Any]],
                                   level_name: str,
                                   period_str: str) -> Optional[List[str]]:
    """Run interpretation, write report, return interpretation lines (or None)."""
    if not diag_bayes:
        return None

    try:
        bayesian_interpretation = DiagnosticInterpreter.interpret_bayesian_diagnostics(diag_bayes)

        # Check if we should recommend non-centered parametrization
        pct_divergences = diag_bayes.get('pct_divergences', 0)
        current_param = orchestrator.config.get('bayesian_parametrization', 'centered')

        if pct_divergences > 5.0 and current_param == 'centered':
            logger.warning("\n" + "=" * 80)
            logger.warning("[WARN]  RECOMMENDATION: RESTART WITH NON-CENTERED PARAMETRIZATION")
            logger.warning("=" * 80)
            logger.warning(f"Current divergences: {pct_divergences:.1f}% (threshold: 5%)")
            logger.warning("Non-centered parametrization will reduce divergences for small samples.")
            logger.warning("\nHow to restart:")
            logger.warning("1. Run the script again")
            logger.warning("2. Choose '2 - Non-centered' in Bayesian Parametrization menu")
            logger.warning("=" * 80 + "\n")

        # Save interpretation report
        is_hex = level_name.startswith("Hex_Res")
        interp_output_file = orchestrator.get_output_path(
            "bayesian", level_name,
            f"Interpretation_Bayesian_{level_name}_{period_str}.txt",
            is_hex=is_hex,
        )
        with open(interp_output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(bayesian_interpretation))

            # Add parametrization recommendation if needed
            if pct_divergences > 5.0 and current_param == 'centered':
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("[WARN]  CRITICAL RECOMMENDATION: RESTART WITH NON-CENTERED\n")
                f.write("=" * 80 + "\n")
                f.write(f"Divergences: {pct_divergences:.1f}% (threshold: 5%)\n")
                f.write("Non-centered parametrization will reduce divergences.\n")
                f.write("\nHow to restart:\n")
                f.write("1. Run the script again\n")
                f.write("2. Choose '2 - Non-centered' in menu\n")

        logger.info(f"[OK] Bayesian interpretation saved: {interp_output_file}")
        return bayesian_interpretation
    except (IOError, KeyError, AttributeError) as e:
        logger.error(f"Failed to interpret Bayesian diagnostics: {e}")
        return None
