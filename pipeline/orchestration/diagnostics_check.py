"""
Boolean gate over the model-diagnostics dict.

:func:`check_diagnostics` returns ``True`` iff the model converged.
Shapiro-Wilk normality of residuals is intentionally NOT a gate -- count
models (Negative Binomial / Poisson / Binomial) do not produce normally-
distributed residuals, so failing the Shapiro-Wilk test on them would
incorrectly reject converged models.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_diagnostics(diagnostics: dict) -> bool:
    """Return True when the diagnostics dict says the model converged."""
    if not diagnostics:
        return False

    if not diagnostics.get('converged', True):
        logger.warning("Model did not converge")
        return False

    return True
