"""
Default result dicts for outbreak / testing-artifact detection.

When :func:`detect_outbreak_and_artifact` cannot run hard stratification
(typically because a territory has fewer than three tests in one of the
risk groups), it falls back to the soft-stratification path and emits
the :func:`soft_fallback_result` dict. Keeping the schema in one place
ensures the downstream Excel report has a stable set of columns
regardless of which path produced the row.
"""

from __future__ import annotations

from typing import Any, Dict


def soft_fallback_result() -> Dict[str, Any]:
    """Default outbreak/artifact result for the SOFT stratification fallback."""
    return {
        'high_outbreak': False,
        'low_outbreak': False,
        'testing_artifact': False,
        'artifact_contribution': 0.0,
        'artifact_severity': 'NO_ARTIFACT',
        'outbreak_type': 'INSUFFICIENT DATA FOR HARD STRATIFICATION',
        'explanation': 'Insufficient data for separate risk group analysis (need ≥3 tests in each group). Used SOFT stratification.',
        'high_pvalue': 1.0,
        'low_pvalue': 1.0,
        'testing_shift': 0.0,
        'stratification_method': 'SOFT',
        'high_observed_curr': 0.0,
        'low_observed_curr': 0.0,
        'high_ci_lower': 0.0,
        'high_ci_upper': 0.0,
        'low_ci_lower': 0.0,
        'low_ci_upper': 0.0,
    }
