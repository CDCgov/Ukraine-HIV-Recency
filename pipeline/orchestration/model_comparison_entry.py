"""
Build the per-level entry that goes into the model-comparison report.

One row per level summarising the run: the model that actually fitted,
its convergence flag and max R-hat, the recommended model and a short
text reason. Used by :func:`pipeline.orchestration.generate_model_comparison`
to render ``Model_Comparison_<period>.xlsx`` at the end of the run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def build_model_comparison_entry(level_name: str, model_used: str,
                                 final_diag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compose the per-level entry dict from the analyzer's final diagnostics."""
    entry: Dict[str, Any] = {
        'level': level_name,
        'model_used': model_used,
    }

    if final_diag and 'convergence_ok' in final_diag:
        entry['final_converged'] = final_diag.get('convergence_ok', 'N/A')
        entry['final_rhat_max'] = final_diag.get('rhat_max', np.nan)
    else:
        entry['final_converged'] = 'N/A'
        entry['final_rhat_max'] = np.nan

    if model_used == 'Bayesian':
        entry['recommended_model'] = 'Bayesian'
        entry['reason'] = 'Bayesian model selected'
    elif model_used == 'Bayesian with Covariates':
        entry['recommended_model'] = 'Bayesian with Covariates'
        entry['reason'] = 'Bayesian with Covariates selected'
    else:
        entry['recommended_model'] = model_used
        entry['reason'] = 'Model selection completed'

    return entry
