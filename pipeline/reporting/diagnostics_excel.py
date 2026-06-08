"""
Excel writer for the per-run diagnostics dump.

The :func:`save_diagnostics` routine takes the analyzer's accumulated
``diagnostics`` list (one dict per fit) and writes a two-sheet Excel
file: a flat ``Model Diagnostics`` table with the priority columns
(level / timestamp / model / convergence) on the left, and an
``Interpretation Guide`` sheet with a small lookup table that explains
each metric in plain English so a reader can audit the numbers without
opening the source code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)


def save_diagnostics(diagnostics: Sequence[Dict[str, Any]],
                     output_path: Union[str, Path]) -> None:
    """Write per-fit diagnostics + interpretation guide to ``output_path``."""
    if not diagnostics:
        return

    df_diag = pd.DataFrame(diagnostics)

    priority_cols = ['level', 'timestamp', 'model_name', 'n_territories', 'converged', 'convergence_ok',
                     'overall_quality', 'n_quality_checks_passed', 'n_quality_checks_total']

    existing_priority = [c for c in priority_cols if c in df_diag.columns]
    other_cols = [c for c in df_diag.columns if c not in priority_cols]

    col_order = existing_priority + sorted(other_cols)
    df_diag = df_diag[col_order]

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_diag.to_excel(writer, sheet_name='Model Diagnostics', index=False)

        interpretation = pd.DataFrame({
            'Metric': [
                'AIC/BIC',
                'Pseudo R²',
                'Dispersion Parameter',
                'Shapiro-Wilk p-value',
                'Z-scores Mean',
                'Z-scores Std',
                'Extreme Territories %',
                'Model Converged',
            ],
            'Good Range': [
                'Lower is better',
                '0.1 - 0.5 (acceptable)',
                '~1.0 (if >1.5, overdispersion)',
                '> 0.05 (residuals normal)',
                'Close to 0',
                'Close to 1',
                '~5% (for threshold=2.0)',
                'True',
            ],
            'Interpretation': [
                'Model fit quality (penalizes complexity)',
                'Proportion of deviance explained',
                'Variance/mean ratio (NB handles overdispersion)',
                'Tests if residuals follow normal distribution',
                'Z-scores should be centered at 0',
                'Z-scores should have unit variance',
                'Expected rate of extreme values under normal distribution',
                'Whether optimization algorithm converged',
            ],
        })
        interpretation.to_excel(writer, sheet_name='Interpretation Guide', index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except (AttributeError, TypeError):
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    logger.info(f"[OK] Diagnostics saved: {output_path}")
