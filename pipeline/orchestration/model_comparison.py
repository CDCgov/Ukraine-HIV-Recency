"""
Per-level model comparison Excel report.

Renders the orchestrator's ``model_comparison_data`` list (one row per
admin / hex level) into ``Model_Comparison_<period>.xlsx`` -- a single-
sheet table that shows, for each level, which model actually ran, whether
it converged, the max R-hat, the recommended model, and a short
explanation. Used as a quality summary at the end of every run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)


def generate_model_comparison(model_comparison_data: Sequence[Dict[str, Any]],
                              output_path: Union[str, Path]) -> None:
    """Write the comparison Excel and log a short text summary."""
    if not model_comparison_data:
        logger.warning("No model comparison data to save")
        return

    logger.info("\n" + "=" * 60)
    logger.info("GENERATING MODEL COMPARISON")
    logger.info("=" * 60)

    df_comparison = pd.DataFrame(model_comparison_data)

    column_order = [
        'level',
        'model_used',
        'final_converged',
        'final_rhat_max',
        'recommended_model',
        'reason',
    ]
    df_comparison = df_comparison[column_order]

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_comparison.to_excel(writer, sheet_name='Model Comparison', index=False)

        worksheet = writer.sheets['Model Comparison']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 15
        worksheet.column_dimensions['C'].width = 15
        worksheet.column_dimensions['D'].width = 15
        worksheet.column_dimensions['E'].width = 20
        worksheet.column_dimensions['F'].width = 60

    logger.info(f"[OK] Model comparison saved: {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 80)
    for _, row in df_comparison.iterrows():
        logger.info(f"\n{row['level']}:")
        logger.info(f"  Model Used: {row['model_used']}")
        logger.info(f"  Recommended: {row['recommended_model']}")
        logger.info(f"  Reason: {row['reason']}")
    logger.info("=" * 80 + "\n")
