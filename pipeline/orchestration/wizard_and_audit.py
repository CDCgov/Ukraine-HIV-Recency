"""
Record the fixed model-structure decision on the audit trail.

After the Data Quality Assessment stage this opens a Model Selection stage and
records the one standing decision: exchangeable random effects (the right
structure for facility-based surveillance, where adjacent units need not be
epidemiologically similar). It also returns the structural-zeros percentage the
caller uses downstream.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import geopandas as gpd

logger = logging.getLogger(__name__)


def run_wizard_and_record_decisions(audit_trail: Any,
                                    gdf: gpd.GeoDataFrame,
                                    level_name: str,
                                    config: Dict[str, Any]) -> Dict[str, Any]:
    """Record the exchangeable-structure decision; return the level config."""
    n_active_sites = (gdf['all_tested_curr'] > 0).sum()
    n_total = len(gdf)
    pct_structural_zeros = ((n_total - n_active_sites) / n_total * 100) if n_total > 0 else 0
    if 'site_present' in gdf.columns:
        n_structural_zeros = (~gdf['site_present']).sum()
        pct_structural_zeros = (n_structural_zeros / n_total * 100) if n_total > 0 else 0

    logger.info("Configuration: spatial_structure=exchangeable")

    audit_trail.end_stage()
    audit_trail.start_stage(
        "Model Selection",
        "Fix the model structure for this level"
    )
    audit_trail.add_decision(
        test_name="Spatial Structure Configuration",
        test_type="diagnostic",
        result="Spatial structure: EXCHANGEABLE",
        decision="Use EXCHANGEABLE random effects (facility-based data)",
        reason="Facility-based data — no spatial autocorrelation needed",
        impact="Hierarchical partial pooling without spatial structure",
        details={'spatial_structure': 'exchangeable'},
    )

    return {'pct_structural_zeros': pct_structural_zeros}
