"""
Run the model-configuration wizard and record its decisions on the trail.

After the Data Quality Assessment stage, the wizard decides two
things for this level: the spatial structure (exchangeable, for
facility-based data) and whether to use LOO-IC for model selection.
The CLI arguments (cached on the orchestrator on first level and
reused for later ones) override the interactive flow.

This routine closes the Data Quality Assessment audit stage, opens
the Model Selection stage, runs the wizard, and records two decision
rows on the trail (Spatial structure, Model selection method). It
returns the level-scoped configuration values (use_loo_ic and the
structural-zeros percentage) for the caller to use in downstream
model selection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import geopandas as gpd

from pipeline.config import ModelConfigurationWizard

logger = logging.getLogger(__name__)


def run_wizard_and_record_decisions(audit_trail: Any,
                                    gdf: gpd.GeoDataFrame,
                                    level_name: str,
                                    cli_args: Dict[str, Any],
                                    config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the wizard, record decisions on ``audit_trail``, return the level config."""
    # Analyze data and ask user for configuration (if not provided via CLI)
    n_active_sites = (gdf['all_tested_curr'] > 0).sum()
    n_total = len(gdf)
    pct_structural_zeros = ((n_total - n_active_sites) / n_total * 100) if n_total > 0 else 0

    # Check if site_present column exists
    if 'site_present' in gdf.columns:
        n_structural_zeros = (~gdf['site_present']).sum()
        pct_structural_zeros = (n_structural_zeros / n_total * 100) if n_total > 0 else 0

    # Run wizard (will skip if CLI args provided)
    wizard_config = ModelConfigurationWizard.run_wizard(
        n_active_sites=n_active_sites,
        pct_structural_zeros=pct_structural_zeros,
        level_name=level_name,
        cli_args=cli_args,
        config=config
    )

    # Use local variables for this level (don't overwrite self attributes)
    level_use_loo_ic = wizard_config['use_loo_ic']

    logger.info(f"Configuration: spatial_structure=exchangeable, "
               f"use_loo_ic={level_use_loo_ic}")

    # End Data Quality Assessment stage
    audit_trail.end_stage()

    # Start Model Selection stage
    audit_trail.start_stage(
        "Model Selection",
        "Select appropriate statistical model based on data characteristics and user configuration"
    )

    # Record wizard configuration decisions
    audit_trail.add_decision(
        test_name="Spatial Structure Configuration",
        test_type="diagnostic",
        result="Spatial structure: EXCHANGEABLE",
        decision="Use EXCHANGEABLE random effects (facility-based data)",
        reason="Facility-based data — no spatial autocorrelation needed",
        impact="Hierarchical partial pooling without spatial structure",
        details={'spatial_structure': 'exchangeable'}
    )

    audit_trail.add_decision(
        test_name="Model Selection Method",
        test_type="diagnostic",
        result=f"Use LOO-IC: {level_use_loo_ic}",
        decision="LOO-IC based selection" if level_use_loo_ic else "Heuristic based selection",
        reason="LOO-IC provides rigorous model comparison via cross-validation" if level_use_loo_ic else "Heuristic rules based on data characteristics",
        impact="More computationally intensive but more reliable" if level_use_loo_ic else "Faster but less rigorous",
        details={'use_loo_ic': level_use_loo_ic}
    )

    return {
        'use_loo_ic': level_use_loo_ic,
        'pct_structural_zeros': pct_structural_zeros,
    }
