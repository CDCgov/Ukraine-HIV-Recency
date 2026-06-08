"""
Config validation.

The :func:`validate_config` routine catches type errors, missing keys,
and invalid values at load time -- before any model fitting starts and
before the run timestamps any output directories. Two classes of issue:

* **Errors** -- raise ``ValueError`` immediately. Examples: bayesian
  block fails Pydantic validation; analysis_period dates won't parse;
  hex_resolutions is not a list of ints.
* **Warnings** -- log a warning and continue. Examples: the configured
  Excel file does not exist (so the run will fail later when the loader
  tries to read it; the warning is upstream notice).

A ``BayesianConfig`` Pydantic model is taken as a parameter so this
module doesn't need to track the project's optional Pydantic-availability
state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd

logger = logging.getLogger(__name__)


def validate_config(config: Dict[str, Any],
                    bayesian_config_cls: Optional[Type] = None) -> Dict[str, Any]:
    """Validate the config dict; raise ValueError on errors, log warnings."""
    errors = []
    warnings_list = []

    if 'bayesian' in config and bayesian_config_cls is not None:
        try:
            validated = bayesian_config_cls(**config['bayesian'])
            config['bayesian'] = validated.model_dump()
        except Exception as e:
            errors.append(f"bayesian: {e}")

    if 'analysis_period' in config:
        ap = config['analysis_period']
        if not isinstance(ap, dict):
            errors.append("analysis_period: must be a dict with 'start' and 'end'")
        else:
            for key in ['start', 'end']:
                if key not in ap:
                    errors.append(f"analysis_period: missing '{key}'")
                else:
                    try:
                        pd.to_datetime(ap[key])
                    except (ValueError, TypeError):
                        errors.append(f"analysis_period.{key}: invalid date format '{ap[key]}' (expected YYYY-MM-DD)")

    numeric_fields = {
        'target_crs': (str, None),
    }
    for field, (expected_type, _constraint) in numeric_fields.items():
        if field in config and not isinstance(config[field], expected_type):
            errors.append(f"{field}: expected {expected_type.__name__}, got {type(config[field]).__name__}")

    if 'hex_resolutions' in config:
        hr = config['hex_resolutions']
        if not isinstance(hr, list) or not all(isinstance(x, int) for x in hr):
            errors.append("hex_resolutions: must be a list of integers")

    for path_key in ['excel_path']:
        if path_key in config:
            p = Path(config[path_key])
            if not p.exists():
                warnings_list.append(f"{path_key}: file not found: {config[path_key]}")

    if errors:
        for err in errors:
            logger.error(f"Config validation error: {err}")
        raise ValueError(f"Config validation failed with {len(errors)} error(s). See log for details.")

    if warnings_list:
        for w in warnings_list:
            logger.warning(f"Config validation warning: {w}")

    logger.info("Config validation passed")
    return config
