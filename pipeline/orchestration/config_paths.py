"""
Path resolution helpers for configuration files.

The pipeline's config holds file paths (``excel_path``, geo layers) as
strings. When the user invokes the pipeline from a different working
directory the relative paths break; :func:`resolve_paths` rewrites the
known path keys against the script's parent directory so the pipeline
runs regardless of CWD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


def resolve_paths(config: Dict[str, Any], script_dir: Union[str, Path]) -> Dict[str, Any]:
    """Make known relative path keys absolute under ``script_dir``."""
    script_dir = Path(script_dir)
    path_keys = ['excel_path']
    for key in path_keys:
        if key in config:
            p = Path(config[key])
            if not p.is_absolute():
                config[key] = str(script_dir / p)
    return config
