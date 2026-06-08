"""
Logging setup and runtime sanity checks for the HIV hotspot pipeline.

``setup_logging`` configures the root logger with a console and/or file
handler at the requested level. It is a no-op in spawned multiprocessing
workers (PyMC uses spawn on Windows), which inherit the parent's logger
and would otherwise duplicate every line.

``check_compiler_availability`` warns when g++ is missing, because
PyTensor then falls back to Python interpretation and MCMC sampling
becomes roughly an order of magnitude slower.
"""
from __future__ import annotations

import logging
import multiprocessing
import subprocess
import sys

logger = logging.getLogger(__name__)


def setup_logging(log_to_stdout: bool = True, log_to_file: bool = True,
                  log_file: str = "pipeline.log",
                  log_level: str = "INFO") -> None:
    """Configure the root logger with console and / or file handlers.

    Args:
        log_to_stdout: Send log records to standard output.
        log_to_file: Append log records to ``log_file``.
        log_file: Path of the log file.
        log_level: Threshold name (``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, ``CRITICAL``).
    """
    # Spawned worker processes (PyMC parallel sampling on Windows) inherit
    # the parent's root logger -- reconfiguring it here would duplicate
    # every line, so the workers exit early.
    if multiprocessing.current_process().name != "MainProcess":
        return

    handlers: list[logging.Handler] = []
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_to_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.StreamHandler(sys.stdout))

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logger.info(
        f"Logging configured: stdout={log_to_stdout}, "
        f"file={log_to_file}, level={log_level}"
    )


def check_compiler_availability() -> bool:
    """Return True if g++ is on PATH, otherwise warn and return False.

    PyTensor uses C compilation for the gradient code paths. Without g++
    it falls back to a much slower pure-Python interpretation and MCMC
    sampling runs roughly an order of magnitude longer.
    """
    try:
        result = subprocess.run(
            ["g++", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            logger.info("[OK] g++ compiler detected - PyTensor will use C-compilation")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    bar = "=" * 80
    logger.warning(bar)
    logger.warning("[WARN] g++ COMPILER NOT DETECTED")
    logger.warning(bar)
    logger.warning("PyTensor will use Python interpretation instead of C-compilation.")
    logger.warning("This will make MCMC sampling approximately 10x SLOWER.")
    logger.warning("")
    logger.warning("Expected runtime:")
    logger.warning("  - Without g++: ~2 hours")
    logger.warning("  - With g++:    ~12 minutes")
    logger.warning("")
    logger.warning("TO FIX (Windows):")
    logger.warning("   conda install m2w64-toolchain")
    logger.warning("")
    logger.warning("TO FIX (Linux):")
    logger.warning("   conda install -c conda-forge gxx_linux-64")
    logger.warning("")
    logger.warning("After installation, restart terminal and re-run pipeline.")
    logger.warning(bar)
    return False
