"""
Side-effect bootstrap module: configure environment variables,
warning filters, and noisy third-party loggers BEFORE anything in
the pipeline touches PyMC, PyTensor, or NumPy.

Imported at the very top of the entry-point script (and re-imported
by tests that drive the pipeline) so the process-wide config is
applied in a single place. Importing this module is the side-effect;
it exposes no public API.

Why so many warning filters: PyMC/PyTensor/ArviZ/threadpoolctl all
emit log lines that aren't actionable for the user during a normal
run -- convergence and divergence warnings still come through
because they are emitted via :mod:`logging`, not the warnings
system, and are *not* filtered here.

Why both the in-process filters and the ``PYTHONWARNINGS``
environment variable: PyMC's sampler spawns worker processes (on
Windows via ``spawn``), and those workers re-execute module-level
code from scratch. They inherit ``os.environ`` but not the parent's
runtime ``warnings`` state, so the env var is what keeps the
worker output clean. ``KMP_DUPLICATE_LIB_OK`` also has to be set
before any OpenMP-linked library loads, hence the very-early
positioning.
"""

from __future__ import annotations

import logging as _logging
import os
import warnings

# === Environment variables (must be set before any OpenMP / PyTensor import) ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

_existing_pw = os.environ.get('PYTHONWARNINGS', '')
_extra_pw = (
    'ignore::UserWarning:threadpoolctl,'
    'ignore::RuntimeWarning:threadpoolctl,'
    'ignore::UserWarning:pymc,'
    'ignore::UserWarning:arviz'
)
os.environ['PYTHONWARNINGS'] = f"{_existing_pw},{_extra_pw}" if _existing_pw else _extra_pw

os.environ.setdefault('PYTENSOR_FLAGS', 'exception_verbosity=low')


# === Noisy third-party loggers ===
for _noisy in (
    'pymc', 'pymc.sampling', 'pymc.sampling.parallel', 'pymc.sampling.mcmc',
    'pymc.backends', 'pymc.util',
    'pytensor', 'pytensor.tensor', 'pytensor.compile',
    'numexpr', 'aesara', 'arviz', 'arviz.stats',
):
    _l = _logging.getLogger(_noisy)
    _l.setLevel(_logging.WARNING)
    _l.propagate = False


# === Warning filters (in-process; the env var above covers child processes) ===
warnings.filterwarnings('ignore', category=UserWarning, module='threadpoolctl')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='threadpoolctl')
warnings.filterwarnings('ignore', category=FutureWarning, module='arviz')

warnings.filterwarnings(
    'ignore',
    message='.*Intel OpenMP.*LLVM OpenMP.*loaded at the same time.*',
    category=UserWarning,
)
# threadpoolctl sometimes emits the OpenMP warning as RuntimeWarning.
warnings.filterwarnings(
    'ignore',
    message='.*Intel OpenMP.*LLVM OpenMP.*loaded at the same time.*',
    category=RuntimeWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*recommend running at least 4 chains.*',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r'.*Estimated shape parameter of Pareto distribution is greater than 0\.\d+.*',
    category=UserWarning,
    module='arviz',
)
warnings.filterwarnings(
    'ignore',
    message='.*The return type of `Dataset.dims` will be changed.*',
    category=FutureWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*Inverting hessian failed.*',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*invalid value encountered in log.*',
    category=RuntimeWarning,
    module='statsmodels',
)
warnings.filterwarnings(
    'ignore',
    message='.*ArviZ is undergoing a major refactor.*',
    category=FutureWarning,
)
