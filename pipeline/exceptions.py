"""
Exception hierarchy for the HIV hotspot detection pipeline.

The hierarchy separates two operational categories so the orchestrator can
react appropriately:

* ``RecoverableError`` -- the problem is local to a single territory or
  administrative level. The offending item can be skipped and processing
  continues (e.g. one polygon has an invalid geometry, one level fails to
  converge).
* ``FatalError`` -- the problem invalidates the whole run and execution must
  stop (e.g. the input file is missing, the configuration is invalid, required
  columns are absent).

Both derive from :class:`PipelineError`, so callers may catch the whole family
with a single ``except PipelineError``.
"""


class PipelineError(Exception):
    """Base class for every error raised by the pipeline."""
    pass


class RecoverableError(PipelineError):
    """A per-item failure that can be skipped without aborting the run."""
    pass


class FatalError(PipelineError):
    """A run-level failure that must stop execution."""
    pass


class DataValidationError(FatalError):
    """Input data failed a structural or type check."""
    pass


class ModelConvergenceError(RecoverableError):
    """The model failed to converge for a specific territory or level."""
    pass


class GeometryError(RecoverableError):
    """A territory has invalid or missing geometry."""
    pass


class InsufficientDataError(RecoverableError):
    """A territory has too little data to analyse."""
    pass
