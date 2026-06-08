"""
Pydantic data models for pipeline configuration.

Used by :func:`pipeline.orchestration.validate_config` (via the
``bayesian_config_cls`` argument) and by config parsing. When
Pydantic isn't installed the names are still importable -- they
just resolve to ``None`` -- so call sites that branch on
``if PYDANTIC_AVAILABLE`` continue to work without modification.

The schema is intentionally permissive on the top-level
:class:`Config` (``extra='allow'``) so config files can carry
fields beyond the validated ones without raising; the strict
checks are concentrated on the substructures that actually feed
the model fits (``BayesianConfig``, ``AnalysisPeriod``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    class BaseModel:  # type: ignore[no-redef]
        """No-op stand-in when Pydantic is unavailable."""
        pass


if PYDANTIC_AVAILABLE:

    class AnalysisPeriod(BaseModel):
        """Analysis period configuration"""
        start: datetime
        end: datetime

        @field_validator('end')
        @classmethod
        def end_after_start(cls, v, info):
            """Reject a period whose end is not strictly after its start."""
            if 'start' in info.data and v <= info.data['start']:
                raise ValueError('end date must be after start date')
            return v


    class ColumnNames(BaseModel):
        """Column names configuration"""
        territory_name: str = "territory_name"
        recent_count_curr: str = "recent_count_curr"
        recent_count_hist: str = "recent_count_hist"
        all_tested_curr: str = "all_tested_curr"
        all_tested_hist: str = "all_tested_hist"
        recent_proportion_curr: str = "recent_proportion_curr"
        recent_proportion_hist: str = "recent_proportion_hist"
        predicted: str = "predicted"
        residual: str = "residual"
        z_national: str = "z_national"
        z_residual: str = "z_residual"
        combined_z: str = "combined_z"
        classification: str = "classification"
        recommended_level: str = "recommended_level"
        deviation_pct: str = "deviation_pct"
        national_baseline: str = "national_baseline"
        log_tested: str = "log_tested"
        predicted_prob: str = "predicted_prob"
        prob_lower: str = "prob_lower"
        prob_upper: str = "prob_upper"
        count_lower: str = "count_lower"
        count_upper: str = "count_upper"
        exceedance_prob: str = "exceedance_prob"


    class BayesianConfig(BaseModel):
        """Bayesian model configuration"""
        # Allow optional, less-common keys (e.g. ``resolution_sigma_multiplier``)
        # to pass validation instead of being silently dropped, so they reach
        # the analyzers that read them.
        model_config = ConfigDict(extra='allow')

        use_non_centered: bool = True
        auto_select_parametrization: bool = True
        non_centered_threshold_territories: int = Field(default=50, ge=1)
        non_centered_threshold_avg_tests: int = Field(default=20, ge=1)
        prior_sigma: float = Field(default=1.0, gt=0)
        prior_sigma_small_sample: float = Field(default=0.5, gt=0)
        small_sample_threshold: int = Field(default=20, ge=1)
        # Optional FRR/MDRI correction for recency assay false recent rate
        frr: Optional[float] = Field(default=None, ge=0, le=1, description="False Recent Rate (0-1). If set, recent counts are corrected: recent_adj = max(0, recent - tested * frr)")
        mdri_days: Optional[int] = Field(default=None, gt=0, description="Mean Duration of Recent Infection in days (informational, not used in correction)")


    class IOOptimization(BaseModel):
        """IO optimization configuration"""
        use_parquet: bool = True
        cache_geodata: bool = True


    class ValidationConfig(BaseModel):
        """Validation configuration"""
        use_pydantic: bool = True
        strict_types: bool = True


    class Config(BaseModel):
        """Main configuration model"""
        model_config = ConfigDict(extra='allow')

        excel_path: str
        analysis_mode: str = Field(default="h3_hexagons", pattern="^(h3_hexagons)$")
        output_dir: str = "output"
        target_crs: str = "EPSG:3857"
        random_seed: int = Field(default=42, description="Random seed for reproducibility")
        analysis_period: AnalysisPeriod
        column_names: ColumnNames = Field(default_factory=ColumnNames)
        bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
        io_optimization: IOOptimization = Field(default_factory=IOOptimization)
        validation: ValidationConfig = Field(default_factory=ValidationConfig)


    class CaseDataRow(BaseModel):
        """Single row of case data from Excel"""
        model_config = ConfigDict(arbitrary_types_allowed=True)

        # Required columns
        test_date: datetime = Field(description="Date of HIV test")
        result: str = Field(description="Test result (Recent/Long-term/Negative)")

        # Administrative location (at least one required)
        adm3: Optional[str] = Field(default=None, description="Community (OTG) name")
        adm2: Optional[str] = Field(default=None, description="District (Rayon) name")
        adm1: Optional[str] = Field(default=None, description="Oblast name")

        # Optional fields
        age: Optional[int] = Field(default=None, ge=0, le=120)
        sex: Optional[str] = Field(default=None, pattern="^(M|F|Male|Female|Чоловік|Жінка)$")
        risk_group: Optional[str] = None

        @field_validator('result')
        @classmethod
        def validate_result(cls, v):
            """Constrain the test result to the accepted EN/UA labels."""
            valid_results = ['Recent', 'Long-term', 'Negative', 'Нещодавня', 'Давня', 'Негативний']
            if v not in valid_results:
                raise ValueError(f'result must be one of {valid_results}')
            return v

        @field_validator('adm3', 'adm2', 'adm1')
        @classmethod
        def validate_location(cls, v, info):
            """Require at least one administrative level to be present."""
            # At least one location field must be present
            if v is None and all(info.data.get(f) is None for f in ['adm3', 'adm2', 'adm1'] if f != info.field_name):
                raise ValueError('At least one of adm3, adm2, or adm1 must be provided')
            return v


    class TerritoryData(BaseModel):
        """Validated territory-level aggregated data"""
        model_config = ConfigDict(arbitrary_types_allowed=True)

        territory_name: str
        all_tested_curr: int = Field(ge=0)
        recent_count_curr: int = Field(ge=0)
        all_tested_hist: int = Field(ge=0)
        recent_count_hist: int = Field(ge=0)

        @field_validator('recent_count_curr')
        @classmethod
        def recent_not_exceed_total(cls, v, info):
            """Recent count cannot exceed the total tested in the period."""
            if 'all_tested_curr' in info.data and v > info.data['all_tested_curr']:
                raise ValueError('recent_count_curr cannot exceed all_tested_curr')
            return v

        @field_validator('recent_count_hist')
        @classmethod
        def recent_hist_not_exceed_total(cls, v, info):
            """Historical recent count cannot exceed historical total tested."""
            if 'all_tested_hist' in info.data and v > info.data['all_tested_hist']:
                raise ValueError('recent_count_hist cannot exceed all_tested_hist')
            return v

else:
    # Pydantic not available - create dummy attributes so importers
    # using ``from pipeline.config.models import BayesianConfig`` continue to
    # work; call sites that need real validation guard on PYDANTIC_AVAILABLE.
    AnalysisPeriod = None
    ColumnNames = None
    BayesianConfig = None
    IOOptimization = None
    ValidationConfig = None
    Config = None
    CaseDataRow = None
    TerritoryData = None
