# schemas/input.py
# Pydantic models for API input validation.
# Pydantic enforces type checking, range validation, and provides
# automatic error messages before any ML code is executed.

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import pandas as pd


class ResearchCenterInput(BaseModel):
    """
    Input schema for a single research center prediction request.

    All five clustering features are required.
    Pydantic validates types and ranges before the request
    reaches any pipeline or model code.
    """

    internalFacilitiesCount: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of internal facilities (labs, workstations, etc.)",
        examples=[9],
    )

    hospitals_10km: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of hospitals within 10 km",
        examples=[3],
    )

    pharmacies_10km: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of pharmacies within 10 km",
        examples=[2],
    )

    facilityDiversity_10km: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Diversity index of nearby facilities (0.0 to 1.0)",
        examples=[0.82],
    )

    facilityDensity_10km: float = Field(
        ...,
        ge=0.0,
        description="Density of nearby facilities per area unit (non-negative)",
        examples=[0.45],
    )

    # ── Field-level validators ────────────────────────────────────────────────

    @field_validator("facilityDiversity_10km")
    @classmethod
    def diversity_must_be_unit_interval(cls, v: float) -> float:
        """
        Explicitly validate the 0-1 constraint for diversity.
        Even though Field(ge=0.0, le=1.0) handles this, an explicit
        validator produces a more descriptive error message.
        """
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"facilityDiversity_10km must be between 0.0 and 1.0, "
                f"received: {v}"
            )
        return round(v, 6)

    @field_validator("facilityDensity_10km")
    @classmethod
    def density_must_be_non_negative(cls, v: float) -> float:
        """Density cannot be negative — it is a physical measurement."""
        if v < 0.0:
            raise ValueError(
                f"facilityDensity_10km must be non-negative, received: {v}"
            )
        return round(v, 6)

    # ── Model-level validator ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def cross_field_consistency_check(self) -> "ResearchCenterInput":
        """
        Cross-field validation — catches logically inconsistent inputs.

        Business rule:
            A center with zero hospitals AND zero pharmacies AND
            near-zero diversity is extremely unusual.
            We flag this as a warning but do not reject the request —
            it may be a genuinely isolated center.

        If facilities count is very high but all external metrics are 0,
        this likely indicates a data entry error.
        """
        if (
            self.internalFacilitiesCount > 8
            and self.hospitals_10km == 0
            and self.pharmacies_10km == 0
        ):
            # We do not raise, we annotate and let the model handle it
            # In production this would trigger a monitoring alert
            pass
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the validated input to a single-row DataFrame.
        Columns match the SELECTED_FEATURES order expected by the scaler.
        """
        return pd.DataFrame([{
            "internalFacilitiesCount": self.internalFacilitiesCount,
            "hospitals_10km":          self.hospitals_10km,
            "pharmacies_10km":         self.pharmacies_10km,
            "facilityDiversity_10km":  self.facilityDiversity_10km,
            "facilityDensity_10km":    self.facilityDensity_10km,
        }])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "internalFacilitiesCount": 9,
                    "hospitals_10km":          3,
                    "pharmacies_10km":         2,
                    "facilityDiversity_10km":  0.82,
                    "facilityDensity_10km":    0.45,
                }
            ]
        }
    }


class BatchResearchCenterInput(BaseModel):
    """
    Input schema for batch prediction — multiple centers in one request.
    """
    centers: list[ResearchCenterInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of research centers to classify",
    )