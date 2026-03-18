# schemas/output.py
# Pydantic models for API response structure.
# Consistent response shapes make the API easier to consume and test.

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionOutput(BaseModel):
    """Single prediction response."""

    predictedCategory: str = Field(
        ...,
        description="Predicted quality tier: Premium | Standard | Basic",
        examples=["Premium"],
    )
    clusterNumber: int = Field(
        ...,
        description="Raw cluster number assigned by K-Means (0, 1, or 2)",
        examples=[0],
    )
    confidence: Optional[float] = Field(
        None,
        description=(
            "Confidence proxy: 1 - (distance_to_assigned_centroid / "
            "distance_to_nearest_other_centroid). "
            "Range 0.0 to 1.0. Higher = more confident assignment."
        ),
        examples=[0.87],
    )
    modelVersion: str = Field(
        ...,
        description="MLflow run ID of the model used for this prediction",
        examples=["a1b2c3d4e5f6"],
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the prediction",
    )


class BatchPredictionOutput(BaseModel):
    """Batch prediction response."""

    predictions:  List[PredictionOutput]
    totalCount:   int
    tierSummary:  Dict[str, int] = Field(
        description="Count of each tier in the batch result"
    )


class TrainingOutput(BaseModel):
    """Response returned after a training run completes."""

    status:          str
    mlflowRunId:     str
    mlflowRunUrl:    str
    silhouetteScore: float
    inertia:         float
    clusterSizes:    Dict[str, int]
    tierMapping:     Dict[int, str]
    featuresUsed:    List[str]
    hyperparameters: Dict[str, Any]
    trainedAt:       datetime = Field(default_factory=datetime.utcnow)


class ValidationReport(BaseModel):
    """Detailed report from the data validation pipeline step."""

    isValid:          bool
    rowCount:         int
    columnCount:      int
    missingValues:    Dict[str, int]
    outOfRangeValues: Dict[str, int]
    duplicateRows:    int
    warnings:         List[str]
    errors:           List[str]


class HealthOutput(BaseModel):
    """Health check response."""

    status:       str
    appName:      str
    version:      str
    environment:  str
    modelLoaded:  bool
    mlflowStatus: str
    timestamp:    datetime = Field(default_factory=datetime.utcnow)