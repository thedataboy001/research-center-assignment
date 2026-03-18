# api/routes/predict.py
# Prediction endpoint — classifies a research center into a quality tier.
# Loads model artefacts once at startup using FastAPI lifespan,
# then reuses them for every request without reloading from disk.

from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated
import mlflow
import os

from app.schemas.input  import ResearchCenterInput, BatchResearchCenterInput
from app.schemas.output import PredictionOutput, BatchPredictionOutput
from app.pipeline.preprocessor import Preprocessor
from app.pipeline.trainer import Trainer
from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


# Model state: loaded once when the application starts

class ModelState:
    """
    Singleton that holds loaded model artefacts in memory.
    Avoids reloading from disk on every request.

    Attributes:
        preprocessor  : Fitted StandardScaler wrapper
        trainer       : Fitted KMeans wrapper with tier mapping
        run_id        : MLflow run ID of the loaded model
        is_loaded     : Whether artefacts have been successfully loaded
    """
    preprocessor : Preprocessor | None = None
    trainer      : Trainer      | None = None
    run_id       : str                 = "unknown"
    is_loaded    : bool                = False


model_state = ModelState()


def load_model_artefacts() -> None:
    """
    Load model, scaler, and tier mapping from disk into ModelState.
    Called once at application startup via the lifespan handler in main.py.

    If artefacts are not found, the app starts in an unloaded state —
    the /predict endpoint will return 503 until /train is called.
    """
    artefact_dir = settings.artefact_dir
    model_path   = os.path.join(artefact_dir, "kmeans_model.pkl")
    scaler_path  = os.path.join(artefact_dir, "scaler.pkl")
    mapping_path = os.path.join(artefact_dir, "cluster_to_tier.pkl")
    

    # Check all required artefacts exist
    missing = [
        p for p in [model_path, scaler_path, mapping_path]
        if not os.path.exists(p)
    ]

    if missing:
        logger.warning(
            f"Model artefacts not found: {missing}. "
            "Call POST /train to train the model first."
        )
        model_state.is_loaded = False
        return

    try:
        # Load preprocessor (scaler)
        preprocessor = Preprocessor()
        preprocessor.load_scaler(scaler_path)

        # Load trainer (KMeans model + tier mapping)
        trainer = Trainer()
        trainer.load(
            model_path   = model_path,
            mapping_path = mapping_path,
        )

        model_state.preprocessor = preprocessor
        model_state.trainer      = trainer
        model_state.is_loaded    = True

        logger.info(
            "Model artefacts loaded successfully at startup.\n"
            f"  Tier mapping: {trainer.cluster_to_tier}"
        )

    except Exception as e:
        logger.error(f"Failed to load model artefacts: {e}")
        model_state.is_loaded = False


def get_model_state() -> ModelState:
    """
    FastAPI dependency: injects ModelState into endpoints.
    Returns 503 if model has not been trained yet.
    """
    load_model_artefacts() # Ensure artefacts are loaded (no-op if already loaded)

    if not model_state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "fix":   "Call POST /train to train the model first, "
                         "then retry this request.",
            },
        )
    return model_state


# Endpoints

@router.post(
    "/",
    response_model=PredictionOutput,
    summary="Classify a single research center",
    description=(
        "Accepts five feature values for a research center and "
        "returns its predicted quality tier: Premium, Standard, or Basic.\n\n"
        "Input is validated by Pydantic before reaching the model.\n"
        "Features are scaled using training-time StandardScaler statistics."
    ),
)
async def predict_single(
    center: ResearchCenterInput,
    state:  Annotated[ModelState, Depends(get_model_state)],
) -> PredictionOutput:
    """
    Single-center prediction endpoint.

    Pipeline:
        1. Pydantic validates input types and ranges (automatic)
        2. Convert to DataFrame
        3. Scale with training-time scaler
        4. K-Means assigns nearest centroid
        5. Map cluster number → tier name
        6. Compute confidence score
        7. Return structured response

    Args:
        center: Validated ResearchCenterInput from request body
        state:  Injected ModelState (raises 503 if model not loaded)

    Returns:
        PredictionOutput with tier, cluster, confidence, and metadata
    """
    logger.info(
        f"Prediction request: "
        f"internalFacilitiesCount={center.internalFacilitiesCount}, "
        f"hospitals={center.hospitals_10km}, "
        f"pharmacies={center.pharmacies_10km}, "
        f"diversity={center.facilityDiversity_10km}, "
        f"density={center.facilityDensity_10km}"
    )

    try:
        # ── Step 1: Convert Pydantic model to DataFrame ───────────────────────
        input_df = center.to_dataframe()

        # ── Step 2: Scale using training-time statistics ──────────────────────
        X_scaled = state.preprocessor.transform(input_df)

        # ── Step 3: Predict cluster and confidence ────────────────────────────
        cluster_labels, confidence_scores = state.trainer.predict(X_scaled)

        cluster_number  = int(cluster_labels[0])
        confidence      = float(confidence_scores[0])
        predicted_tier  = state.trainer.cluster_to_tier[cluster_number]

        logger.info(
            f"Prediction result: "
            f"cluster={cluster_number}, "
            f"tier={predicted_tier}, "
            f"confidence={confidence:.4f}"
        )

        return PredictionOutput(
            predictedCategory = predicted_tier,
            clusterNumber     = cluster_number,
            confidence        = confidence,
            modelVersion      = state.run_id,
        )

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Prediction failed",
                "details": str(e),
            },
        )


@router.post(
    "/batch",
    response_model=BatchPredictionOutput,
    summary="Classify multiple research centers in one request",
    description=(
        "Accepts a list of research centers (max 1000) and returns "
        "quality tier predictions for each one. "
        "More efficient than repeated single calls for bulk classification."
    ),
)
async def predict_batch(
    payload: BatchResearchCenterInput,
    state:   Annotated[ModelState, Depends(get_model_state)],
) -> BatchPredictionOutput:
    """
    Batch prediction endpoint.

    Processes all centers in a single vectorised operation —
    more efficient than looping through individual /predict calls.

    Args:
        payload: List of ResearchCenterInput objects (1–1000)
        state:   Injected ModelState

    Returns:
        BatchPredictionOutput with all predictions and a tier summary
    """
    logger.info(
        f"Batch prediction request: {len(payload.centers)} centers"
    )

    try:
        import pandas as pd

        # ── Step 1: Convert all inputs to a single DataFrame ──────────────────
        batch_df = pd.concat(
            [center.to_dataframe() for center in payload.centers],
            ignore_index=True,
        )

        # ── Step 2: Scale entire batch at once ────────────────────────────────
        X_scaled = state.preprocessor.transform(batch_df)

        # ── Step 3: Predict all clusters at once ──────────────────────────────
        cluster_labels, confidence_scores = state.trainer.predict(X_scaled)

        # ── Step 4: Build individual prediction outputs ───────────────────────
        predictions = []
        tier_counts : dict[str, int] = {}

        for i in range(len(payload.centers)):
            cluster_number = int(cluster_labels[i])
            confidence     = float(confidence_scores[i])
            predicted_tier = state.trainer.cluster_to_tier[cluster_number]

            tier_counts[predicted_tier] = (
                tier_counts.get(predicted_tier, 0) + 1
            )

            predictions.append(
                PredictionOutput(
                    predictedCategory = predicted_tier,
                    clusterNumber     = cluster_number,
                    confidence        = confidence,
                    modelVersion      = state.run_id,
                )
            )

        logger.info(
            f"Batch prediction complete: "
            f"tier distribution: {tier_counts}"
        )

        return BatchPredictionOutput(
            predictions = predictions,
            totalCount  = len(predictions),
            tierSummary = tier_counts,
        )

    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Batch prediction failed",
                "details": str(e),
            },
        )