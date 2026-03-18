# api/routes/health.py
# Health check endpoint — used by Docker, load balancers, and monitoring tools
# to verify the service is alive and correctly configured.

from fastapi import APIRouter
from app.schemas.output import HealthOutput
from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes.predict import model_state
import mlflow

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/",
    response_model=HealthOutput,
    summary="Service health check",
    description=(
        "Returns application status, model load state, "
        "and MLflow connectivity. "
        "Returns HTTP 200 if healthy, HTTP 503 if degraded."
    ),
)
async def health_check() -> HealthOutput:
    """
    Health check for load balancers and container orchestration.

    Checks:
        - Application is running
        - Model artefacts are loaded
        - MLflow tracking server is reachable

    Returns:
        HealthOutput: always HTTP 200 (status field indicates health)
    """
    # ── Check MLflow connectivity ─────────────────────────────────────────────
    mlflow_status = "unreachable"
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.search_experiments()
        mlflow_status = "connected"
    except Exception as e:
        logger.warning(f"MLflow health check failed: {e}")
        mlflow_status = f"unreachable — {str(e)[:60]}"

    status = (
        "healthy"
        if model_state.is_loaded and mlflow_status == "connected"
        else "degraded"
    )

    logger.info(
        f"Health check: status={status}, "
        f"model_loaded={model_state.is_loaded}, "
        f"mlflow={mlflow_status}"
    )

    return HealthOutput(
        status       = status,
        appName      = settings.app_name,
        version      = settings.app_version,
        environment  = settings.app_env,
        modelLoaded  = model_state.is_loaded,
        mlflowStatus = mlflow_status,
    )


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Kubernetes/Docker readiness probe. "
        "Returns 200 only when model is loaded and ready for predictions. "
        "Returns 503 otherwise."
    ),
)
async def readiness_probe() -> dict:
    """
    Readiness probe: returns 503 if model is not loaded.
    Unlike /health, this will fail hard if the model is unavailable.
    """
    from fastapi import HTTPException

    if not model_state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: model not loaded. Call POST /train first.",
        )
    return {"status": "ready"}


@router.get(
    "/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe: confirms process is alive.",
)
async def liveness_probe() -> dict:
    """
    Liveness probe: always returns 200 if the process is running.
    Does not check model or MLflow state.
    """
    return {"status": "alive"}