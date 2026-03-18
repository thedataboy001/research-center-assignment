# app/main.py
# FastAPI application factory.
# Uses the lifespan context manager for startup/shutdown logic —
# the modern FastAPI pattern replacing deprecated @app.on_event handlers.

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes import predict, train, health
from app.api.routes.predict import load_model_artefacts

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup & shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.

    On startup:
        - Logs configuration summary
        - Attempts to load saved model artefacts into memory
          (graceful — if artefacts not found, app starts in unloaded state)

    On shutdown:
        - Logs graceful shutdown message
        - Could be extended to flush metrics, close DB connections, etc.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"  Environment  : {settings.app_env}")
    logger.info(f"  Debug mode   : {settings.debug}")
    logger.info(f"  MLflow URI   : {settings.mlflow_tracking_uri}")
    logger.info(f"  Data path    : {settings.data_path}")
    logger.info(f"  Artefact dir : {settings.artefact_dir}")
    logger.info("=" * 60)

    # Attempt to load previously trained model artefacts
    load_model_artefacts()

    yield   # Application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info(f"Shutting down {settings.app_name}...")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Separating creation into a factory function makes
    the app easier to test — tests can call create_app()
    to get a fresh instance with test configuration.
    """
    app = FastAPI(
        title       = settings.app_name,
        version     = settings.app_version,
        description = (
            "## Research Center Quality Classification API\n\n"
            "End-to-end ML pipeline for classifying UK research centers "
            "into **Premium**, **Standard**, or **Basic** quality tiers "
            "using K-Means clustering.\n\n"
            "### Pipeline Steps\n"
            "1. **Data Loading**: Ingests `data/research_centers.csv`\n"
            "2. **Data Validation**: range checks, missing values, "
            "type validation\n"
            "3. **Preprocessing**: StandardScaler normalisation\n"
            "4. **Training**: K-Means (K=3, k-means++ initialisation)\n"
            "5. **Evaluation**: silhouette score, per-cluster metrics\n"
            "6. **MLflow Tracking**: params, metrics, artefacts, "
            "model registry\n"
            "7. **Prediction**: single and batch classification endpoints\n\n"
            "### Tier Definitions\n"
            "| Tier | Description |\n"
            "|---|---|\n"
            "| **Premium** | Highest internal capacity + rich external "
            "healthcare access |\n"
            "| **Standard** | Moderate facilities and access |\n"
            "| **Basic** | Limited internal capacity and external access |\n"
        ),
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        openapi_url = "/openapi.json",
    )

    # ── CORS Middleware ───────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # Restrict in production
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(
        request: Request, call_next
    ):
        """
        Adds X-Process-Time header to every response.
        Useful for latency monitoring.
        """
        start_time = time.time()
        response   = await call_next(request)
        process_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Process-Time-ms"] = str(process_ms)
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catches any unhandled exception and returns a structured JSON error.
        Prevents raw Python stack traces from leaking to API consumers.
        """
        logger.exception(f"Unhandled exception on {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error":   "Internal server error",
                "details": str(exc) if settings.debug else
                           "An unexpected error occurred.",
                "path":    str(request.url),
            },
        )

    # ── Register routers ──────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(train.router)
    app.include_router(predict.router)

    # ── Root endpoint ─────────────────────────────────────────────────────────
    @app.get(
        "/",
        tags=["Root"],
        summary="API root — links to docs and health",
    )
    async def root() -> dict:
        return {
            "app":         settings.app_name,
            "version":     settings.app_version,
            "environment": settings.app_env,
            "docs":        "/docs",
            "health":      "/health",
            "train":       "POST /train",
            "predict":     "POST /predict",
            "batch":       "POST /predict/batch",
        }

    logger.info("FastAPI application configured successfully")
    return app


# ── Module-level app instance ─────────────────────────────────────────────────
# This is what uvicorn imports:  uvicorn app.main:app
app = create_app()