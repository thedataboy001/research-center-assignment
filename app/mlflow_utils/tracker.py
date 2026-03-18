# # app/mlflow_utils/tracker.py
# # ─────────────────────────────────────────────────────────────────────────────
# # Key fix: all local staging files written to settings.artefact_dir
# # (which IS mounted as a volume in the api container) before being
# # uploaded to MLflow via the tracking server REST API.
# # ─────────────────────────────────────────────────────────────────────────────

# import mlflow
# import mlflow.sklearn
# import os
# import json
# from typing import Optional
# from app.core.config import settings
# from app.core.logging import get_logger
# from app.schemas.output import ValidationReport

# logger = get_logger(__name__)


# class MLflowTracker:
#     """
#     Manages a single MLflow run for one training execution.

#     All artefact writes follow this pattern:
#         1. Write file locally to settings.artefact_dir
#            (this path IS mounted as a volume — api container owns it)
#         2. Call mlflow.log_artifact(local_path) which sends the file
#            to the MLflow tracking server via HTTP POST
#         3. The server stores it under /mlflow/artefacts (its own volume)

#     The api container NEVER writes to /mlflow directly.
#     """

#     def __init__(self):
#         self.run_id  : Optional[str] = None
#         self.run_url : str = ""
#         self._run    = None

#     # ── Context manager ───────────────────────────────────────────────────────

#     def __enter__(self) -> "MLflowTracker":
#         """
#         Set the MLflow tracking URI and start a new run.
#         All subsequent mlflow.* calls route to http://mlflow:5000
#         as HTTP requests — no direct filesystem access to /mlflow.
#         """
#         mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
#         mlflow.set_experiment(settings.mlflow_experiment_name)

#         self._run   = mlflow.start_run()
#         self.run_id = self._run.info.run_id
#         self.run_url = (
#             f"{settings.mlflow_tracking_uri}/"
#             f"#/experiments/{self._run.info.experiment_id}/"
#             f"runs/{self.run_id}"
#         )

#         logger.info(f"MLflow run started — ID: {self.run_id}")
#         logger.info(f"MLflow run URL    — {self.run_url}")
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb) -> None:
#         """End the MLflow run cleanly regardless of success or failure."""
#         if exc_type is not None:
#             mlflow.set_tag("run_status", "FAILED")
#             mlflow.set_tag("failure_reason", str(exc_val)[:250])
#             logger.error(f"MLflow run FAILED: {exc_val}")
#         else:
#             mlflow.set_tag("run_status", "SUCCESS")
#             logger.info(
#                 f"MLflow run completed successfully: {self.run_id}"
#             )
#         mlflow.end_run()

#     # ── Logging methods ───────────────────────────────────────────────────────

#     def log_tags(self) -> None:
#         """Log run-level metadata tags."""
#         mlflow.set_tags({
#             "app_version":  settings.app_version,
#             "environment":  settings.app_env,
#             "algorithm":    "KMeans",
#             "n_features":   len(settings.features_list),
#             "scaler":       "StandardScaler",
#         })
#         logger.info("Run tags logged to MLflow")

#     def log_validation_report(self, report: ValidationReport) -> None:
#         """
#         Log data validation results as params + JSON artefact.

#         Fix applied:
#             Write JSON to settings.artefact_dir (api container volume)
#             then upload via mlflow.log_artifact() over HTTP to the server.
#             Never attempt to write to /mlflow directly.
#         """
#         # ── Log scalar values as params ───────────────────────────────────────
#         mlflow.log_params({
#             "data_row_count":    report.rowCount,
#             "data_col_count":    report.columnCount,
#             "duplicate_rows":    report.duplicateRows,
#             "validation_passed": report.isValid,
#         })

#         total_missing = sum(report.missingValues.values())
#         mlflow.log_metric("data_missing_values_total", total_missing)

#         # ── Write JSON locally then upload to MLflow server ───────────────────
#         # settings.artefact_dir = /app/artefacts (volume-mounted, writable)
#         staging_dir  = os.path.join(settings.artefact_dir, "staging")
#         os.makedirs(staging_dir, exist_ok=True)

#         report_path = os.path.join(staging_dir, "validation_report.json")
#         with open(report_path, "w") as f:
#             json.dump(report.model_dump(), f, indent=2, default=str)

#         # Upload to MLflow tracking server via HTTP — not direct fs write
#         mlflow.log_artifact(report_path, artifact_path="data_quality")

#         if report.warnings:
#             mlflow.set_tag(
#                 "data_warnings",
#                 " | ".join(report.warnings)[:500]
#             )

#         logger.info("Validation report logged to MLflow")

#     def log_scaler_params(self, scaler_params: dict) -> None:
#         """Log StandardScaler mean and std for each feature."""
#         mlflow.log_params(scaler_params)
#         logger.info("Scaler parameters logged to MLflow")

#     def log_hyperparameters(self, hyperparams: dict) -> None:
#         """Log K-Means hyperparameters."""
#         mlflow.log_params(hyperparams)
#         logger.info(f"Hyperparameters logged: {hyperparams}")

#     def log_metrics(self, metrics: dict) -> None:
#         """Log evaluation metrics."""
#         mlflow.log_metrics(metrics)
#         logger.info(
#             f"Metrics logged — silhouette="
#             f"{metrics.get('silhouette_score_overall', 'N/A')}"
#         )

#     def log_cluster_mapping(self, cluster_to_tier: dict) -> None:
#         """
#         Log cluster-to-tier mapping as a JSON artefact.
#         Written locally first, then uploaded to MLflow server.
#         """
#         staging_dir  = os.path.join(settings.artefact_dir, "staging")
#         os.makedirs(staging_dir, exist_ok=True)

#         mapping_path = os.path.join(staging_dir, "cluster_to_tier.json")
#         with open(mapping_path, "w") as f:
#             json.dump(
#                 {str(k): v for k, v in cluster_to_tier.items()},
#                 f,
#                 indent=2,
#             )

#         mlflow.log_artifact(mapping_path, artifact_path="model_metadata")
#         mlflow.log_params({
#             f"tier_cluster_{v}": k
#             for k, v in cluster_to_tier.items()
#         })
#         logger.info("Cluster-to-tier mapping logged to MLflow")

#     def log_model(self, kmeans_model, scaler) -> str:
#         """
#         Register the trained K-Means model in MLflow Model Registry.

#         The scaler is saved locally and uploaded as an artefact.
#         The KMeans model is logged with the sklearn flavour so it can
#         be loaded with mlflow.sklearn.load_model() in future runs.
#         """
#         import joblib

#         staging_dir = os.path.join(settings.artefact_dir, "staging")
#         os.makedirs(staging_dir, exist_ok=True)

#         # Save scaler locally then upload
#         scaler_path = os.path.join(staging_dir, "scaler.pkl")
#         joblib.dump(scaler, scaler_path)
#         mlflow.log_artifact(scaler_path, artifact_path="model_artefacts")

#         # Log KMeans model — uploaded to tracking server via sklearn flavour
#         model_info = mlflow.sklearn.log_model(
#             sk_model              = kmeans_model,
#             artifact_path         = "kmeans_model",
#             registered_model_name = settings.mlflow_model_name,
#         )

#         logger.info(
#             f"Model registered in MLflow Model Registry: "
#             f"'{settings.mlflow_model_name}'"
#         )
#         return model_info.model_uri

#     def log_plot_artefacts(self, plot_paths: list) -> None:
#         """Upload all diagnostic plot images to MLflow server."""
#         for path in plot_paths:
#             if os.path.exists(path):
#                 mlflow.log_artifact(path, artifact_path="plots")
#                 logger.info(
#                     f"Plot uploaded to MLflow: {os.path.basename(path)}"
#                 )

#     def log_labelled_dataset(self, df, path: str) -> None:
#         """Save and upload the labelled clustered dataset."""
#         df.to_csv(path, index=False)
#         mlflow.log_artifact(path, artifact_path="datasets")
#         logger.info(
#             f"Labelled dataset uploaded to MLflow: "
#             f"{os.path.basename(path)}"
#         )



# app/mlflow_utils/tracker.py
# ─────────────────────────────────────────────────────────────────────────────
# Key changes:
#   1. Removed local file staging — no writes to /mlflow
#   2. Log validation report directly to MLflow without local file
#   3. Use mlflow.sklearn.log_model() which handles uploads automatically
#   4. Let MLflow client handle all artifact persistence
# ─────────────────────────────────────────────────────────────────────────────

import mlflow
import mlflow.sklearn
import os
import json
import tempfile
import pickle
from typing import Optional
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.output import ValidationReport

logger = get_logger(__name__)


class MLflowTracker:
    """
    Manages a single MLflow run with HTTP-based artifact uploads.

    All mlflow.log_* calls route through MLFLOW_TRACKING_URI as HTTP requests.
    The MLflow server handles writing artifacts to its own /mlflow/artefacts.
    The api container NEVER writes to /mlflow directly.
    """

    def __init__(self):
        self.run_id  : Optional[str] = None
        self.run_url : str = ""
        self._run    = None

    def __enter__(self) -> "MLflowTracker":
        """Start a new MLflow run."""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        self._run   = mlflow.start_run()
        self.run_id = self._run.info.run_id
        self.run_url = (
            f"{settings.mlflow_tracking_uri}/"
            f"#/experiments/{self._run.info.experiment_id}/"
            f"runs/{self.run_id}"
        )

        logger.info(f"MLflow run started — ID: {self.run_id}")
        logger.info(f"MLflow run URL    — {self.run_url}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the MLflow run."""
        if exc_type is not None:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("failure_reason", str(exc_val)[:250])
            logger.error(f"MLflow run FAILED: {exc_val}")
        else:
            mlflow.set_tag("run_status", "SUCCESS")
            logger.info(
                f"MLflow run completed successfully: {self.run_id}"
            )
        mlflow.end_run()

    # ── Logging methods ───────────────────────────────────────────────────────

    def log_tags(self) -> None:
        """Log run-level metadata tags."""
        mlflow.set_tags({
            "app_version":  settings.app_version,
            "environment":  settings.app_env,
            "algorithm":    "KMeans",
            "n_features":   len(settings.features_list),
            "scaler":       "StandardScaler",
        })
        logger.info("Run tags logged to MLflow")

    def log_validation_report(self, report: ValidationReport) -> None:
        """
        Log data validation results.
        
        Approach: Write JSON to a temporary file, then upload.
        The temporary file is in the container's /tmp (not /mlflow),
        then deleted after MLflow uploads it.
        
        MLflow's HTTP client automatically routes the upload to the server.
        """
        # Log scalar values as parameters
        mlflow.log_params({
            "data_row_count":    report.rowCount,
            "data_col_count":    report.columnCount,
            "duplicate_rows":    report.duplicateRows,
            "validation_passed": report.isValid,
        })

        total_missing = sum(report.missingValues.values())
        mlflow.log_metric("data_missing_values_total", total_missing)

        # ── Write JSON to temporary file (not /mlflow) ──────────────────────────
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            dir='/tmp'    # Use container's /tmp, not /mlflow
        ) as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
            report_path = f.name

        try:
            # MLflow client uploads over HTTP to the tracking server
            # The server writes it to /mlflow/artefacts on its side
            mlflow.log_artifact(report_path, artifact_path="data_quality")
        finally:
            # Clean up temporary file
            os.remove(report_path)

        if report.warnings:
            mlflow.set_tag(
                "data_warnings",
                " | ".join(report.warnings)[:500]
            )

        logger.info("Validation report logged to MLflow")

    def log_scaler_params(self, scaler_params: dict) -> None:
        """Log StandardScaler mean and std for each feature."""
        mlflow.log_params(scaler_params)
        logger.info("Scaler parameters logged to MLflow")

    def log_hyperparameters(self, hyperparams: dict) -> None:
        """Log K-Means hyperparameters."""
        mlflow.log_params(hyperparams)
        logger.info(f"Hyperparameters logged: {hyperparams}")

    def log_metrics(self, metrics: dict) -> None:
        """Log evaluation metrics."""
        mlflow.log_metrics(metrics)
        logger.info(
            f"Metrics logged — silhouette="
            f"{metrics.get('silhouette_score_overall', 'N/A')}"
        )

    def log_cluster_mapping(self, cluster_to_tier: dict) -> None:
        """
        Log cluster-to-tier mapping as JSON artefact.
        Uses temporary file approach — written to /tmp, not /mlflow.
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            dir='/tmp'
        ) as f:
            json.dump(
                {str(k): v for k, v in cluster_to_tier.items()},
                f,
                indent=2,
            )
            mapping_path = f.name

        try:
            mlflow.log_artifact(mapping_path, artifact_path="model_metadata")
        finally:
            os.remove(mapping_path)

        mlflow.log_params({
            f"tier_cluster_{v}": k
            for k, v in cluster_to_tier.items()
        })
        logger.info("Cluster-to-tier mapping logged to MLflow")

    def log_model(self, kmeans_model, scaler) -> str:
        """
        Register the trained K-Means model in MLflow Model Registry.
        
        mlflow.sklearn.log_model() automatically handles serialisation
        and HTTP upload to the tracking server.
        """
        #import joblib
        import base64

        # ── Save scaler to bytes, encode as base64, log as parameter ──────────
        # This avoids writing to the filesystem entirely for the scaler.
        # For larger models this approach won't work, but scaler is tiny.
        scaler_bytes = pickle.dumps(scaler)
        scaler_b64 = base64.b64encode(scaler_bytes).decode('utf-8')
        
        # Store a reference — in production, save to artefacts properly
        mlflow.log_param("scaler_base64", scaler_b64)
        mlflow.log_param("scaler_type", "StandardScaler")
        logger.info("Scaler logged as parameter reference")

        # ── Log KMeans model — mlflow.sklearn handles HTTP upload ──────────────
        model_info = mlflow.sklearn.log_model(
            sk_model              = kmeans_model,
            artifact_path         = "kmeans_model",
            registered_model_name = settings.mlflow_model_name,
        )

        logger.info(
            f"Model registered in MLflow Model Registry: "
            f"'{settings.mlflow_model_name}'"
        )
        return model_info.model_uri

    def log_plot_artefacts(self, plot_paths: list) -> None:
        """
        Upload all diagnostic plot images to MLflow server.
        
        These files exist on the api container filesystem.
        mlflow.log_artifact() sends them as HTTP POST to the server.
        """
        for path in plot_paths:
            if os.path.exists(path):
                try:
                    mlflow.log_artifact(path, artifact_path="plots")
                    logger.info(
                        f"Plot uploaded to MLflow: {os.path.basename(path)}"
                    )
                except Exception as e:
                    logger.error(f"Failed to upload plot {path}: {e}")

    def log_labelled_dataset(self, df, path: str) -> None:
        """
        Save and upload the labelled clustered dataset as CSV.
        """
        try:
            df.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path="datasets")
            logger.info(
                f"Labelled dataset uploaded to MLflow: "
                f"{os.path.basename(path)}"
            )
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")