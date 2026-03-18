# api/routes/train.py
# Training endpoint — triggers the full ML pipeline end-to-end.

from fastapi import APIRouter, HTTPException
from app.schemas.output import TrainingOutput
from app.pipeline.data_loader import DataLoader
from app.pipeline.data_validator import DataValidator
from app.pipeline.preprocessor import Preprocessor
from app.pipeline.trainer import Trainer
from app.pipeline.evaluator import Evaluator
from app.mlflow_utils.tracker import MLflowTracker
from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes.predict import load_model_artefacts

import os 


logger = get_logger(__name__)
router = APIRouter(prefix="/train", tags=["Training"])


@router.post(
    "/",
    response_model=TrainingOutput,
    summary="Run full ML pipeline training",
    description=(
        "Runs the complete ML pipeline:\n"
        "1.  Load data from CSV\n"
        "2.  Validate data quality\n"
        "3.  Preprocess: feature selection & StandardScaler\n"
        "4.  Train K-Means (K=3) with k-means++ initialisation\n"
        "5.  Evaluate: silhouette score & per-cluster metrics\n"
        "6.  Generate diagnostic plots\n"
        "7.  Log all params, metrics & artefacts to MLflow\n"
        "8.  Register model in MLflow Model Registry\n"
        "9.  Save model artefacts to disk for API inference"
    ),
)
async def train_model() -> TrainingOutput:
    """
    Executes the full end-to-end training pipeline.

    Returns:
        TrainingOutput containing MLflow run ID, silhouette score,
        inertia, cluster sizes, and tier mapping.

    Raises:
        HTTPException 422: Data validation failed
        HTTPException 500: Unexpected pipeline error
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE: STARTED")
    logger.info("=" * 60)

    try:
        with MLflowTracker() as tracker:

            # ── Step 1: Log run metadata tags ─────────────────────────────────
            logger.info("Step 1/9: Logging run metadata")
            tracker.log_tags()

            # ── Step 2: Load data ─────────────────────────────────────────────
            logger.info("Step 2/9: Loading data")
            loader = DataLoader(data_path=settings.data_path)
            raw_df = loader.load()

            # ── Step 3: Validate data ─────────────────────────────────────────
            logger.info("Step 3/9: Validating data")
            validator         = DataValidator(raw_df)
            try:
                clean_df, report  = validator.validate()
            except ValueError as ve:
                # Validation errors are user-facing — 422 Unprocessable Entity
                logger.error(f"Data validation failed: {ve}")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error":   "Data validation failed",
                        "details": str(ve),
                    },
                )

            # Log validation results to MLflow
            tracker.log_validation_report(report)

            # ── Step 4: Preprocess — scale features ───────────────────────────
            logger.info("Step 4/9: Preprocessing features")
            preprocessor          = Preprocessor()
            X_scaled, X_scaled_df = preprocessor.fit_transform(clean_df)

            # Log scaler mean/std parameters to MLflow
            tracker.log_scaler_params(preprocessor.get_scaler_params())

            # ── Step 5: Train K-Means ─────────────────────────────────────────
            logger.info("Step 5/9: Training K-Means model")
            trainer = Trainer()
            trainer.train(
                X_scaled   = X_scaled,
                X_original = clean_df[settings.features_list],
            )

            # Log hyperparameters to MLflow
            hyperparams = trainer.get_hyperparameters()
            tracker.log_hyperparameters(hyperparams)

            # Log cluster-to-tier mapping
            tracker.log_cluster_mapping(trainer.cluster_to_tier)

            # ── Step 6: Evaluate ──────────────────────────────────────────────
            logger.info("Step 6/9: Evaluating model")
            evaluator = Evaluator(
                X_scaled        = X_scaled,
                labels          = trainer.labels_,
                centroids       = trainer.model.cluster_centers_,
                cluster_to_tier = trainer.cluster_to_tier,
                feature_names   = settings.features_list,
            )
            metrics = evaluator.compute_all_metrics()

            # Add inertia to metrics
            metrics["inertia"]              = round(
                float(trainer.model.inertia_), 6
            )
            metrics["n_iter_to_converge"]   = int(trainer.model.n_iter_)

            # Log metrics to MLflow
            tracker.log_metrics(metrics)

            # ── Step 7: Generate diagnostic plots ─────────────────────────────
            logger.info("Step 7/9: Generating diagnostic plots")
            plot_paths = evaluator.generate_all_plots()
            tracker.log_plot_artefacts(plot_paths)

            # ── Step 8: Save model artefacts to disk ──────────────────────────
            logger.info("Step 8/9: Saving model artefacts")
            artefact_dir  = settings.artefact_dir
            model_path    = os.path.join(artefact_dir, "kmeans_model.pkl")
            mapping_path  = os.path.join(artefact_dir, "cluster_to_tier.pkl")
            scaler_path   = os.path.join(artefact_dir, "scaler.pkl")

            trainer.save(
                model_path   = model_path,
                mapping_path = mapping_path,
            )
            preprocessor.save_scaler(path=scaler_path)

            # ── Step 9: Register model in MLflow Model Registry ───────────────
            logger.info("Step 9/9: Registering model in MLflow")
            model_uri = tracker.log_model(
                kmeans_model = trainer.model,
                scaler       = preprocessor.scaler,
            )

            # Log the final labelled dataset as a CSV artefact
            clean_df["cluster"]     = trainer.labels_
            clean_df["qualityTier"] = clean_df["cluster"].map(
                trainer.cluster_to_tier
            )
            labelled_path = os.path.join(
                artefact_dir, "research_centers_clustered.csv"
            )
            tracker.log_labelled_dataset(clean_df, labelled_path)

            logger.info("Refreshing in-memory model state with new artefacts")
            load_model_artefacts()

            # ── Build response ─────────────────────────────────────────────────
            logger.info("=" * 60)
            logger.info(
                f"TRAINING PIPELINE: COMPLETED SUCCESSFULLY\n"
                f"  MLflow Run ID      : {tracker.run_id}\n"
                f"  Silhouette Score   : "
                f"{metrics['silhouette_score_overall']:.4f}\n"
                f"  Inertia            : {metrics['inertia']:.4f}\n"
                f"  Tier mapping       : {trainer.cluster_to_tier}"
            )
            logger.info("=" * 60)

            return TrainingOutput(
                status          = "success",
                mlflowRunId     = tracker.run_id,
                mlflowRunUrl    = tracker.run_url,
                silhouetteScore = metrics["silhouette_score_overall"],
                inertia         = metrics["inertia"],
                clusterSizes    = trainer.get_cluster_sizes(),
                tierMapping     = trainer.cluster_to_tier,
                featuresUsed    = settings.features_list,
                hyperparameters = hyperparams,
            )

    except HTTPException:
        # Re-raise HTTPExceptions without wrapping them
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in training pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "Training pipeline failed unexpectedly",
                "details": str(e),
            },
        )
    

