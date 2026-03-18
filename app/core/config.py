# ─────────────────────────────────────────────────────────────────────────────
# core/config.py
# Centralised configuration using Pydantic Settings.
# All values can be overridden by environment variables or a .env file.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    Pydantic Settings automatically reads from the environment,
    meaning no configuration is hard-coded in the application logic.
    """

    # ── Application ──────────────────────────────────────────
    app_name:    str = "ResearchCenterQualityAPI"
    app_version: str = "1.0.0"
    app_env:     str = "development"
    debug:       bool = True

    # ── Data ─────────────────────────────────────────────────
    data_path: str = "data/research_centers.csv"

    # Stored as comma-separated string in .env; parsed into list
    selected_features: str = (
        "internalFacilitiesCount,"
        "hospitals_10km,"
        "pharmacies_10km,"
        "facilityDiversity_10km,"
        "facilityDensity_10km"
    )

    @property
    def features_list(self) -> List[str]:
        """Parse the comma-separated features string into a Python list."""
        return [f.strip() for f in self.selected_features.split(",")]

    # ── Model hyperparameters ─────────────────────────────────
    n_clusters:   int = 3
    n_init:       int = 50
    max_iter:     int = 300
    random_state: int = 42

    # ── MLflow ───────────────────────────────────────────────
    mlflow_tracking_uri:    str = "http://localhost:5000"
    mlflow_experiment_name: str = "research_center_quality"
    mlflow_model_name:      str = "kmeans_quality_classifier"

    # ── Artefact storage ─────────────────────────────────────
    artefact_dir: str = "artefacts"

    # ── Validation boundaries (used in data_validator.py) ────
    # These match the domain knowledge established in EDA
    min_facilities:  int   = 0
    max_facilities:  int   = 100
    min_hospitals:   int   = 0
    max_hospitals:   int   = 50
    min_pharmacies:  int   = 0
    max_pharmacies:  int   = 100
    min_diversity:   float = 0.0
    max_diversity:   float = 1.0
    min_density:     float = 0.0
    max_density:     float = float("inf")   # no upper bound for density

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # ignore any unrecognised env vars
    )


# ── Singleton instance ────────────────────────────────────────────────────────
settings = Settings()

# Ensure artefact directory exists at startup
os.makedirs(settings.artefact_dir, exist_ok=True)


if __name__ == "__main__":
    print("Running config test...\n")

    print("App Name:", settings.app_name)
    print("Environment:", settings.app_env)
    print("Debug Mode:", settings.debug)

    print("\nData Path:", settings.data_path)

    print("\nParsed Features List:")
    print(settings.features_list)

    print("\nModel Parameters:")
    print({
        "n_clusters": settings.n_clusters,
        "n_init": settings.n_init,
        "max_iter": settings.max_iter,
        "random_state": settings.random_state,
    })

    print("\nMLflow Config:")
    print({
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment": settings.mlflow_experiment_name,
    })

    print("\nArtefact Directory Exists:", os.path.exists(settings.artefact_dir))


# To run this test, execute `uv run -m app.core.config` from the project root.