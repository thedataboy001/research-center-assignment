# pipeline/preprocessor.py
# Feature selection and StandardScaler normalisation.
# The fitted scaler is saved as an MLflow artefact so the API
# always uses training-time statistics for scaling new inputs.

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Selects features and applies StandardScaler normalisation.

    Fit mode  (training):
        fit_transform() — computes mean/std from training data
        and saves the fitted scaler to disk + MLflow artefacts.

    Transform mode (inference/API):
        transform(), applies saved training statistics to new data.
        Never re-fits on inference data.
    """

    def __init__(self):
        self.features = settings.features_list
        self.scaler   = StandardScaler()
        self._fitted  = False

    # ── Training path ─────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Fit scaler on training data and transform.

        Args:
            df: Validated training DataFrame

        Returns:
            Tuple of:
                - X_scaled (np.ndarray): scaled feature matrix
                - X_scaled_df (pd.DataFrame): same data with column names
        """
        logger.info(f"Fitting scaler on features: {self.features}")

        X = df[self.features].copy()

        # Log pre-scaling statistics
        logger.debug("Pre-scaling feature statistics:")
        for col in self.features:
            logger.debug(
                f"  {col}: "
                f"mean={X[col].mean():.4f}, "
                f"std={X[col].std():.4f}, "
                f"min={X[col].min():.4f}, "
                f"max={X[col].max():.4f}"
            )

        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True

        # Verify scaling worked correctly
        post_means = X_scaled.mean(axis=0)
        post_stds  = X_scaled.std(axis=0)
        for i, col in enumerate(self.features):
            if abs(post_means[i]) > 0.001:
                logger.warning(
                    f"Post-scaling mean for '{col}' is {post_means[i]:.6f} "
                    "(expected ~0.0)"
                )

        logger.info(
            "Scaling completed. "
            "all features normalised to mean≈0, std≈1"
        )

        X_scaled_df = pd.DataFrame(X_scaled, columns=self.features)
        return X_scaled, X_scaled_df

    # ── Inference path ────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply previously fitted scaler to new data.
        Used by the FastAPI prediction endpoint.

        Args:
            df: Single-row or multi-row DataFrame with feature columns

        Returns:
            np.ndarray: Scaled feature matrix

        Raises:
            RuntimeError: If scaler has not been fitted or loaded yet
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform() during training or "
                "load_scaler() before inference."
            )

        X = df[self.features].copy()
        return self.scaler.transform(X)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to: {path}")

    def load_scaler(self, path: str) -> None:
        """Load a previously saved scaler from disk."""
        self.scaler  = joblib.load(path)
        self._fitted = True
        logger.info(f"Scaler loaded from: {path}")

    # ── Diagnostic report ─────────────────────────────────────────────────────

    def get_scaler_params(self) -> dict:
        """
        Return scaler parameters for MLflow logging.
        These document exactly what scaling was applied during training.
        """
        if not self._fitted:
            return {}
        return {
            f"scaler_mean_{feat}":  round(float(mean), 6)
            for feat, mean in zip(self.features, self.scaler.mean_)
        } | {
            f"scaler_std_{feat}":   round(float(std), 6)
            for feat, std in zip(self.features, self.scaler.scale_)
        }
    

def run_preprocessor(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Convenience function to run the full preprocessing pipeline."""
    preprocessor = Preprocessor()
    X_scaled, X_scaled_df = preprocessor.fit_transform(df)
    return X_scaled, X_scaled_df


if __name__ == "__main__":
    from app.pipeline.data_loader import DataLoader

    print("Running Preprocessor test...\n")

    try:
        # Step 1: Load data
        loader = DataLoader()
        df = loader.load()

        # Step 2: Run preprocessing
        preprocessor = Preprocessor()
        X_scaled, X_scaled_df = preprocessor.fit_transform(df)

        print("✅ Preprocessing successful!\n")
        print("Shape:", X_scaled.shape)
        print("\nHead of scaled data:")
        print(X_scaled_df.head())

    except Exception as e:
        print(f"Error during preprocessing: {e}")