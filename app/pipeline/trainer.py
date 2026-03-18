# pipeline/trainer.py
# K-Means model training with cluster-to-tier mapping.

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Trains the K-Means clustering model and establishes
    the cluster-to-tier mapping.

    After training:
        - self.model           → fitted KMeans instance
        - self.cluster_to_tier → {0: "Premium", 1: "Standard", 2: "Basic"}
          (assigned by ranking clusters on internalFacilitiesCount centroid)
        - self.labels_         → cluster assignment for every training point
    """

    def __init__(self):
        self.model           = None
        self.cluster_to_tier = {}
        self.labels_         = None
        self._feature_names  = settings.features_list

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_scaled:    np.ndarray,
        X_original:  pd.DataFrame,
    ) -> "Trainer":
        """
        Fit K-Means and map clusters to quality tiers.

        Args:
            X_scaled:   StandardScaler-transformed feature matrix (for fitting)
            X_original: Original unscaled DataFrame (for centroid interpretation)

        Returns:
            self — enables method chaining
        """
        logger.info(
            f"Training K-Means: "
            f"K={settings.n_clusters}, "
            f"n_init={settings.n_init}, "
            f"max_iter={settings.max_iter}, "
            f"random_state={settings.random_state}"
        )

        self.model = KMeans(
            n_clusters   = settings.n_clusters,
            init         = "k-means++",
            n_init       = settings.n_init,
            max_iter     = settings.max_iter,
            tol          = 1e-4,
            random_state = settings.random_state,
            algorithm    = "lloyd",
        )

        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_

        logger.info(
            f"Training converged in {self.model.n_iter_} iterations, "
            f"final inertia: {self.model.inertia_:.4f}"
        )

        # ── Assign cluster labels → tier names ───────────────────────────────
        self.cluster_to_tier = self._build_tier_mapping(X_original)

        logger.info("Cluster → Tier mapping established:")
        for cluster_id, tier in self.cluster_to_tier.items():
            n = int((self.labels_ == cluster_id).sum())
            logger.info(f"  Cluster {cluster_id} → {tier}  (n={n})")

        return self

    def _build_tier_mapping(
        self, X_original: pd.DataFrame
    ) -> dict:
        """
        Map raw cluster numbers to quality tier names.

        Strategy:
            Rank clusters by their mean internalFacilitiesCount centroid
            (original scale), highest = Premium, middle = Standard,
            lowest = Basic.

        This ranking is deterministic given fixed hyperparameters and
        random_state, but saving it ensures API consistency even if
        K-Means assigns different numbers on future re-runs.
        """
        # Compute mean internalFacilitiesCount for each cluster
        anchor_feature = "internalFacilitiesCount"
        cluster_means = {
            cluster_id: X_original.loc[
                self.labels_ == cluster_id, anchor_feature
            ].mean()
            for cluster_id in range(settings.n_clusters)
        }

        # Sort clusters by mean (descending) → assign tier labels
        sorted_clusters = sorted(
            cluster_means.items(),
            key=lambda x: x[1],
            reverse=True
        )
        tier_names = ["Premium", "Standard", "Basic"]

        return {
            cluster_id: tier_names[rank]
            for rank, (cluster_id, _) in enumerate(sorted_clusters)
        }

    def predict(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments and compute confidence scores.

        Confidence proxy:
            For each point, compute distance to all centroids.
            confidence = 1 - (d_assigned / d_nearest_other)
            Range: 0 (on the boundary) to 1 (perfectly central).

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            Tuple of (cluster_labels, confidence_scores)
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call train() first "
                "or load a saved model with load_model()."
            )

        cluster_labels = self.model.predict(X_scaled)

        # Compute distances to all centroids
        distances = self.model.transform(X_scaled)  # shape (n, K)

        confidence_scores = []
        for i, assigned_cluster in enumerate(cluster_labels):
            d_assigned = distances[i, assigned_cluster]
            # Distance to nearest OTHER centroid
            other_distances = np.delete(distances[i], assigned_cluster)
            d_nearest_other = other_distances.min()

            if d_nearest_other == 0:
                confidence = 0.0
            else:
                confidence = float(
                    1.0 - (d_assigned / (d_assigned + d_nearest_other))
                )
            confidence_scores.append(round(confidence, 4))

        return cluster_labels, np.array(confidence_scores)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_path: str, mapping_path: str) -> None:
        """Save model and tier mapping to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model,           model_path)
        joblib.dump(self.cluster_to_tier, mapping_path)
        logger.info(f"Model saved to:   {model_path}")
        logger.info(f"Mapping saved to: {mapping_path}")

    def load(self, model_path: str, mapping_path: str) -> None:
        """Load model and tier mapping from disk."""
        self.model           = joblib.load(model_path)
        self.cluster_to_tier = joblib.load(mapping_path)
        logger.info(f"Model loaded from:   {model_path}")
        logger.info(f"Mapping loaded from: {mapping_path}")

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def get_hyperparameters(self) -> dict:
        """Return hyperparameters dict for MLflow logging."""
        return {
            "n_clusters":   settings.n_clusters,
            "n_init":       settings.n_init,
            "max_iter":     settings.max_iter,
            "random_state": settings.random_state,
            "init_method":  "k-means++",
            "algorithm":    "lloyd",
            "tol":          1e-4,
        }

    def get_cluster_sizes(self) -> dict:
        """Return count of centers per tier for MLflow logging."""
        if self.labels_ is None:
            return {}
        return {
            self.cluster_to_tier.get(int(c), str(c)): int(count)
            for c, count in zip(
                *np.unique(self.labels_, return_counts=True)
            )
        }
    


if __name__ == "__main__":
    from app.pipeline.data_loader import DataLoader
    from app.pipeline.preprocessor import Preprocessor

    print("Running Trainer test...\n")

    try:
        # Step 1: Load data
        loader = DataLoader()
        df = loader.load()

        # Step 2: Preprocess
        preprocessor = Preprocessor()
        X_scaled, X_scaled_df = preprocessor.fit_transform(df)

        # Step 3: Train model
        trainer = Trainer()
        trainer.train(X_scaled, df)

        print("\n Training successful!\n")

        # Show mapping
        print("Cluster → Tier Mapping:")
        print(trainer.cluster_to_tier)

        # Show cluster sizes
        print("\nCluster Sizes:")
        print(trainer.get_cluster_sizes())

        # Optional: test prediction
        labels, confidence = trainer.predict(X_scaled[:5])

        print("\nSample Predictions:")
        for i in range(len(labels)):
            print(
                f"Cluster: {labels[i]}, "
                f"Tier: {trainer.cluster_to_tier[labels[i]]}, "
                f"Confidence: {confidence[i]}"
            )

    except Exception as e:
        print(f"Error during training: {e}")


# To run this test, execute `python -m app.pipeline.trainer` from the project root.