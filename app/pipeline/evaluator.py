# pipeline/evaluator.py
# Computes clustering evaluation metrics logged to MLflow.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluates clustering quality and generates diagnostic plots.
    All metrics and plots are logged to MLflow as artefacts.

    Metrics computed:
        - Silhouette score (overall and per-cluster)
        - Inertia (within-cluster sum of squares)
        - Cluster sizes and balance
        - Per-point silhouette scores

    Plots generated:
        - Silhouette plot (per-point)
        - 2D PCA cluster scatter
        - Centroid heatmap
        - Elbow curve
    """

    def __init__(
        self,
        X_scaled:    np.ndarray,
        labels:      np.ndarray,
        centroids:   np.ndarray,
        cluster_to_tier: dict,
        feature_names:   list,
    ):
        self.X_scaled        = X_scaled
        self.labels          = labels
        self.centroids       = centroids
        self.cluster_to_tier = cluster_to_tier
        self.feature_names   = feature_names
        self.n_clusters      = settings.n_clusters
        self.plot_dir        = os.path.join(settings.artefact_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_all_metrics(self) -> dict:
        """
        Compute all evaluation metrics.

        Returns:
            dict: All metrics ready for MLflow logging
        """
        metrics = {}

        # Overall silhouette score
        overall_sil = silhouette_score(self.X_scaled, self.labels)
        metrics["silhouette_score_overall"] = round(float(overall_sil), 6)

        # Per-cluster silhouette scores
        per_point_sil = silhouette_samples(self.X_scaled, self.labels)
        for cluster_id in range(self.n_clusters):
            tier   = self.cluster_to_tier.get(cluster_id, str(cluster_id))
            mask   = self.labels == cluster_id
            c_mean = float(per_point_sil[mask].mean())
            c_min  = float(per_point_sil[mask].min())
            metrics[f"silhouette_{tier}_mean"] = round(c_mean, 6)
            metrics[f"silhouette_{tier}_min"]  = round(c_min, 6)

        logger.info(
            f"Overall silhouette score: {overall_sil:.4f}, "
            f"{'Strong' if overall_sil > 0.7 else 'Reasonable' if overall_sil > 0.5 else 'Weak'} "
            f"cluster separation"
        )

        return metrics

    # ── Plots ─────────────────────────────────────────────────────────────────

    def generate_all_plots(self) -> list:
        """
        Generate all diagnostic plots and save to artefact directory.

        Returns:
            list: File paths of all saved plot images
        """
        paths = []
        paths.append(self._plot_silhouette())
        paths.append(self._plot_pca_scatter())
        paths.append(self._plot_centroid_heatmap())
        return [p for p in paths if p is not None]

    def _plot_silhouette(self) -> str:
        """Per-point silhouette plot."""
        per_point_sil = silhouette_samples(self.X_scaled, self.labels)
        overall_sil   = per_point_sil.mean()

        colours = ["gold", "steelblue", "tomato"]
        fig, ax  = plt.subplots(figsize=(10, 7))
        y_lower  = 10

        for cluster_id in range(self.n_clusters):
            tier   = self.cluster_to_tier.get(cluster_id, str(cluster_id))
            vals   = np.sort(per_point_sil[self.labels == cluster_id])
            n      = len(vals)
            y_upper = y_lower + n

            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, vals,
                facecolor=colours[cluster_id], alpha=0.75,
                label=f"{tier} (n={n})"
            )
            ax.text(-0.05, y_lower + n / 2, tier[:3],
                    fontsize=9, fontweight="bold",
                    color=colours[cluster_id])
            y_lower = y_upper + 10

        ax.axvline(x=overall_sil, color="crimson", linestyle="--",
                   linewidth=1.5,
                   label=f"Overall = {overall_sil:.3f}")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_title("Silhouette Plot: Per-Point Analysis",
                     fontweight="bold")
        ax.set_yticks([])
        ax.legend(loc="upper right")
        plt.tight_layout()

        path = os.path.join(self.plot_dir, "silhouette_plot.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def _plot_pca_scatter(self) -> str:
        """2D PCA cluster scatter plot."""
        pca    = PCA(n_components=2, random_state=settings.random_state)
        X_pca  = pca.fit_transform(self.X_scaled)
        var    = pca.explained_variance_ratio_

        tier_colours = {"Premium": "gold", "Standard": "steelblue",
                        "Basic": "tomato"}

        fig, ax = plt.subplots(figsize=(9, 7))

        for cluster_id in range(self.n_clusters):
            tier = self.cluster_to_tier.get(cluster_id, str(cluster_id))
            mask = self.labels == cluster_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=tier_colours[tier], label=tier,
                       s=100, edgecolors="grey", linewidth=0.5, alpha=0.85)

        # Plot centroids
        centroids_pca = pca.transform(self.centroids)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   marker="X", s=250, c="black", zorder=5,
                   label="Centroids")
        for cid, (cx, cy) in enumerate(centroids_pca):
            ax.annotate(
                self.cluster_to_tier.get(cid, ""),
                xy=(cx, cy), xytext=(8, 8),
                textcoords="offset points",
                fontsize=9, fontweight="bold"
            )

        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
        ax.set_title("K-Means Clusters — 2D PCA Projection",
                     fontweight="bold")
        ax.legend(title="Quality Tier")
        plt.tight_layout()

        path = os.path.join(self.plot_dir, "pca_scatter.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def _plot_centroid_heatmap(self) -> str:
        """Centroid feature values heatmap."""
        tier_labels = [
            self.cluster_to_tier.get(i, str(i))
            for i in range(self.n_clusters)
        ]
        centroid_df = pd.DataFrame(
            self.centroids,
            columns=self.feature_names,
            index=tier_labels
        ).round(3)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            centroid_df, annot=True, fmt=".2f",
            cmap="YlOrRd", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Scaled Value (z-score)"}
        )
        ax.set_title("Cluster Centroids: Scaled Feature Space",
                     fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.plot_dir, "centroid_heatmap.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info(f"Saved: {path}")
        return path