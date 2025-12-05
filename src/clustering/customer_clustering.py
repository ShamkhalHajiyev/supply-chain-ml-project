"""
Customer Clustering Module for Supply Chain ML Project.

This module provides customer segmentation using KMeans and DBSCAN algorithms.
Customer segmentation helps identify groups of customers with similar purchasing
behavior, which can be used for targeted interventions and analysis.

Usage:
    from src.clustering import run_kmeans_clustering, run_dbscan_clustering

    # Run KMeans with automatic optimal K selection
    labels, model, metrics = run_kmeans_clustering(
        data=customer_features,
        n_clusters=None,  # Auto-select
        max_clusters=10
    )

    # Run DBSCAN for density-based clustering
    labels, model, metrics = run_dbscan_clustering(
        data=customer_features,
        eps=0.5,
        min_samples=5
    )
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


def prepare_clustering_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    scale: bool = True,
    handle_missing: str = 'median'
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Prepare features for clustering analysis.

    This function selects relevant features, handles missing values,
    and optionally scales the data for better clustering results.

    Args:
        df: DataFrame with customer/order data
        feature_columns: List of columns to use (None = auto-select numeric)
        scale: Whether to standardize features (recommended)
        handle_missing: How to handle NaN values ('median', 'mean', 'drop')

    Returns:
        Tuple of (scaled_data, scaler, feature_names)

    Example:
        X, scaler, features = prepare_clustering_features(
            customer_data,
            feature_columns=['total_orders', 'avg_order_value', 'days_since_order']
        )
    """
    # Select features
    if feature_columns is None:
        # Auto-select numeric columns, excluding IDs and targets
        exclude_patterns = ['_id', 'id_', 'target', 'label', 'late_delivery', 'delivery_status']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_columns = [c for c in numeric_cols
                         if not any(p in c.lower() for p in exclude_patterns)]

    # Ensure all columns exist
    feature_columns = [c for c in feature_columns if c in df.columns]

    if len(feature_columns) == 0:
        raise ValueError("No valid feature columns found for clustering")

    # Extract features
    X = df[feature_columns].copy()

    # Handle missing values
    if handle_missing == 'median':
        X = X.fillna(X.median())
    elif handle_missing == 'mean':
        X = X.fillna(X.mean())
    elif handle_missing == 'drop':
        X = X.dropna()

    # Scale features
    scaler = StandardScaler() if scale else None
    if scale:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    return X_scaled, scaler, feature_columns


def run_kmeans_clustering(
    data: np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, KMeans, Dict[str, Any]]:
    """
    Run KMeans clustering on the data.

    KMeans groups customers into K distinct clusters based on their similarity.
    Each customer is assigned to the cluster with the nearest center (centroid).

    **Non-technical explanation:**
    Think of it like sorting customers into groups where everyone in the same
    group has similar shopping habits. The algorithm finds the best way to
    create these groups so that customers within each group are as similar
    as possible.

    Args:
        data: Scaled feature array of shape (n_samples, n_features)
        n_clusters: Number of clusters (None = auto-select using elbow method)
        max_clusters: Maximum clusters to try if auto-selecting
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for KMeans

    Returns:
        Tuple of (cluster_labels, fitted_model, metrics_dict)

    Example:
        labels, model, metrics = run_kmeans_clustering(X_scaled, n_clusters=4)
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    """
    # Auto-select optimal K if not specified
    if n_clusters is None:
        optimal_k, elbow_data = find_optimal_clusters(
            data, max_k=max_clusters, method='silhouette', random_state=random_state
        )
        n_clusters = optimal_k
        print(f"✅ Auto-selected optimal K = {n_clusters}")

    # Fit KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        **kwargs
    )

    labels = kmeans.fit_predict(data)

    # Calculate metrics
    metrics = {
        'n_clusters': n_clusters,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_score(data, labels) if n_clusters > 1 else 0,
        'calinski_harabasz_score': calinski_harabasz_score(data, labels) if n_clusters > 1 else 0,
        'davies_bouldin_score': davies_bouldin_score(data, labels) if n_clusters > 1 else 0,
        'cluster_sizes': dict(zip(*np.unique(labels, return_counts=True))),
        'centroids': kmeans.cluster_centers_
    }

    print(f"\n📊 KMeans Clustering Results:")
    print(f"   Clusters: {n_clusters}")
    print(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"   Cluster sizes: {metrics['cluster_sizes']}")

    return labels, kmeans, metrics


def run_dbscan_clustering(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    **kwargs
) -> Tuple[np.ndarray, DBSCAN, Dict[str, Any]]:
    """
    Run DBSCAN clustering on the data.

    DBSCAN finds clusters based on density - areas where many customers
    are close together form clusters, while isolated customers are marked
    as outliers (noise).

    **Non-technical explanation:**
    Unlike KMeans which forces every customer into a group, DBSCAN finds
    "natural" groups where customers cluster together. Customers who don't
    fit into any group are identified as outliers - these might be unusual
    customers worth investigating separately.

    Args:
        data: Scaled feature array of shape (n_samples, n_features)
        eps: Maximum distance between samples to be considered neighbors
        min_samples: Minimum samples to form a dense region
        **kwargs: Additional arguments for DBSCAN

    Returns:
        Tuple of (cluster_labels, fitted_model, metrics_dict)
        - Labels of -1 indicate noise/outliers

    Example:
        labels, model, metrics = run_dbscan_clustering(X_scaled, eps=0.5, min_samples=5)
        n_outliers = metrics['n_noise']
    """
    # Fit DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        **kwargs
    )

    labels = dbscan.fit_predict(data)

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    # Only calculate silhouette if we have valid clusters
    if n_clusters > 1 and n_noise < len(labels):
        # Exclude noise points for silhouette calculation
        valid_mask = labels != -1
        silhouette = silhouette_score(data[valid_mask], labels[valid_mask]) if valid_mask.sum() > n_clusters else 0
    else:
        silhouette = 0

    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels),
        'eps': eps,
        'min_samples': min_samples,
        'silhouette_score': silhouette,
        'cluster_sizes': dict(zip(*np.unique(labels[labels >= 0], return_counts=True))) if n_clusters > 0 else {}
    }

    print(f"\n📊 DBSCAN Clustering Results:")
    print(f"   Clusters found: {n_clusters}")
    print(f"   Noise points: {n_noise} ({metrics['noise_ratio']:.1%})")
    if n_clusters > 1:
        print(f"   Silhouette Score: {silhouette:.3f}")
    print(f"   Cluster sizes: {metrics['cluster_sizes']}")

    return labels, dbscan, metrics


def find_optimal_clusters(
    data: np.ndarray,
    max_k: int = 10,
    method: str = 'silhouette',
    random_state: int = 42
) -> Tuple[int, Dict[str, List]]:
    """
    Find optimal number of clusters using elbow method or silhouette analysis.

    This function tests different numbers of clusters and helps identify
    the best choice based on clustering quality metrics.

    **Non-technical explanation:**
    We try different numbers of groups (2, 3, 4, etc.) and measure how
    well each option separates customers. The "elbow" method looks for
    diminishing returns - where adding more groups stops helping much.

    Args:
        data: Scaled feature array
        max_k: Maximum number of clusters to try
        method: 'elbow' (inertia-based) or 'silhouette' (quality-based)
        random_state: Random seed

    Returns:
        Tuple of (optimal_k, analysis_data)
        analysis_data contains k_values, inertias, and silhouette_scores

    Example:
        optimal_k, data = find_optimal_clusters(X_scaled, max_k=10)
        print(f"Recommended: {optimal_k} clusters")
    """
    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    # Determine optimal K
    if method == 'silhouette':
        # Pick K with highest silhouette score
        optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        # Elbow method: find point of maximum curvature
        # Using second derivative approximation
        inertias_arr = np.array(inertias)
        if len(inertias_arr) > 2:
            second_diff = np.diff(np.diff(inertias_arr))
            elbow_idx = np.argmax(second_diff) + 1
            optimal_k = k_values[elbow_idx]
        else:
            optimal_k = k_values[0]

    analysis_data = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'method': method
    }

    return optimal_k, analysis_data


def interpret_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: List[str],
    top_n_features: int = 5
) -> pd.DataFrame:
    """
    Interpret cluster characteristics by comparing feature means.

    This function helps understand what makes each customer segment unique
    by showing how their average feature values compare to the overall average.

    **Non-technical explanation:**
    For each customer group, we look at their typical characteristics
    (average order value, purchase frequency, etc.) and compare them to
    the overall average. This tells us what makes each group special.

    Args:
        df: Original DataFrame with features
        labels: Cluster labels
        feature_columns: Features used in clustering
        top_n_features: Number of top distinguishing features to highlight

    Returns:
        DataFrame with cluster profiles and interpretations

    Example:
        profiles = interpret_clusters(customer_data, labels, features)
        print(profiles['interpretation'])
    """
    # Add labels to dataframe
    df_labeled = df.copy()
    df_labeled['cluster'] = labels

    # Calculate overall means
    overall_means = df_labeled[feature_columns].mean()

    # Calculate cluster means
    cluster_means = df_labeled.groupby('cluster')[feature_columns].mean()

    # Calculate relative difference from overall mean
    relative_diff = (cluster_means - overall_means) / (overall_means.abs() + 1e-10) * 100

    # Create interpretation for each cluster
    interpretations = []
    for cluster_id in sorted(df_labeled['cluster'].unique()):
        if cluster_id == -1:
            interpretations.append({
                'cluster': cluster_id,
                'size': (labels == cluster_id).sum(),
                'interpretation': 'Outliers/Noise - customers with unusual behavior patterns'
            })
            continue

        cluster_diff = relative_diff.loc[cluster_id].sort_values(key=abs, ascending=False)

        # Get top distinguishing features
        high_features = cluster_diff[cluster_diff > 20].head(top_n_features)
        low_features = cluster_diff[cluster_diff < -20].head(top_n_features)

        # Build interpretation
        interp_parts = []
        if len(high_features) > 0:
            high_str = ', '.join([f"{feat} (+{val:.0f}%)" for feat, val in high_features.items()])
            interp_parts.append(f"Higher than average: {high_str}")
        if len(low_features) > 0:
            low_str = ', '.join([f"{feat} ({val:.0f}%)" for feat, val in low_features.items()])
            interp_parts.append(f"Lower than average: {low_str}")

        interpretation = "; ".join(interp_parts) if interp_parts else "Similar to average customer"

        interpretations.append({
            'cluster': cluster_id,
            'size': (labels == cluster_id).sum(),
            'interpretation': interpretation
        })

    return pd.DataFrame(interpretations)


class CustomerClusterAnalyzer:
    """
    Complete customer clustering analysis pipeline.

    This class provides a unified interface for customer segmentation analysis,
    including feature preparation, clustering, and interpretation.

    **Non-technical explanation:**
    This tool automatically groups your customers based on their behavior
    and provides easy-to-understand descriptions of each group.

    Example:
        analyzer = CustomerClusterAnalyzer()
        analyzer.fit(customer_data, feature_columns=['orders', 'value', 'recency'])

        # Get cluster labels
        labels = analyzer.labels_

        # Get cluster descriptions
        descriptions = analyzer.get_cluster_descriptions()
    """

    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize the analyzer.

        Args:
            method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters (None = auto for kmeans)
            random_state: Random seed
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler_ = None
        self.model_ = None
        self.labels_ = None
        self.metrics_ = None
        self.feature_columns_ = None
        self.pca_ = None
        self.X_reduced_ = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'CustomerClusterAnalyzer':
        """
        Fit the clustering model to the data.

        Args:
            df: Customer data DataFrame
            feature_columns: Columns to use for clustering
            **kwargs: Additional parameters for clustering algorithm

        Returns:
            self (fitted analyzer)
        """
        # Prepare features
        X_scaled, self.scaler_, self.feature_columns_ = prepare_clustering_features(
            df, feature_columns
        )

        # Run clustering
        if self.method == 'kmeans':
            self.labels_, self.model_, self.metrics_ = run_kmeans_clustering(
                X_scaled,
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **kwargs
            )
        elif self.method == 'dbscan':
            self.labels_, self.model_, self.metrics_ = run_dbscan_clustering(
                X_scaled,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # PCA for visualization
        if X_scaled.shape[1] > 2:
            self.pca_ = PCA(n_components=2, random_state=self.random_state)
            self.X_reduced_ = self.pca_.fit_transform(X_scaled)
        else:
            self.X_reduced_ = X_scaled

        return self

    def get_cluster_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get human-readable descriptions of each cluster.

        Args:
            df: Original DataFrame (same as used in fit)

        Returns:
            DataFrame with cluster descriptions
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return interpret_clusters(
            df, self.labels_, self.feature_columns_
        )

    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for 2D visualization.

        Returns:
            Tuple of (reduced_features_2d, cluster_labels)
        """
        if self.X_reduced_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        centroids_2d = None
        if self.method == 'kmeans' and hasattr(self.model_, 'cluster_centers_'):
            if self.pca_ is not None:
                centroids_2d = self.pca_.transform(self.model_.cluster_centers_)
            else:
                centroids_2d = self.model_.cluster_centers_

        return self.X_reduced_, self.labels_, centroids_2d


if __name__ == '__main__':
    print("Customer Clustering Module")
    print("=" * 40)
    print("\nAvailable functions:")
    print("- run_kmeans_clustering(data, n_clusters)")
    print("- run_dbscan_clustering(data, eps, min_samples)")
    print("- find_optimal_clusters(data, max_k)")
    print("- prepare_clustering_features(df, columns)")
    print("- interpret_clusters(df, labels, columns)")
    print("\nClasses:")
    print("- CustomerClusterAnalyzer")

