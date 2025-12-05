"""
Customer Clustering Module for Supply Chain ML Project.

Provides KMeans and DBSCAN clustering for customer segmentation analysis.

Usage:
    from src.clustering import run_kmeans_clustering, run_dbscan_clustering

    # KMeans
    labels, model, metrics = run_kmeans_clustering(data, n_clusters=4)

    # DBSCAN
    labels, model, metrics = run_dbscan_clustering(data, eps=0.5, min_samples=5)
"""

from .customer_clustering import (
    run_kmeans_clustering,
    run_dbscan_clustering,
    find_optimal_clusters,
    prepare_clustering_features,
    interpret_clusters,
    CustomerClusterAnalyzer
)

__all__ = [
    'run_kmeans_clustering',
    'run_dbscan_clustering',
    'find_optimal_clusters',
    'prepare_clustering_features',
    'interpret_clusters',
    'CustomerClusterAnalyzer'
]

