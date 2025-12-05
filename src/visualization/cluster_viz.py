"""
Cluster Visualization Functions for Supply Chain ML Project.

Dark-mode friendly visualizations for customer segmentation analysis.

Usage:
    from src.visualization.cluster_viz import plot_clusters_2d, plot_cluster_profiles

    # 2D scatter plot of clusters
    fig = plot_clusters_2d(X_reduced, labels, title="Customer Segments")
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any

from .dark_theme import DARK_COLORS, get_color_palette


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Customer Clusters",
    feature_names: Optional[List[str]] = None,
    show_centroids: bool = True,
    centroids: Optional[np.ndarray] = None,
    hover_data: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Create a 2D scatter plot of clustered data.

    This visualization shows different customer groups as distinct colors,
    making it easy to see how customers are segmented based on their behavior.

    Args:
        X: 2D array of shape (n_samples, 2) - typically PCA/UMAP reduced
        labels: Cluster labels for each sample
        title: Plot title
        feature_names: Names for x and y axes
        show_centroids: Whether to show cluster centers
        centroids: Cluster centroids (optional)
        hover_data: Additional data for hover tooltips

    Returns:
        Plotly Figure object
    """
    n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1)
    colors = DARK_COLORS['clusters'][:n_clusters]

    fig = go.Figure()

    # Plot each cluster
    for i, cluster_id in enumerate(sorted(np.unique(labels))):
        if cluster_id == -1:
            # Noise points (DBSCAN)
            cluster_name = "Noise/Outliers"
            color = DARK_COLORS['text']['secondary']
            opacity = 0.3
        else:
            cluster_name = f"Cluster {cluster_id}"
            color = colors[i % len(colors)]
            opacity = 0.7

        mask = labels == cluster_id

        hover_text = None
        if hover_data is not None:
            hover_text = [
                f"Cluster: {cluster_id}<br>" +
                "<br>".join([f"{col}: {val:.2f}" if isinstance(val, float) else f"{col}: {val}"
                            for col, val in row.items()])
                for _, row in hover_data[mask].iterrows()
            ]

        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=cluster_name,
            marker=dict(
                size=8,
                color=color,
                opacity=opacity,
                line=dict(width=0.5, color=DARK_COLORS['background']['dark'])
            ),
            hovertext=hover_text,
            hoverinfo='text' if hover_text else 'name'
        ))

    # Add centroids
    if show_centroids and centroids is not None:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(
                size=15,
                color=DARK_COLORS['primary']['gold'],
                symbol='x',
                line=dict(width=2, color=DARK_COLORS['text']['primary'])
            )
        ))

    # Layout
    x_label = feature_names[0] if feature_names and len(feature_names) > 0 else "Component 1"
    y_label = feature_names[1] if feature_names and len(feature_names) > 1 else "Component 2"

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor=DARK_COLORS['background']['dark'],
        plot_bgcolor=DARK_COLORS['background']['card'],
        font=dict(color=DARK_COLORS['text']['primary']),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=DARK_COLORS['text']['primary'])
        ),
        height=500
    )

    return fig


def plot_cluster_distribution(
    labels: np.ndarray,
    title: str = "Cluster Distribution"
) -> go.Figure:
    """
    Create a bar chart showing the distribution of samples across clusters.

    This shows how many customers fall into each segment, helping identify
    if segments are balanced or if some groups are much larger than others.

    Args:
        labels: Cluster labels
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Count samples per cluster
    unique, counts = np.unique(labels, return_counts=True)

    # Sort by cluster ID
    sort_idx = np.argsort(unique)
    unique = unique[sort_idx]
    counts = counts[sort_idx]

    # Create labels
    cluster_names = [f"Cluster {c}" if c >= 0 else "Noise" for c in unique]
    colors = [DARK_COLORS['clusters'][i % len(DARK_COLORS['clusters'])]
              if c >= 0 else DARK_COLORS['text']['secondary']
              for i, c in enumerate(unique)]

    # Calculate percentages
    total = counts.sum()
    percentages = counts / total * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cluster_names,
        y=counts,
        marker_color=colors,
        text=[f"{c:,}<br>({p:.1f}%)" for c, p in zip(counts, percentages)],
        textposition='outside',
        textfont=dict(color=DARK_COLORS['text']['primary'])
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title="Cluster",
        yaxis_title="Number of Customers",
        paper_bgcolor=DARK_COLORS['background']['dark'],
        plot_bgcolor=DARK_COLORS['background']['card'],
        font=dict(color=DARK_COLORS['text']['primary']),
        showlegend=False,
        height=400
    )

    return fig


def plot_cluster_profiles(
    data: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
    title: str = "Cluster Profiles",
    normalize: bool = True
) -> go.Figure:
    """
    Create a radar/polar chart showing the profile of each cluster.

    This visualization helps understand what makes each customer segment unique
    by comparing their average characteristics across different features.

    Args:
        data: DataFrame with features
        labels: Cluster labels
        features: List of feature names to include
        title: Plot title
        normalize: Whether to normalize features (recommended for comparison)

    Returns:
        Plotly Figure object
    """
    n_clusters = len(np.unique(labels[labels >= 0]))
    colors = DARK_COLORS['clusters'][:n_clusters]

    # Calculate cluster means
    cluster_profiles = {}
    for cluster_id in sorted(np.unique(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        cluster_profiles[cluster_id] = data.loc[mask, features].mean()

    # Normalize if requested
    if normalize:
        all_means = pd.DataFrame(cluster_profiles).T
        all_means = (all_means - all_means.min()) / (all_means.max() - all_means.min() + 1e-10)
        for cluster_id in cluster_profiles:
            cluster_profiles[cluster_id] = all_means.loc[cluster_id]

    fig = go.Figure()

    for i, (cluster_id, profile) in enumerate(cluster_profiles.items()):
        # Close the polygon
        values = profile.values.tolist() + [profile.values[0]]
        feature_labels = features + [features[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_labels,
            fill='toself',
            name=f'Cluster {cluster_id}',
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)],
            opacity=0.3
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        polar=dict(
            bgcolor=DARK_COLORS['background']['card'],
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else None,
                gridcolor=DARK_COLORS['background']['surface'],
                linecolor=DARK_COLORS['background']['surface']
            ),
            angularaxis=dict(
                gridcolor=DARK_COLORS['background']['surface'],
                linecolor=DARK_COLORS['background']['surface']
            )
        ),
        paper_bgcolor=DARK_COLORS['background']['dark'],
        font=dict(color=DARK_COLORS['text']['primary']),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=DARK_COLORS['text']['primary'])
        ),
        height=500
    )

    return fig


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    optimal_k: Optional[int] = None,
    title: str = "Elbow Curve for Optimal K"
) -> go.Figure:
    """
    Create an elbow curve plot for K-Means cluster selection.

    This visualization helps determine the optimal number of clusters
    by showing where the "elbow" (diminishing returns) occurs.

    Args:
        k_values: List of K values tested
        inertias: Corresponding inertia (within-cluster sum of squares) values
        optimal_k: Highlight the optimal K if known
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertias,
        mode='lines+markers',
        name='Inertia',
        line=dict(color=DARK_COLORS['primary']['cyan'], width=3),
        marker=dict(size=10, color=DARK_COLORS['primary']['cyan'])
    ))

    # Highlight optimal K
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[inertias[idx]],
            mode='markers',
            name=f'Optimal K={optimal_k}',
            marker=dict(
                size=20,
                color=DARK_COLORS['primary']['gold'],
                symbol='star',
                line=dict(width=2, color=DARK_COLORS['text']['primary'])
            )
        ))

        # Add annotation
        fig.add_annotation(
            x=optimal_k,
            y=inertias[idx],
            text=f"Optimal: K={optimal_k}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=DARK_COLORS['primary']['gold'],
            font=dict(color=DARK_COLORS['text']['primary']),
            bgcolor=DARK_COLORS['background']['card'],
            bordercolor=DARK_COLORS['primary']['gold']
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Within-cluster Sum of Squares)",
        paper_bgcolor=DARK_COLORS['background']['dark'],
        plot_bgcolor=DARK_COLORS['background']['card'],
        font=dict(color=DARK_COLORS['text']['primary']),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=DARK_COLORS['text']['primary'])
        ),
        height=400
    )

    # Add grid
    fig.update_xaxes(
        gridcolor=DARK_COLORS['background']['surface'],
        dtick=1
    )
    fig.update_yaxes(
        gridcolor=DARK_COLORS['background']['surface']
    )

    return fig


def plot_silhouette_scores(
    k_values: List[int],
    silhouette_scores: List[float],
    optimal_k: Optional[int] = None,
    title: str = "Silhouette Score Analysis"
) -> go.Figure:
    """
    Create a silhouette score plot for cluster quality assessment.

    Higher silhouette scores indicate better-defined clusters. This helps
    validate that customers within a segment are similar to each other
    and different from customers in other segments.

    Args:
        k_values: List of K values tested
        silhouette_scores: Corresponding silhouette scores
        optimal_k: Highlight the optimal K if known
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Bar chart
    colors = [DARK_COLORS['primary']['gold'] if k == optimal_k else DARK_COLORS['primary']['cyan']
              for k in k_values]

    fig.add_trace(go.Bar(
        x=[f"K={k}" for k in k_values],
        y=silhouette_scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in silhouette_scores],
        textposition='outside',
        textfont=dict(color=DARK_COLORS['text']['primary'])
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title="Number of Clusters",
        yaxis_title="Silhouette Score",
        paper_bgcolor=DARK_COLORS['background']['dark'],
        plot_bgcolor=DARK_COLORS['background']['card'],
        font=dict(color=DARK_COLORS['text']['primary']),
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, max(silhouette_scores) * 1.2])
    )

    return fig


if __name__ == '__main__':
    # Demo
    print("Cluster Visualization Module")
    print("=" * 40)
    print("Functions available:")
    print("- plot_clusters_2d()")
    print("- plot_cluster_distribution()")
    print("- plot_cluster_profiles()")
    print("- plot_elbow_curve()")
    print("- plot_silhouette_scores()")

