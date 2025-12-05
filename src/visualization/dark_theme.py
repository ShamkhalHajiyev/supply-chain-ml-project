"""
Dark-Mode Friendly Color Palette for Supply Chain ML Project.

This module provides high-contrast, accessible color schemes optimized for
dark backgrounds commonly used in presentations and IDE themes.

Usage:
    from src.visualization.dark_theme import DARK_COLORS, apply_dark_theme

    # Apply to plotly
    apply_dark_theme()

    # Get specific color palettes
    colors = get_categorical_colors(n=5)
"""

import plotly.io as pio
import plotly.graph_objects as go
from typing import List, Optional

# =============================================================================
# DARK MODE COLOR PALETTE
# =============================================================================

DARK_COLORS = {
    # Primary palette - High contrast, vibrant colors
    'primary': {
        'cyan': '#00D4FF',      # Bright cyan - primary accent
        'magenta': '#FF006E',   # Vivid magenta - alerts/late
        'lime': '#7CFC00',      # Lime green - success/on-time
        'amber': '#FFB000',     # Amber - warnings
        'violet': '#8B5CF6',    # Violet - secondary accent
        'coral': '#FF6B6B',     # Coral - errors
        'teal': '#14B8A6',      # Teal - info
        'gold': '#FFD700',      # Gold - highlights
    },

    # Categorical palette - For distinct categories (up to 10)
    'categorical': [
        '#00D4FF',  # Cyan
        '#FF006E',  # Magenta
        '#7CFC00',  # Lime
        '#FFB000',  # Amber
        '#8B5CF6',  # Violet
        '#14B8A6',  # Teal
        '#FF6B6B',  # Coral
        '#FFD700',  # Gold
        '#E879F9',  # Pink
        '#22D3EE',  # Light cyan
    ],

    # Sequential palette - For gradients (low to high)
    'sequential': {
        'blues': ['#0C1929', '#1A365D', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD'],
        'greens': ['#052E16', '#14532D', '#15803D', '#22C55E', '#4ADE80', '#86EFAC'],
        'reds': ['#450A0A', '#7F1D1D', '#B91C1C', '#EF4444', '#F87171', '#FCA5A5'],
        'purples': ['#1E1B4B', '#3730A3', '#4F46E5', '#6366F1', '#818CF8', '#A5B4FC'],
    },

    # Diverging palette - For comparison (negative-neutral-positive)
    'diverging': {
        'red_blue': ['#FF006E', '#FF4D94', '#FF99BB', '#FFFFFF', '#99CCFF', '#4D99FF', '#00D4FF'],
        'green_red': ['#7CFC00', '#9DFC4D', '#BEFC99', '#FFFFFF', '#FCA5A5', '#F87171', '#EF4444'],
    },

    # Background colors
    'background': {
        'dark': '#0D1117',      # GitHub dark
        'darker': '#010409',    # Pure dark
        'card': '#161B22',      # Card background
        'surface': '#21262D',   # Surface/hover
    },

    # Text colors
    'text': {
        'primary': '#F0F6FC',   # Primary text
        'secondary': '#8B949E', # Secondary/muted
        'accent': '#58A6FF',    # Links/accent
    },

    # Status colors (for late delivery context)
    'status': {
        'late': '#FF006E',      # Late delivery - magenta
        'ontime': '#7CFC00',    # On-time - lime
        'risk_high': '#FF6B6B', # High risk - coral
        'risk_med': '#FFB000',  # Medium risk - amber
        'risk_low': '#14B8A6',  # Low risk - teal
    },

    # Cluster colors - For customer segmentation
    'clusters': [
        '#00D4FF',  # Cluster 0 - Cyan
        '#FF006E',  # Cluster 1 - Magenta
        '#7CFC00',  # Cluster 2 - Lime
        '#FFB000',  # Cluster 3 - Amber
        '#8B5CF6',  # Cluster 4 - Violet
        '#14B8A6',  # Cluster 5 - Teal
        '#FF6B6B',  # Cluster 6 - Coral
        '#FFD700',  # Cluster 7 - Gold
    ],
}


def get_categorical_colors(n: int = 10) -> List[str]:
    """
    Get n categorical colors for distinct categories.

    Args:
        n: Number of colors needed (max 10)

    Returns:
        List of hex color codes
    """
    return DARK_COLORS['categorical'][:min(n, 10)]


def get_sequential_colors(palette: str = 'blues', n: int = 6) -> List[str]:
    """
    Get sequential colors for gradients.

    Args:
        palette: Name of palette ('blues', 'greens', 'reds', 'purples')
        n: Number of colors needed

    Returns:
        List of hex color codes
    """
    colors = DARK_COLORS['sequential'].get(palette, DARK_COLORS['sequential']['blues'])
    return colors[:min(n, len(colors))]


def get_diverging_colors(palette: str = 'red_blue') -> List[str]:
    """
    Get diverging colors for comparison visualizations.

    Args:
        palette: Name of palette ('red_blue', 'green_red')

    Returns:
        List of hex color codes
    """
    return DARK_COLORS['diverging'].get(palette, DARK_COLORS['diverging']['red_blue'])


def get_color_palette(palette_type: str = 'categorical', **kwargs) -> List[str]:
    """
    Get color palette by type.

    Args:
        palette_type: 'categorical', 'sequential', 'diverging', 'clusters', 'status'
        **kwargs: Additional arguments (n, palette name)

    Returns:
        List of hex color codes
    """
    if palette_type == 'categorical':
        return get_categorical_colors(kwargs.get('n', 10))
    elif palette_type == 'sequential':
        return get_sequential_colors(kwargs.get('palette', 'blues'), kwargs.get('n', 6))
    elif palette_type == 'diverging':
        return get_diverging_colors(kwargs.get('palette', 'red_blue'))
    elif palette_type == 'clusters':
        n = kwargs.get('n', 8)
        return DARK_COLORS['clusters'][:min(n, 8)]
    elif palette_type == 'status':
        return list(DARK_COLORS['status'].values())
    else:
        return DARK_COLORS['categorical']


def apply_dark_theme(template_name: str = 'plotly_dark_custom') -> None:
    """
    Apply dark theme to all Plotly figures.

    Creates and sets a custom dark template optimized for presentations.

    Args:
        template_name: Name for the custom template
    """
    # Create custom dark template
    custom_template = go.layout.Template()

    # Layout settings
    custom_template.layout = go.Layout(
        # Background
        paper_bgcolor=DARK_COLORS['background']['dark'],
        plot_bgcolor=DARK_COLORS['background']['card'],

        # Fonts
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color=DARK_COLORS['text']['primary']
        ),
        title=dict(
            font=dict(
                size=18,
                color=DARK_COLORS['text']['primary']
            ),
            x=0.5,
            xanchor='center'
        ),

        # Axes
        xaxis=dict(
            gridcolor=DARK_COLORS['background']['surface'],
            linecolor=DARK_COLORS['background']['surface'],
            tickfont=dict(color=DARK_COLORS['text']['secondary']),
            title=dict(font=dict(color=DARK_COLORS['text']['primary']))
        ),
        yaxis=dict(
            gridcolor=DARK_COLORS['background']['surface'],
            linecolor=DARK_COLORS['background']['surface'],
            tickfont=dict(color=DARK_COLORS['text']['secondary']),
            title=dict(font=dict(color=DARK_COLORS['text']['primary']))
        ),

        # Legend
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=DARK_COLORS['text']['primary'])
        ),

        # Color sequence
        colorway=DARK_COLORS['categorical'],

        # Margins
        margin=dict(l=60, r=30, t=60, b=60)
    )

    # Register template
    pio.templates[template_name] = custom_template
    pio.templates.default = template_name

    print(f"✅ Dark theme '{template_name}' applied")


def create_dark_figure(**kwargs) -> go.Figure:
    """
    Create a new figure with dark theme applied.

    Args:
        **kwargs: Additional arguments passed to go.Figure

    Returns:
        Plotly Figure with dark theme
    """
    fig = go.Figure(**kwargs)
    fig.update_layout(
        paper_bgcolor=DARK_COLORS['background']['dark'],
        plot_bgcolor=DARK_COLORS['background']['card'],
        font=dict(color=DARK_COLORS['text']['primary']),
        colorway=DARK_COLORS['categorical']
    )
    return fig


# Matplotlib dark theme (for SHAP plots)
MATPLOTLIB_DARK_STYLE = {
    'figure.facecolor': DARK_COLORS['background']['dark'],
    'axes.facecolor': DARK_COLORS['background']['card'],
    'axes.edgecolor': DARK_COLORS['background']['surface'],
    'axes.labelcolor': DARK_COLORS['text']['primary'],
    'text.color': DARK_COLORS['text']['primary'],
    'xtick.color': DARK_COLORS['text']['secondary'],
    'ytick.color': DARK_COLORS['text']['secondary'],
    'grid.color': DARK_COLORS['background']['surface'],
    'legend.facecolor': DARK_COLORS['background']['card'],
    'legend.edgecolor': DARK_COLORS['background']['surface'],
}


def apply_matplotlib_dark_theme():
    """Apply dark theme to matplotlib (for SHAP and other plots)."""
    import matplotlib.pyplot as plt

    for key, value in MATPLOTLIB_DARK_STYLE.items():
        plt.rcParams[key] = value

    print("✅ Matplotlib dark theme applied")


if __name__ == '__main__':
    # Demo the colors
    print("Dark Mode Color Palette")
    print("=" * 40)
    print("\nCategorical colors:", get_categorical_colors(5))
    print("Sequential (blues):", get_sequential_colors('blues', 4))
    print("Status colors:", DARK_COLORS['status'])
    print("Cluster colors:", DARK_COLORS['clusters'][:5])

