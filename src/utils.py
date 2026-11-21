"""
Utility functions for data visualization and formatting.
"""

from typing import List
import plotly.graph_objects as go


def create_skill_coverage_chart(matched_count: int, missing_count: int) -> go.Figure:
    """
    Generates a donut chart displaying JD skill coverage.
    """
    labels = ["Matched Skills", "Missing Skills"]
    values = [matched_count, missing_count]
    colors = ["#22c55e", "#ef4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors),
                textinfo="label+value",
                hoverinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
    )
    return fig


def format_score_badge(score: float) -> str:
    """Returns a percentage string for display."""
    return f"{round(score * 100, 1)}%"
    