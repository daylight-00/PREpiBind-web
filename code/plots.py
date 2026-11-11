"""Figures shared by the pages."""
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def kde_figure(scores):
    """Density of the prediction scores with a rug under it and the decision threshold marked.
    Plotly 7 removed figure_factory.create_distplot, which the pages used; this draws the same figure: a Gaussian KDE evaluated at 500 points from the lowest to the highest score, filled, and one rug mark per score on a second axis."""
    scores = np.asarray(scores, dtype=float)
    start, end = scores.min(), scores.max()
    x = [start + i * (end - start) / 500 for i in range(500)]
    y = gaussian_kde(scores)(x)
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=y, xaxis="x1", yaxis="y1", mode="lines", name="Prediction", legendgroup="Prediction", showlegend=True, marker=dict(color="#29BDFD"), fill="tozeroy"),
            go.Scatter(x=scores, y=["Prediction"] * len(scores), xaxis="x1", yaxis="y2", mode="markers", name="Prediction", legendgroup="Prediction", showlegend=False, marker=dict(color="#29BDFD", symbol="line-ns-open")),
        ],
        layout=go.Layout(
            barmode="overlay",
            hovermode="closest",
            legend=dict(traceorder="reversed"),
            xaxis1=dict(domain=[0.0, 1.0], anchor="y2", zeroline=False),
            yaxis1=dict(domain=[0.35, 1], anchor="free", position=0.0),
            yaxis2=dict(domain=[0, 0.25], anchor="x1", dtick=1, showticklabels=False),
        ),
    )
    fig.update_layout(
        xaxis_title="Predictions",
        yaxis_title="Density",
        margin=dict(l=40, r=30, t=80, b=40),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridwidth=1)
    fig.add_vline(
        x=0.5,
        line_width=2,
        line_dash="dash",
        line_color="#F53255",
        annotation_text="Threshold (0.5)",
        annotation_position="top left",
    )
    return fig
