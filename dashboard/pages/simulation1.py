import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import config
import os

dash.register_page(__name__, path="/simulation1", name="Simulation 1")

meta_df = pd.read_parquet(
    os.path.join(
        config.DATA_DIR,
        "001_fnr_impulse_response",
        "meta_info.parquet"
    )
)

theta_vals = meta_df["theta_vals"].iloc[0]
C_vals = meta_df["C_vals"].iloc[0]

layout = [
    html.H1(children="Simulation 1",
            style={"textAlign": "center"}),
    html.Div([
        html.Div([
            html.P(children="In simulation 1, we did...")
        ]),
        html.Div([
            html.Div([
                html.Label("theta_A_w1", style={"textAlign": "center"}),
                dcc.Slider(
                    min=min(theta_vals),
                    max=max(theta_vals),
                    step=None,
                    value=theta_vals[0],
                    marks={int(v): "" for v in theta_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    dots=False,
                    id="sim1-theta_A_w1-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("C", style={"textAlign": "center"}),
                dcc.Slider(
                    min=min(C_vals),
                    max=max(C_vals),
                    step=None,
                    dots=False,
                    value=C_vals[-1],
                    marks={float(v): "" for v in C_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim1-C-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
        ], style={"display": "flex", "align-items": "center", "gap": "10px",
                  "padding-bottom": "40px"}),
        html.Div([
            dcc.Graph(id='sim1-graph-content')
        ], style={"width": "45%"})
    ])
]


@callback(
    Output("sim1-graph-content", "figure"),
    Input("sim1-theta_A_w1-slider", "value"),
    Input("sim1-C-slider", "value")
)
def update_graph(theta_A_w1, C):
    df = pd.read_parquet(
        os.path.join(
            config.DATA_DIR,
            "001_fnr_impulse_response",
            "001_fnr_impulse_response.parquet"
        )
    )

    dff = df[(df["theta_A_w1"] == theta_A_w1) &
             (np.isclose(df["C"], C))]

    fig = px.line(
        dff,
        x="Step",
        y=["p_A", dff["M_A"].abs()],
        labels={"value": "Value", "variable": "Series"},
        title="Anger and aggressive action"
    )

    fig.update_xaxes(range=[0, 40])
    fig.update_yaxes(range=[0, 1])

    # rename the series in the legend
    fig.data[0].name = "Probability aggressive"
    fig.data[1].name = "Anger/frustration (absolute)"

    return fig
