import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np

dash.register_page(__name__, path="/simulation2", name="Simulation 2")

df2 = pd.read_parquet("data/002_fnr_value_learning.parquet")

lambda_A_vals = sorted(df2["lambda_A"].unique())
eta_vals = sorted(df2["eta"].unique())
gamma_vals = sorted(df2["gamma"].unique())
C_vals = sorted(df2["C"].unique())

print(lambda_A_vals)

layout = [
    html.H1(children="Simulation 2",
            style={"textAlign": "center"}),
    html.Div([
        html.Div([
            html.P(children="In simulation 2, we did...")
        ]),
        html.Div([
            html.Div([
                html.Label("lambda_A", style={"textAlign": "center"}),
                dcc.Slider(
                    min=min(lambda_A_vals),
                    max=max(lambda_A_vals),
                    step=None,
                    value=lambda_A_vals[0],
                    marks={float(v): "" for v in lambda_A_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    dots=False,
                    id="sim2-lambda_A-slider"
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
                    id="sim2-C-slider"
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("eta", style={"textAlign": "center"}),
                dcc.Slider(
                    min=min(eta_vals),
                    max=max(eta_vals),
                    step=None,
                    dots=False,
                    value=eta_vals[-1],
                    marks={float(v): "" for v in eta_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim2-eta-slider"
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("gamma", style={"textAlign": "center"}),
                dcc.Slider(
                    min=min(gamma_vals),
                    max=max(gamma_vals),
                    step=None,
                    dots=False,
                    value=gamma_vals[-1],
                    marks={float(v): "" for v in gamma_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim2-gamma-slider"
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
        ], style={"display": "flex", "align-items": "center", "gap": "10px"}),
        html.Div([
            dcc.Graph(id='sim2-graph-content')
        ], style={"width": "45%"})
    ])
]


@callback(
    Output("sim2-graph-content", "figure"),
    Input("sim2-lambda_A-slider", "value"),
    Input("sim2-eta-slider", "value"),
    Input("sim2-gamma-slider", "value"),
    Input("sim2-C-slider", "value")
)

def update_graph(lambda_A, eta, gamma, C):
    dff = df2[
        (np.isclose(df2["lambda_A"], lambda_A)) &
        (np.isclose(df2["eta"], eta)) &
        (np.isclose(df2["gamma"], gamma)) &
        (np.isclose(df2["C"], C))
    ]

    fig = px.line(
        dff,
        x="Step",
        y=["V", "M_A"],
        labels={"value": "Value", "variable": "Series"},
    )

    # rename legend entries
    fig.data[0].name = "Value of state"
    fig.data[1].name = "Anger/frustration"

    # horizontal and vertical reference lines
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.add_vline(
        x=100,
        line_width=1,
        line_color="black",
        line_dash="dash",
    )

    return fig
