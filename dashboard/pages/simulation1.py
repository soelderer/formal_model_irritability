import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import config
import os

dash.register_page(__name__, path="/simulation1", name="Simulation 1")

description = """
Simulation 1 illustrates a minimal model of irritability using a single agent
that can act either friendly or aggressively. The agent experiences one
unexpected nonreward, followed by a sequence of neutral outcomes, allowing us
to isolate the emotional impact of a single frustrating event.
Anger/frustration is elicited by this event and then gradually decays over time
in the absence of further negative feedback. Importantly, the tendency to act
aggressively is directly modulated by the current level of anger, leading to a
temporary increase in aggressive behavior. This setup captures the clinically
relevant idea that irritability involves a prolonged and disproportionate
response to an isolated frustration.
"""

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
            html.P(children=description),
            html.Table(
                [
                    html.Tr(
                        [
                            html.Th("Parameter", style={"padding": "0 12px"}),
                            html.Th("Range", style={"padding": "0 12px"}),
                            html.Th("Interpretation", style={
                                    "padding": "0 12px"}),
                        ],
                        style={**config.toprule, **config.midrule},
                    ),
                    html.Tr(
                        [
                            html.Td("θ_A_w1", style={"padding": "0 12px"}),
                            html.Td("ℝ", style={"padding": "0 12px"}),
                            html.Td(
                                "Reactive aggression gain: logit-linear scaling of aggressive behavior as a function of current anger/frustration",
                                style={"padding": "0 12px"},
                            ),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("C", style={"padding": "0 12px"}),
                            html.Td("[0, 1]", style={"padding": "0 12px"}),
                            html.Td(
                                "Perceived controllability of the environment",
                                style={"padding": "0 12px"},
                            ),
                        ],
                        style=config.bottomrule,
                    ),
                ],
                style=config.table_style,
            )
        ], style={"paddingBottom": "20px"}),
        html.Div([
            html.Div([
                html.Label("theta_A_w1", style={
                    "textAlign": "center"}),
                dcc.Slider(
                    min=min(theta_vals),
                    max=max(theta_vals),
                    step=None,
                    value=theta_vals[0],
                    marks={int(v): "" for v in theta_vals},
                    tooltip={"always_visible": True,
                             "placement": "bottom"},
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
                    tooltip={"always_visible": True,
                             "placement": "bottom"},
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
