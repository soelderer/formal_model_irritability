import dash
from dash import html, dcc, callback, Output, Input, State, ALL
import plotly.express as px
import pandas as pd
import numpy as np
import os
import gc
import json
import base64
import config
import callbacks

page_prefix = "simulation"
page_id = "1"

dash.register_page(
    __name__, path=f"/{page_prefix + page_id}", name=f"Simulation {page_id}"
)

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
lambda_A_vals = meta_df["lambda_A_vals"].iloc[0]

del meta_df
gc.collect()

defaults = {
    "theta_A_w1": theta_vals[0],
    "C": C_vals[-1],
    "lambda_A": lambda_A_vals[0]
}


def layout(state_str: str = None, **_kwargs):
    # Decode the state from the hash
    state = defaults | (json.loads(
        base64.b64decode(state_str)) if state_str else {})

    layout = [
        html.H1(children="Simulation 1",
                style={"textAlign": "center"}),
        html.Div([
            dcc.Store(id=f"{page_prefix + page_id}-store",
                      storage_type="memory"),
            html.Div([
                html.P(children=description),
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Th("Parameter", style={
                                        "padding": "0 12px"}),
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
                        ),
                        html.Tr([
                            html.Td("λ_A", style={"padding": "0 12px"}),
                            html.Td("[0, 1]", style={"padding": "0 12px"}),
                            html.Td("Affective inertia: higher values → slower emotion updates", style={
                                    "padding": "0 12px"}),
                        ], style=config.bottomrule),
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
                        value=state.get("theta_A_w1"),
                        marks={int(v): "" for v in theta_vals},
                        tooltip={"always_visible": True,
                                 "placement": "bottom"},
                        updatemode="drag",
                        dots=False,
                        id={
                            "type": "control",
                            "page": f"{page_prefix + page_id}",
                            "name": "theta_A_w1"
                        },
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
                        value=state.get("C"),
                        marks={float(v): "" for v in C_vals},
                        tooltip={"always_visible": True,
                                 "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": f"{page_prefix + page_id}",
                            "name": "C"
                        },
                        persistence=True,
                    ),
                ], style={"width": "20%",
                          "display": "inline-block",
                          "padding": "0 10px"}),
                html.Div([
                    html.Label("lambda_A", style={"textAlign": "center"}),
                    dcc.Slider(
                        min=min(lambda_A_vals),
                        max=max(lambda_A_vals),
                        step=None,
                        value=state.get("lambda_A"),
                        marks={float(v): "" for v in lambda_A_vals},
                        tooltip={"always_visible": True,
                                 "placement": "bottom"},
                        updatemode="drag",
                        dots=False,
                        id={
                            "type": "control",
                            "page": f"{page_prefix + page_id}",
                            "name": "lambda_A"
                        },
                        persistence=True,
                    ),
                ], style={"width": "20%",
                          "display": "inline-block",
                          "padding": "0 10px"}),
            ], style={"display": "flex", "align-items": "center", "gap": "10px",
                      "padding-bottom": "40px"}),
            html.Div([
                dcc.Graph(id={
                    "type": "graph",
                    "page": f"{page_prefix + page_id}",
                    "name": "content"
                })
            ], style={"width": "45%"})
        ])
    ]

    return layout


@callback(
    Output(f"{page_prefix + page_id}-store", "data"),
    Input("main-url", "hash"),
    State(f"{page_prefix + page_id}-store", "data"),
    prevent_initial_call=False,
)
def load_hash_to_store(hash_value, current_store):
    return callbacks._load_hash_to_store_generic(
        hash_value, current_store, defaults
    )


@callback(
    Output({"type": "control", "page": f"{page_prefix + page_id}",
            "name": ALL}, "value"),
    Input(f"{page_prefix + page_id}-store", "data"),
)
def sync_sliders(store):
    return callbacks._sync_sliders_generic(store)


@callback(
    Output({"type": "graph", "name": "content",
            "page": f"{page_prefix + page_id}"},
           "figure"),
    Input(f"{page_prefix + page_id}-store", "data"),
    prevent_initial_call=True,
)
def update_graph(store):
    theta_A_w1, C, lambda_A = (store[k]
                               for k in ["theta_A_w1", "C", "lambda_A"])

    print("updating graph")

    df = pd.read_parquet(
        os.path.join(
            config.DATA_DIR,
            "001_fnr_impulse_response",
            "001_fnr_impulse_response.parquet"
        )
    )

    dff = df[(df["theta_A_w1"] == theta_A_w1) &
             (np.isclose(df["C"], C)) &
             (np.isclose(df["lambda_A"], lambda_A))
             ]

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
