import dash
from dash import html, dcc, callback, Output, Input, State, ALL
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import gc
import glob
import re
import os
import json
import base64
import config
import callbacks
import shared_content

page_prefix = "simulation"
page_id = "2"

dash.register_page(
    __name__, path=f"/{page_prefix + page_id}", name=f"Simulation {page_id}"
)

description = """
Simulation 2 models a frustration-induction task in which participants learn
whether their actions are rewarded over time. In an initial block, rewards are
delivered almost consistently, leading to a strong expectation that the task is
controllable and rewarding. In a subsequent block, outcomes are systematically
worse despite continued effort, mimicking rigged feedback commonly used to
elicit frustration. Agents gradually update their expectations based on
experienced outcomes, capturing how perceived task value declines when
expectations are violated. Frustration/anger is modeled as accumulating over
repeated negative surprises."""


meta_df = pd.read_parquet(
    os.path.join(
        config.DATA_DIR,
        "002_fnr_value_learning",
        "meta_info.parquet"
    )
)

files = glob.glob(
    os.path.join(
        config.DATA_DIR,
        "002_fnr_value_learning",
        "002_fnr_value_learning_*.parquet"
    )
)

# Determine the available iterations by filename
# Files must follow this convention: 002_value_learning_i.parquet
n_iterations = 0
iterations = []
for f in files:
    match = re.search(r"_(\d+)\.parquet$", f)

    if match:
        iterations += [int(match.group(1))]

if iterations:
    n_iterations = len(iterations)

lambda_A_vals = meta_df["lambda_A_vals"].iloc[0]
eta_vals = meta_df["eta_vals"].iloc[0]
gamma_vals = meta_df["gamma_vals"].iloc[0]
C_vals = meta_df["C_vals"].iloc[0]

del meta_df
gc.collect()

defaults = {
    "lambda_A": lambda_A_vals[0],
    "eta": eta_vals[-1],
    "gamma": gamma_vals[-1],
    "C": C_vals[-1],
    "iteration": "expected",
}


def layout(state_str: str = None, **_kwargs):
    iteration_vals = [{"label": "Expected", "value": "expected"}] + [
        {"label": f"Iteration {i}", "value": i} for i in iterations
    ]

    # Decode the state from the hash
    state = defaults | (json.loads(
        base64.b64decode(state_str)) if state_str else {})

    layout = [
        html.H1(children="Simulation 2",
                style={"textAlign": "center"}),
        html.Div([
            dcc.Store(id=f"{page_prefix + page_id}-store",
                      storage_type="memory"),
            html.Div([
                html.P(children=description),
                shared_content.parameter_table(
                    ["lambda_A", "C", "eta", "gamma"]
                )
            ], style={"paddingBottom": "20px"}),

            # ---------------------- SLIDERS ---------------------------------#

            html.Div([
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
                    html.Label("eta", style={"textAlign": "center"}),
                    dcc.Slider(
                        min=min(eta_vals),
                        max=max(eta_vals),
                        step=None,
                        dots=False,
                        value=state.get("eta"),
                        marks={float(v): "" for v in eta_vals},
                        tooltip={"always_visible": True,
                                 "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": f"{page_prefix + page_id}",
                            "name": "eta"
                        },
                        persistence=True,
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
                        value=state.get("gamma"),
                        marks={float(v): "" for v in gamma_vals},
                        tooltip={"always_visible": True,
                                 "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": f"{page_prefix + page_id}",
                            "name": "gamma"
                        },
                        persistence=True,
                    ),
                ], style={"width": "20%",
                          "display": "inline-block",
                          "padding": "0 10px"}),
            ], style={"display": "flex", "align-items": "center", "gap": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id={
                        "type": "control",
                        "page": f"{page_prefix + page_id}",
                        "name": "iteration",
                    },
                    options=iteration_vals,
                    value=state.get("iteration"),
                    clearable=False,
                    style={"width": "200px"},
                    persistence=True,
                ),
            ], style={"paddingTop": "20px"}),
            html.Div([
                dcc.Graph(id={
                    "type": "graph",
                    "name": "content",
                    "page": f"{page_prefix + page_id}"
                },
                    config={"responsive": True},)
            ], style={"width": "60%", "height": "80vh"},
            )
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
    lambda_A, eta, gamma, C, iteration = (store[k] for k in [
        "lambda_A", "eta", "gamma", "C", "iteration"
    ])

    if iteration == "expected":
        filters = [
            ("lambda_A", "=", lambda_A),
            ("eta", "=", eta),
            ("gamma", "=", gamma),
            ("C", "=", C)
        ]

        # read only filtered rows and needed columns
        cols_needed = ["Step", "V_mean", "V_std", "M_A_mean", "M_A_std"]

        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "002_fnr_value_learning",
                "002_fnr_value_learning_summary.parquet"
            ),
            columns=cols_needed,
            filters=filters
        )

        dff = table.to_pandas().sort_values("Step")

        # shaded bounds
        V_upper = dff["V_mean"] + dff["V_std"]
        V_lower = dff["V_mean"] - dff["V_std"]
        M_A_upper = dff["M_A_mean"] + dff["M_A_std"]
        M_A_lower = dff["M_A_mean"] - dff["M_A_std"]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        # --- V (top) ---
        fig.add_trace(go.Scatter(
            x=dff["Step"],
            y=dff["V_mean"],
            mode="lines",
            name="Value of state",
            line=dict(color="blue"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
            y=np.concatenate([V_upper, V_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,255,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)

        fig.add_hline(y=0, row=1, col=1)

        # --- M_A (bottom) ---
        fig.add_trace(go.Scatter(
            x=dff["Step"],
            y=dff["M_A_mean"],
            mode="lines",
            name="Anger/frustration",
            line=dict(color="red"),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
            y=np.concatenate([M_A_upper, M_A_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ), row=2, col=1)

        fig.add_hline(y=0, row=2, col=1)
        fig.add_vline(x=100, line_dash="dash")

        fig.update_yaxes(title_text="Value of state", row=1, col=1)
        fig.update_yaxes(title_text="Anger/frustration",
                         range=[-2, 2], row=2, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=1)

        return fig

    # Single model iteration
    else:
        filters = [
            ("lambda_A", "=", lambda_A),
            ("eta", "=", eta),
            ("gamma", "=", gamma),
            ("C", "=", C)
        ]

        cols_needed = ["Step", "V", "M_A"]

        # read only filtered rows and selected columns
        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "002_fnr_value_learning",
                f"002_fnr_value_learning_{iteration}.parquet"
            ),
            columns=cols_needed,
            filters=filters
        )

        # convert to pandas and sort
        dff = table.to_pandas().sort_values("Step")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        # --- V (top, auto scale) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["V"],
                mode="lines",
                name="Value of state",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        fig.add_hline(y=0, row=1, col=1)

        # --- M_A (bottom, fixed scale) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["M_A"],
                mode="lines",
                name="Anger/frustration",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0, row=2, col=1)
        fig.add_vline(x=100, line_dash="dash")

        fig.update_yaxes(title_text="Value of state", row=1, col=1)
        fig.update_yaxes(
            title_text="Anger/frustration",
            range=[-2, 2],
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Step", row=2, col=1)

        return fig
