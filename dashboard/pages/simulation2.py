import dash
from dash import html, dcc, callback, Output, Input
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
import config

dash.register_page(__name__, path="/simulation2", name="Simulation 2")

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

# Dropdown options
dropdown_options = [{"label": "Expected", "value": "expected"}] + [
    {"label": f"Iteration {i}", "value": i} for i in iterations
]


layout = [
    html.H1(children="Simulation 2",
            style={"textAlign": "center"}),
    html.Div([
        html.Div([
            html.P(children="In simulation 2, we did..."),
            html.Table(
                [
                    html.Tr([
                        html.Th("Parameter", style={"padding": "0 12px"}),
                        html.Th("Range", style={"padding": "0 12px"}),
                        html.Th("Interpretation", style={"padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("η", style={"padding": "0 12px"}),
                        html.Td("[0, 1]", style={"padding": "0 12px"}),
                        html.Td("Learning rate: how quickly value expectations update", style={
                                "padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("γ", style={"padding": "0 12px"}),
                        html.Td("[0, 1]", style={"padding": "0 12px"}),
                        html.Td("Discount factor: weight given to future rewards", style={
                                "padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("λ_A", style={"padding": "0 12px"}),
                        html.Td("[0, 1]", style={"padding": "0 12px"}),
                        html.Td("Affective inertia: higher values → slower emotion updates", style={
                                "padding": "0 12px"}),
                    ]),
                ],
                style={"borderCollapse": "separate", "borderSpacing": "0 6px"},
            )
        ], style={"paddingBottom": "20px"}),
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
                    id="sim2-lambda_A-slider",
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
                    id="sim2-C-slider",
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
                    value=eta_vals[-1],
                    marks={float(v): "" for v in eta_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim2-eta-slider",
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
                    value=gamma_vals[-1],
                    marks={float(v): "" for v in gamma_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim2-gamma-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
        ], style={"display": "flex", "align-items": "center", "gap": "10px"}),
        html.Div([
            dcc.Dropdown(
                id="sim2-iteration-selector",
                options=dropdown_options,
                value="expected",
                clearable=False,
                style={"width": "200px"},
                persistence=True,
            ),
        ], style={"paddingTop": "20px"}),
        html.Div([
            dcc.Graph(id="sim2-graph-content",
                      config={"responsive": True},)
        ], style={"width": "60%", "height": "80vh"},
        )
    ])
]


@callback(
    Output("sim2-graph-content", "figure"),
    Input("sim2-lambda_A-slider", "value"),
    Input("sim2-eta-slider", "value"),
    Input("sim2-gamma-slider", "value"),
    Input("sim2-C-slider", "value"),
    Input("sim2-iteration-selector", "value"),
)
def update_graph(lambda_A, eta, gamma, C, selected_iteration):
    if selected_iteration == "expected":
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
                f"002_fnr_value_learning_{selected_iteration}.parquet"
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
