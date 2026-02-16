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

dash.register_page(
    __name__, path="/simulation6", name="Simulation 6")

description = """
In Simulation 6, we extend the previous setup by introducing response
inhibition as an additional modulatory factor on response vigor. As in
Simulation 5, anger/frustration and sadness emerge as a function of perceived
controllability, and emotional intensity determines the urgency with which
actions are executed. Response inhibition selectively attenuates the
translation of emotional action readiness into overt response vigor, without
affecting emotional magnitude, action selection, or learning.

Using the same task structure and controllability manipulation as in Simulation
5, we examine how developmental differences in response inhibition shape
behavioral outcomes under frustrative nonreward. High response inhibition
dampens vigorous responding even in the presence of intense anger, whereas low
response inhibition allows emotional urgency to translate more directly into
overt action. This extension illustrates how variation in inhibitory control
can dissociate emotional intensity from behavioral expression, providing a
mechanistic account of divergent developmental trajectories under otherwise
similar emotional dynamics.
"""

meta_df = pd.read_parquet(
    os.path.join(
        config.DATA_DIR,
        "006_response_inhibition",
        "meta_info.parquet"
    )
)

files = glob.glob(
    os.path.join(
        config.DATA_DIR,
        "006_response_inhibition",
        "006_response_inhibition_*.parquet"
    )
)

# Determine the available iterations by filename
# Files must follow this convention: 006_response_inhibition_i.parquet
n_iterations = 0
iterations = []
for f in files:
    match = re.search(
        r"_(\d+)\.parquet$", f)

    if match:
        iterations += [
            int(match.group(1))]

if iterations:
    n_iterations = len(
        iterations)

lambda_A_vals = meta_df["lambda_A_vals"].iloc[0]
eta_vals = meta_df["eta_vals"].iloc[0]
gamma_vals = meta_df["gamma_vals"].iloc[0]
alpha_vals = meta_df["alpha_vals"].iloc[0]
kappa_vals = meta_df["kappa_vals"].iloc[0]
lambda_C_vals = meta_df["lambda_C_vals"].iloc[0]
midpoint_vals = meta_df["midpoint_vals"].iloc[0]
w_v_A_vals = meta_df["w_v_A_vals"].iloc[0]
I_vals = meta_df["I_vals"].iloc[0]

del meta_df
gc.collect()

# Dropdown options
dropdown_options = [{"label": "Expected", "value": "expected"}] + [
    {"label": f"Iteration {i}", "value": i} for i in iterations
]


layout = [
    html.H1(children="Simulation 6",
            style={"textAlign": "center"}),
    html.Div([
        html.Div([
            html.P(
                children=description
            ),
            html.Table(
                [
                    html.Tr([
                        html.Th("Parameter", style={
                                "padding": "0 12px"}),
                        html.Th("Range", style={
                                "padding": "0 12px"}),
                        html.Th("Interpretation", style={
                                "padding": "0 12px"}),
                    ], style={**config.toprule, **config.midrule}),
                    html.Tr([
                        html.Td("η", style={
                                "padding": "0 12px"}),
                        html.Td("[0, 1]", style={
                                "padding": "0 12px"}),
                        html.Td(
                            ("Learning rate: how quickly value "
                             "expectations update"),
                            style={"padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("γ", style={
                                "padding": "0 12px"}),
                        html.Td("[0, 1]", style={
                                "padding": "0 12px"}),
                        html.Td(("Discount factor: weight given to "
                                 "future rewards"),
                                style={"padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("λ_A", style={
                                "padding": "0 12px"}),
                        html.Td("[0, 1]", style={
                                "padding": "0 12px"}),
                        html.Td(("Affective inertia: higher values → "
                                 "slower emotion updates"),
                                style={"padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("α", style={
                                "padding": "0 12px"}),
                        html.Td("[0, 1]", style={
                                "padding": "0 12px"}),
                        html.Td(("Relative weighting of prediction errors "
                                 "versus absolute rewards as affective inputs "
                                 "(1 ... only RPE)"),
                                style={"padding": "0 12px"}),
                    ]),
                    html.Tr([
                        html.Td("κ", style={
                                "padding": "0 12px"}),
                        html.Td("> 1", style={
                                "padding": "0 12px"}),
                        html.Td((
                            "Negativity bias: negative affective inputs are "
                            "amplified relative to positive ones"),
                            style={"padding": "0 12px"},),
                    ]),
                    html.Tr([
                        html.Td("λ_C", style={
                                "padding": "0 12px"}),
                        html.Td("[0,1]", style={
                                "padding": "0 12px"}),
                        html.Td((
                            "Controllability learning rate: speed at which "
                            "perceived controllability declines across "
                            "trials in the uncontrollable block"),
                            style={"padding": "0 12px"},),
                    ]),
                    html.Tr([
                        html.Td("midpoint", style={
                                "padding": "0 12px"}),
                        html.Td("[100,200]", style={
                                "padding": "0 12px"}),
                        html.Td((
                            "Trial number at which C reaches 50% "
                            "(inflection point of the decay)"),
                            style={"padding": "0 12px"},),
                    ]),
                    html.Tr([
                        html.Td("v_w_A", style={
                                "padding": "0 12px"}),
                        html.Td(">=0", style={
                                "padding": "0 12px"}),
                        html.Td((
                            "Contribution of frustration/anger to "
                            "response vigor"),
                            style={"padding": "0 12px"},),
                    ],),
                    html.Tr([
                        html.Td("I", style={
                                "padding": "0 12px"}),
                        html.Td("[0,1]", style={
                                "padding": "0 12px"}),
                        html.Td((
                            "Degree of response inhibition"),
                            style={"padding": "0 12px"},),
                    ],
                        style=config.bottomrule),
                ],
                style=config.table_style,
            )
        ], style={"paddingBottom": "20px"}),
        html.Div([
            html.Div([
                html.Label("lambda_A", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        lambda_A_vals),
                    max=max(
                        lambda_A_vals),
                    step=None,
                    value=lambda_A_vals[0],
                    marks={float(
                        v): "" for v in lambda_A_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    dots=False,
                    id="sim6-lambda_A-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("eta", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        eta_vals),
                    max=max(
                        eta_vals),
                    step=None,
                    dots=False,
                    value=eta_vals[-1],
                    marks={float(
                        v): "" for v in eta_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-eta-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("gamma", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        gamma_vals),
                    max=max(
                        gamma_vals),
                    step=None,
                    dots=False,
                    value=gamma_vals[-1],
                    marks={float(
                        v): "" for v in gamma_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-gamma-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("alpha", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        alpha_vals),
                    max=max(
                        alpha_vals),
                    step=None,
                    dots=False,
                    value=alpha_vals[-1],
                    marks={float(
                        v): "" for v in alpha_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-alpha-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("kappa", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        kappa_vals),
                    max=max(
                        kappa_vals),
                    step=None,
                    dots=False,
                    value=kappa_vals[-1],
                    marks={float(
                        v): "" for v in kappa_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-kappa-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("lambda_C", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        lambda_C_vals),
                    max=max(
                        lambda_C_vals),
                    step=None,
                    dots=False,
                    value=lambda_C_vals[-1],
                    marks={float(
                        v): "" for v in lambda_C_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-lambda_C-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("midpoint", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        midpoint_vals),
                    max=max(
                        midpoint_vals),
                    step=None,
                    dots=False,
                    value=midpoint_vals[-1],
                    marks={float(
                        v): "" for v in midpoint_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-midpoint-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("w_v_A", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        w_v_A_vals),
                    max=max(
                        w_v_A_vals),
                    step=None,
                    dots=False,
                    value=w_v_A_vals[-1],
                    marks={float(
                        v): "" for v in w_v_A_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-w_v_A-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("I", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        I_vals),
                    max=max(
                        I_vals),
                    step=None,
                    dots=False,
                    value=I_vals[-1],
                    marks={float(
                        v): "" for v in I_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim6-I-slider",
                    persistence=True,
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
        ], style={"display": "flex", "align-items": "center", "gap": "10px"}),
        html.Div([
            dcc.Dropdown(
                id="sim6-iteration-selector",
                options=dropdown_options,
                value="expected",
                clearable=False,
                style={
                    "width": "200px"},
                persistence=True,
            ),
        ], style={"paddingTop": "20px"}),
        html.Div([
            dcc.Graph(
                id="sim6-graph-content",
                config={"responsive": True})
        ], style={"width": "60%", "height": "80vh"})
    ])
]


@callback(
    Output(
        "sim6-graph-content", "figure"),
    Input(
        "sim6-lambda_A-slider", "value"),
    Input(
        "sim6-eta-slider", "value"),
    Input(
        "sim6-gamma-slider", "value"),
    Input(
        "sim6-alpha-slider", "value"),
    Input(
        "sim6-kappa-slider", "value"),
    Input(
        "sim6-lambda_C-slider", "value"),
    Input(
        "sim6-midpoint-slider", "value"),
    Input(
        "sim6-w_v_A-slider", "value"),
    Input(
        "sim6-I-slider", "value"),
    Input(
        "sim6-iteration-selector", "value"),
)
def update_graph(lambda_A, eta, gamma, alpha, kappa, lambda_C, midpoint,
                 w_v_A, I, selected_iteration):
    if selected_iteration == "expected":
        filters = [
            ("lambda_A",
             "=", lambda_A),
            ("eta",
             "=", eta),
            ("gamma",
             "=", gamma),
            ("alpha",
             "=", alpha),
            ("kappa",
             "=", kappa),
            ("lambda_C",
             "=", lambda_C),
            ("midpoint",
             "=", midpoint),
            ("w_v_A",
             "=", w_v_A),
            ("I",
             "=", I),
        ]

        # read only filtered rows and needed columns
        cols_needed = ["Step", "V_mean", "V_std", "M_A_mean", "M_A_std",
                       "M_S_mean", "M_S_std", "C", "v_mean", "v_std"]

        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "006_response_inhibition",
                "006_response_inhibition_summary.parquet"
            ),
            columns=cols_needed,
            filters=filters
        )

        dff = table.to_pandas(
        ).sort_values("Step")

        # shaded bounds
        V_upper = dff["V_mean"] + dff["V_std"]
        V_lower = dff["V_mean"] - dff["V_std"]
        M_A_upper = dff["M_A_mean"] + dff["M_A_std"]
        M_A_lower = dff["M_A_mean"] - dff["M_A_std"]
        M_S_upper = dff["M_S_mean"] + dff["M_S_std"]
        M_S_lower = dff["M_S_mean"] - dff["M_S_std"]
        v_upper = dff["v_mean"] + dff["v_std"]
        v_lower = dff["v_mean"] - dff["v_std"]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        # --- V (top) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["V_mean"],
                mode="lines",
                name="Value of state",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
                y=np.concatenate([V_upper, V_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,255,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_hline(y=0, row=1, col=1)

        # --- Emotions (bottom: M_A + M_S, same axis) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["M_A_mean"],
                mode="lines",
                name="Anger/frustration",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
                y=np.concatenate([M_A_upper, M_A_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["M_S_mean"],
                mode="lines",
                name="Sadness",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
                y=np.concatenate([M_S_upper, M_S_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(255,165,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # --- C trace (bottom: emotions, no stderr) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["C"],
                mode="lines",
                name="Controllability",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0, row=2, col=1)
        fig.add_vline(x=100, line_dash="dash")

        fig.update_yaxes(title_text="Value of state", row=1, col=1)
        fig.update_yaxes(
            title_text="Emotions",
            range=[-2, 2],
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Step", row=2, col=1)

        # --- Response vigor ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["v_mean"],
                mode="lines",
                name="Response vigor",
                line=dict(color="magenta"),
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
                y=np.concatenate([v_upper, v_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(255,0,255,0.2)",
                line=dict(color="rgba(255,0,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        fig.update_yaxes(
            title_text="Response vigor",
            range=[0, 1],
            row=3,
            col=1,
        )
        fig.add_hline(y=0.5, row=3, col=1)
        fig.add_vline(x=100, line_dash="dash")

        fig.update_xaxes(title_text="Step", row=3, col=1)

        return fig

    # Single model iteration
    else:
        filters = [
            ("lambda_A",
             "=", lambda_A),
            ("eta",
             "=", eta),
            ("gamma",
             "=", gamma),
            ("alpha",
             "=", alpha),
            ("kappa",
             "=", kappa),
            ("lambda_C",
             "=", lambda_C),
            ("midpoint",
             "=", midpoint),
            ("w_v_A",
             "=", w_v_A),
            ("I",
             "=", I),
        ]

        cols_needed = [
            "Step", "V", "M_A", "M_S", "C", "v"]

        # read only filtered rows and selected columns
        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "006_response_inhibition",
                f"006_response_inhibition_{selected_iteration}.parquet"
            ),
            columns=cols_needed,
            filters=filters
        )

        # convert to pandas and sort
        dff = table.to_pandas(
        ).sort_values("Step")

        fig = make_subplots(
            rows=3,
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

        # --- Emotions (bottom: M_A + M_S, same axis) ---
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

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["M_S"],
                mode="lines",
                name="Sadness",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

        # --- C trace (bottom: emotions, no stderr) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["C"],
                mode="lines",
                name="Controllability",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0, row=2, col=1)
        fig.add_vline(x=100, line_dash="dash")

        fig.update_yaxes(title_text="Value of state", row=1, col=1)
        fig.update_yaxes(
            title_text="Emotions",
            range=[-2, 2],
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Step", row=2, col=1)

        # --- Response vigor ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["v"],
                mode="lines",
                name="Response vigor",
                line=dict(color="magenta"),
            ),
            row=3,
            col=1,
        )

        fig.update_yaxes(
            title_text="Response vigor",
            range=[0, 1],
            row=3,
            col=1,
        )

        fig.add_hline(y=0.5, row=3, col=1)
        fig.add_vline(x=100, line_dash="dash")

        fig.update_xaxes(title_text="Step", row=3, col=1)

        return fig
