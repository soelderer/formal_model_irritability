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
    __name__, path="/simulation200", name="Simulation 200")

description = """
In Simulation 200, we extend the previous setup by introducing response
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
        "200_development",
        "meta_info.parquet"
    )
)

files = glob.glob(
    os.path.join(
        config.DATA_DIR,
        "200_development",
        "200_development_*.parquet"
    )
)

# Determine the available iterations by filename
# Files must follow this convention: 200_development_i.parquet
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
C_start_vals = meta_df["C_start_vals"].iloc[0]
C_end_vals = meta_df["C_end_vals"].iloc[0]
lambda_C_vals = meta_df["lambda_C_vals"].iloc[0]
midpoint_C_vals = meta_df["midpoint_C_vals"].iloc[0]
I_start_vals = meta_df["I_start_vals"].iloc[0]
I_end_vals = meta_df["I_end_vals"].iloc[0]
lambda_I_vals = meta_df["lambda_I_vals"].iloc[0]
midpoint_I_vals = meta_df["midpoint_I_vals"].iloc[0]
w_v_A_vals = meta_df["w_v_A_vals"].iloc[0]
environment_type_vals = meta_df["environment_type_vals"].iloc[0]


del meta_df
gc.collect()

# Dropdown options
dropdown_options = [{"label": "Expected", "value": "expected"}] + [
    {"label": f"Iteration {i}", "value": i} for i in iterations
]

layout = [
    html.H1(children="Simulation 200",
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
                    id="sim200-lambda_A-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
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
                    id="sim200-eta-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
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
                    id="sim200-gamma-slider",
                    persistence=False,
                ),
            ], style={"width": "100%",
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
                    id="sim200-alpha-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
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
                    id="sim200-kappa-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("C_start", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        C_start_vals),
                    max=max(
                        C_start_vals),
                    step=None,
                    dots=False,
                    value=C_start_vals[-1],
                    marks={float(
                        v): "" for v in C_start_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-C_start-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("C_end", style={
                    "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        C_end_vals),
                    max=max(
                        C_end_vals),
                    step=None,
                    dots=False,
                    value=C_end_vals[-1],
                    marks={float(
                        v): "" for v in C_end_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-C_end-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
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
                    id="sim200-lambda_C-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("midpoint_C", style={
                    "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        midpoint_C_vals),
                    max=max(
                        midpoint_C_vals),
                    step=None,
                    dots=False,
                    value=midpoint_C_vals[-1],
                    marks={float(
                        v): "" for v in midpoint_C_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-midpoint_C-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),

            html.Div([
                html.Label("I_start", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        I_start_vals),
                    max=max(
                        I_start_vals),
                    step=None,
                    dots=False,
                    value=I_start_vals[-1],
                    marks={float(
                        v): "" for v in I_start_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-I_start-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("I_end", style={
                    "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        I_end_vals),
                    max=max(
                        I_end_vals),
                    step=None,
                    dots=False,
                    value=I_end_vals[-1],
                    marks={float(
                        v): "" for v in I_end_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-I_end-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("lambda_I", style={
                           "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        lambda_I_vals),
                    max=max(
                        lambda_I_vals),
                    step=None,
                    dots=False,
                    value=lambda_I_vals[-1],
                    marks={float(
                        v): "" for v in lambda_I_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-lambda_I-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
            html.Div([
                html.Label("midpoint_I", style={
                    "textAlign": "center"}),
                dcc.Slider(
                    min=min(
                        midpoint_I_vals),
                    max=max(
                        midpoint_I_vals),
                    step=None,
                    dots=False,
                    value=midpoint_I_vals[-1],
                    marks={float(
                        v): "" for v in midpoint_I_vals},
                    tooltip={
                        "always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    id="sim200-midpoint_I-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
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
                    id="sim200-w_v_A-slider",
                    persistence=True,
                ),
            ], style={"width": "100%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
                  "align-items": "center", "gap": "10px"}),
        html.Div([
            dcc.Dropdown(
                id="sim200-iteration-selector",
                options=dropdown_options,
                value="expected",
                clearable=False,
                style={
                    "width": "200px"},
                persistence=True,
            ),
            dcc.Dropdown(
                id="sim200-environment_type-selector",
                options=environment_type_vals,
                value="EnvironmentAversive",
                clearable=False,
                style={
                    "width": "200px"},
                persistence=True,
            ),
        ], style={"display": "flex", "gap": "12px", "paddingTop": "20px"}),
        html.Div([
            dcc.Graph(
                id="sim200-graph-content",
                config={"responsive": True})
        ], style={"width": "60%", "height": "80vh"})
    ])
]


@callback(
    Output(
        "sim200-graph-content", "figure"),
    Input(
        "sim200-lambda_A-slider", "value"),
    Input(
        "sim200-eta-slider", "value"),
    Input(
        "sim200-gamma-slider", "value"),
    Input(
        "sim200-alpha-slider", "value"),
    Input(
        "sim200-kappa-slider", "value"),
    Input(
        "sim200-C_start-slider", "value"),
    Input(
        "sim200-C_end-slider", "value"),
    Input(
        "sim200-lambda_C-slider", "value"),
    Input(
        "sim200-midpoint_C-slider", "value"),
    Input(
        "sim200-I_start-slider", "value"),
    Input(
        "sim200-I_end-slider", "value"),
    Input(
        "sim200-lambda_I-slider", "value"),
    Input(
        "sim200-midpoint_I-slider", "value"),
    Input(
        "sim200-w_v_A-slider", "value"),
    Input(
        "sim200-iteration-selector", "value"),
    Input(
        "sim200-environment_type-selector", "value"),
)
def update_graph(lambda_A, eta, gamma, alpha, kappa, C_start, C_end, lambda_C,
                 midpoint_C, I_start, I_end, lambda_I, midpoint_I,
                 w_v_A, selected_iteration, environment_type):
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
            ("C_start",
             "=", C_start),
            ("C_end",
             "=", C_end),
            ("lambda_C",
             "=", lambda_C),
            ("midpoint_C",
             "=", midpoint_C),
            ("I_start",
             "=", I_start),
            ("I_end",
             "=", I_end),
            ("lambda_I",
             "=", lambda_I),
            ("midpoint_I",
             "=", midpoint_I),
            ("w_v_A",
             "=", w_v_A),
            ("environment_type",
             "=", environment_type),
        ]

        # read only filtered rows and needed columns
        cols_needed = ["Step", "V_mean", "V_std", "M_A_mean", "M_A_std",
                       "M_S_mean", "M_S_std", "C", "v_mean", "v_std", "I"]

        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "200_development",
                "200_development_summary.parquet"
            ),
            columns=cols_needed,
            filters=filters
        )

        dff = (
            table.
            to_pandas()
            .query("Step % 2 == 0")  # only even Steps
            .sort_values("Step")
        )

        print(dff.columns)

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
            rows=4,
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

        fig.update_yaxes(title_text="Value of state", row=1, col=1)

        # --- C trace  ---
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

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["I"],
                mode="lines",
                name="Response inhibition",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0, row=2, col=1)
        # fig.add_vline(x=100, line_dash="dash")

        fig.update_yaxes(
            title_text="Controllability and inhibition",
            range=[0, 1],
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Step", row=2, col=1)

        # --- Emotions (bottom: M_A + M_S, same axis) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["M_A_mean"],
                mode="lines",
                name="Anger/frustration",
                line=dict(color="red"),
            ),
            row=3,
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
            row=3,
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
            row=3,
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
            row=3,
            col=1,
        )

        fig.update_yaxes(
            title_text="Emotions",
            range=[-2, 2],
            row=3,
            col=1,
        )

        fig.add_hline(y=0, row=3, col=1)

        # --- Response vigor ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["v_mean"],
                mode="lines",
                name="Response vigor",
                line=dict(color="magenta"),
            ),
            row=4,
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
            row=4,
            col=1,
        )

        fig.update_yaxes(
            title_text="Response vigor",
            range=[0, 1],
            row=4,
            col=1,
        )
        fig.add_hline(y=0.5, row=4, col=1)
        # fig.add_vline(x=100, line_dash="dash")

        fig.update_xaxes(title_text="Step", row=4, col=1)

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
            ("C_start",
             "=", C_start),
            ("C_end",
             "=", C_end),
            ("lambda_C",
             "=", lambda_C),
            ("midpoint_C",
             "=", midpoint_C),
            ("I_start",
             "=", I_start),
            ("I_end",
             "=", I_end),
            ("lambda_I",
             "=", lambda_I),
            ("midpoint_I",
             "=", midpoint_I),
            ("w_v_A",
             "=", w_v_A),
            ("environment_type",
             "=", environment_type),
        ]

        cols_needed = ["Step", "V", "M_A", "M_S", "C", "v", "I"]

        # read only filtered rows and selected columns
        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "200_development",
                f"200_development_{selected_iteration}.parquet"
            ),
            columns=cols_needed,
            filters=filters
        )

        print(table)

        dff = (
            table.
            to_pandas()
            .query("Step % 2 == 0")  # only even Steps
            .sort_values("Step")
        )

        fig = make_subplots(
            rows=4,
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

        fig.update_yaxes(title_text="Value of state", row=1, col=1)
        fig.add_hline(y=0, row=1, col=1)

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

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["I"],
                mode="lines",
                name="Response inhibition",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(
            title_text="Controllability and inhibition",
            range=[0, 1],
            row=2,
            col=1,
        )

        fig.add_hline(y=0, row=2, col=1)
        # fig.add_vline(x=100, line_dash="dash")

        # --- Emotions (bottom: M_A + M_S, same axis) ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["M_A"],
                mode="lines",
                name="Anger/frustration",
                line=dict(color="red"),
            ),
            row=3,
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
            row=3,
            col=1,
        )

        fig.update_yaxes(
            title_text="Emotions",
            range=[-2, 2],
            row=3,
            col=1,
        )
        fig.update_xaxes(title_text="Step", row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["I"],
                mode="lines",
                name="Response inhibition",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )

        # --- Response vigor ---
        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["v"],
                mode="lines",
                name="Response vigor",
                line=dict(color="magenta"),
            ),
            row=4,
            col=1,
        )

        fig.update_yaxes(
            title_text="Response vigor",
            range=[0, 1],
            row=4,
            col=1,
        )

        fig.add_hline(y=0.5, row=4, col=1)
        # fig.add_vline(x=100, line_dash="dash")

        fig.update_xaxes(title_text="Step", row=4, col=1)

        return fig
