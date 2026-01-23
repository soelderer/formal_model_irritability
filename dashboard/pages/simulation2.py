import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import gc

dash.register_page(__name__, path="/simulation2", name="Simulation 2")

df2 = pd.read_parquet("data/002_fnr_value_learning/002_fnr_value_learning_summary.parquet")

lambda_A_vals = sorted(df2["lambda_A"].unique())
eta_vals = sorted(df2["eta"].unique())
gamma_vals = sorted(df2["gamma"].unique())
C_vals = sorted(df2["C"].unique())

# Get number of iterations
with open("data/002_fnr_value_learning/n_iterations", "r") as file:
    n_iterations = int(file.readline())

# Memory management
del df2
gc.collect()

# Dropdown options
dropdown_options = [{"label": "Expected", "value": "expected"}] + [
    {"label": f"Iteration {i}", "value": i} for i in range(n_iterations)
]


layout = [
    html.H1(children="Simulation 2",
            style={"textAlign": "center"}),
    html.Div([
        html.Div([
            html.P(children="In simulation 2, we did...")
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
            dcc.Dropdown(
                id="sim2-iteration-selector",
                options=dropdown_options,
                value="expected",
                clearable=False,
                style={"width": "200px"}
            ),
        ], style={"paddingTop": "20px"}),
        html.Div([
            dcc.Graph(id='sim2-graph-content')
        ], style={"width": "60%"})
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
            "data/002_fnr_value_learning/002_fnr_value_learning_summary.parquet",
            columns=cols_needed,
            filters=filters
        )

        dff = table.to_pandas().sort_values("Step")

        # shaded bounds
        V_upper = dff["V_mean"] + dff["V_std"]
        V_lower = dff["V_mean"] - dff["V_std"]
        M_A_upper = dff["M_A_mean"] + dff["M_A_std"]
        M_A_lower = dff["M_A_mean"] - dff["M_A_std"]

        # create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # V line + shaded CI on primary y-axis
        fig.add_trace(go.Scatter(
            x=dff["Step"],
            y=dff["V_mean"],
            mode="lines",
            name="Value of state",
            line=dict(color="blue"),
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
            y=np.concatenate([V_upper, V_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ), secondary_y=False)

        # M_A line + shaded CI on secondary y-axis
        fig.add_trace(go.Scatter(
            x=dff["Step"],
            y=dff["M_A_mean"],
            mode="lines",
            name="Anger/frustration",
            line=dict(color="red"),
        ), secondary_y=True)

        fig.add_trace(go.Scatter(
            x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
            y=np.concatenate([M_A_upper, M_A_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ), secondary_y=True)

        # horizontal and vertical reference lines
        # ...for the emotions axis
        fig.add_trace(go.Scatter(
            x=[dff["Step"].min(), dff["Step"].max()],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
            hoverinfo="skip",
            yaxis="y2",
        ))

        # ...and the value axis
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.add_vline(x=100, line_width=1, line_color="black", line_dash="dash")

        # axis labels and range
        fig.update_yaxes(title_text="Value of state", secondary_y=False)
        fig.update_yaxes(title_text="Anger/frustration", secondary_y=True, range=[-0.7,0.7])
        fig.update_xaxes(title_text="Step")

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
            f"data/002_fnr_value_learning/002_fnr_value_learning_{selected_iteration}.parquet",
            columns=cols_needed,
            filters=filters
        )

        # convert to pandas and sort
        dff = table.to_pandas().sort_values("Step")


        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # V on primary y-axis (auto scale)
        fig.add_trace(
            go.Scatter(
                x=dff["Step"], y=dff["V"],
                mode="lines",
                name="Value of state",
                line=dict(color="blue"),
            ),
            secondary_y=False,
        )

        # M_A on secondary y-axis (fixed scale)
        fig.add_trace(
            go.Scatter(
                x=dff["Step"], y=dff["M_A"],
                mode="lines",
                name="Anger/frustration",
                line=dict(color="red"),
            ),
            secondary_y=True,
        )

        # horizontal and vertical reference lines
        # ...for the emotions axis
        fig.add_trace(go.Scatter(
            x=[dff["Step"].min(), dff["Step"].max()],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
            hoverinfo="skip",
            yaxis="y2",
        ))

        # ...and the value axis
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.add_vline(x=100, line_width=1, line_color="black", line_dash="dash")

        fig.update_yaxes(title_text="Value of state", secondary_y=False)
        fig.update_yaxes(title_text="Anger/frustration", secondary_y=True, range=[-0.7, 0.7])
        fig.update_xaxes(title_text="Step")

        return fig
