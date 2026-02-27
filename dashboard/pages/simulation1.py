import dash
from dash import html, dcc, callback, Output, Input, State, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import os
import gc
import json
import base64
import config
import callbacks
import shared_content

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
                shared_content.parameter_table(
                    ["lambda_A", "C", "theta_A_w1"]),
            ], style={"paddingBottom": "40px"}),

            # ---------------------- SLIDERS ---------------------------------#

            html.Div([
                dbc.Card([
                    *shared_content.create_parameter_slider(
                        "theta_A_w1", theta_vals, "int",
                        state.get("theta_A_w1"), page_prefix, page_id),

                    *shared_content.create_parameter_slider(
                        "C", C_vals, "float",
                        state.get("C"), page_prefix, page_id),

                    *shared_content.create_parameter_slider(
                        "lambda_A", lambda_A_vals, "float",
                        state.get("lambda_A"), page_prefix, page_id),
                ], style=config.param_config_box_style, body=True),
                html.Div([
                    dcc.Graph(id={
                        "type": "graph",
                        "page": f"{page_prefix + page_id}",
                        "name": "content"
                    })
                ], style={"width": "60%"})
            ], style=config.plot_and_param_box_style)
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
