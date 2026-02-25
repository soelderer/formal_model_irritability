import dash
from dash import ALL, MATCH, html, dcc, callback, Output, Input, State
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
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

page_prefix = "simulation"
page_id = "200"

dash.register_page(
    __name__, path=f"/{page_prefix + page_id}", name=f"Simulation {page_id}"
)

description = """
    In Simulation 200, we shift gears: instead of simulating experimental trials,
    we simulate development over a longer time period (arbitrary time scale).
    Perceived controllability (C) and response inhibition (I) can be varied across
    development. For pragmatic reasons, both C and I follow a logistic shape, and
    the researcher can tune start/end values, inflection points, and steepness of
    change.

    We simulate many episodes, each with two steps. Agents choose actions that
    are friendly, neutral, or aggressive. In step 1, an action is performed and
    a (non)reward is received. In step 2, agents respond to the first outcome.
    Anger increases momentary aggressive tendencies, and emotional intensity of
    frustration/anger increases response vigor.
    Step 1 always starts with neutral emotions and neutral state values, so
    episodes are decoupled and there is no learning in the present simulation.

    We plot emotions, response vigor, and probabilities of aggressive actions in
    step 2.
    A temper outburst is an aggressive action with high response vigor in response
    to a nonreward.

    Three agent-perceived environments are available: aversive (70% punishments
    -1, 20% neutral, 10% rewards +1); neutral (70% neutral, 10% punishments -1,
    20% rewards +1); appetitive (70% rewards +1, 20% neutral, 10% punishments
    -1).

    This setup lets us explore how emotions and response vigor vary with
    environment and developmental trajectories of C and I.
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
    n_iterations = len(iterations)

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

defaults = {
    "lambda_A": lambda_A_vals[0],
    "eta": eta_vals[-1],
    "gamma": gamma_vals[-1],
    "alpha": alpha_vals[-1],
    "kappa": kappa_vals[-1],
    "C_start": C_start_vals[-1],
    "C_end": C_end_vals[-1],
    "lambda_C": lambda_C_vals[-1],
    "midpoint_C": midpoint_C_vals[-1],
    "I_start": I_start_vals[-1],
    "I_end": I_end_vals[-1],
    "lambda_I": lambda_I_vals[-1],
    "midpoint_I": midpoint_I_vals[-1],
    "w_v_A": w_v_A_vals[-1],
    "iteration": "expected",
    "environment_type": "EnvironmentAversive",
}


def layout(state_str: str = None, **_kwargs):
    iteration_vals = [{"label": "Expected", "value": "expected"}] + [
        {"label": f"Iteration {i}", "value": i} for i in iterations
    ]

    # Decode the state from the hash
    state = defaults | (json.loads(
        base64.b64decode(state_str)) if state_str else {})

    layout = [
        html.H1(children="Simulation 200",
                style={"textAlign": "center"}),
        html.Div([
            dcc.Store(id=f"{page_prefix + page_id}-store",
                      storage_type="memory"),
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
                        # html.Tr([
                        #     html.Td("eta", style={
                        #             "padding": "0 12px"}),
                        #     html.Td("[0, 1]", style={
                        #             "padding": "0 12px"}),
                        #     html.Td(
                        #         ("Learning rate: how quickly value "
                        #          "expectations update"),
                        #         style={"padding": "0 12px"}),
                        # ]),
                        # html.Tr([
                        #     html.Td("gamma", style={
                        #             "padding": "0 12px"}),
                        #     html.Td("[0, 1]", style={
                        #             "padding": "0 12px"}),
                        #     html.Td(("Discount factor: weight given to "
                        #             "future rewards"),
                        #             style={"padding": "0 12px"}),
                        # ]),
                        # html.Tr([
                        #     html.Td("lambda_A", style={
                        #             "padding": "0 12px"}),
                        #     html.Td("[0, 1]", style={
                        #             "padding": "0 12px"}),
                        #     html.Td(("Affective inertia: higher values → "
                        #             "slower emotion updates"),
                        #             style={"padding": "0 12px"}),
                        # ]),
                        # html.Tr([
                        #     html.Td("alpha", style={
                        #             "padding": "0 12px"}),
                        #     html.Td("[0, 1]", style={
                        #             "padding": "0 12px"}),
                        #     html.Td(("Relative weighting of prediction errors "
                        #             "versus absolute rewards as affective inputs "
                        #              "(1 ... only RPE)"),
                        #             style={"padding": "0 12px"}),
                        # ]),
                        html.Tr([
                            html.Td("kappa", style={
                                    "padding": "0 12px"}),
                            html.Td("> 1", style={
                                    "padding": "0 12px"}),
                            html.Td((
                                "Negativity bias: negative affective inputs are "
                                "amplified relative to positive ones"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("C_start", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,1]", style={
                                    "padding": "0 12px"}),
                            html.Td(("Start value of perceived controllability"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("C_end", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,1]", style={
                                    "padding": "0 12px"}),
                            html.Td(("End value of perceived controllability"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("lambda_C", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,1]", style={
                                    "padding": "0 12px"}),
                            html.Td((
                                "Speed at which "
                                "perceived controllability changes across "
                                "development"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("midpoint_C", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,100]", style={
                                    "padding": "0 12px"}),
                            html.Td((
                                "Episode at which C reaches 50% "
                                "(inflection point of the decay)"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("I_start", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,1]", style={
                                    "padding": "0 12px"}),
                            html.Td(("Start value of response inhibition"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("I_end", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,1]", style={
                                    "padding": "0 12px"}),
                            html.Td(("End value of response inhibition"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("lambda_I", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,1]", style={
                                    "padding": "0 12px"}),
                            html.Td((
                                "Speed at which "
                                "response inhibition changes across "
                                "development"),
                                style={"padding": "0 12px"},),
                        ]),
                        html.Tr([
                            html.Td("midpoint_I", style={
                                    "padding": "0 12px"}),
                            html.Td("[0,100]", style={
                                    "padding": "0 12px"}),
                            html.Td((
                                "Episode at which I reaches 50% "
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
                        ], style=config.bottomrule),
                    ],
                    style=config.table_style,
                )
            ], style={"paddingBottom": "20px"}),

            # ---------------------- SLIDERS ---------------------------------#

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
                        value=state.get("lambda_A"),
                        marks={float(
                            v): "" for v in lambda_A_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        dots=False,
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "lambda_A",
                        },
                        persistence=True,
                    ),
                ], style={"width": "100%",
                          "display": "none",
                          "padding": "0 10px", }),
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
                        value=state.get("eta"),
                        marks={float(
                            v): "" for v in eta_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "eta"
                        },
                        persistence=True,
                    ),
                ], style={"width": "100%",
                          "display": "none",
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
                        value=state.get("gamma"),
                        marks={float(
                            v): "" for v in gamma_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "gamma"
                        },
                        persistence=False,
                    ),
                ], style={"width": "100%",
                          "display": "none",
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
                        value=state.get("alpha"),
                        marks={float(
                            v): "" for v in alpha_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "alpha",
                        },
                        persistence=True,
                    ),
                ], style={"width": "100%",
                          "display": "none",
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
                        value=state.get("kappa"),
                        marks={float(
                            v): "" for v in kappa_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "kappa",
                        },
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
                        value=state.get("C_start"),
                        marks={float(
                            v): "" for v in C_start_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "C_start",
                        },
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
                        value=state.get("C_end"),
                        marks={float(
                            v): "" for v in C_end_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "C_end",
                        },
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
                        value=state.get("lambda_C"),
                        marks={float(
                            v): "" for v in lambda_C_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "lambda_C",
                        },
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
                        value=state.get("midpoint_C"),
                        marks={float(
                            v): "" for v in midpoint_C_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "midpoint_C",
                        },
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
                        value=state.get("I_start"),
                        marks={float(
                            v): "" for v in I_start_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "I_start",
                        },
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
                        value=state.get("I_end"),
                        marks={float(
                            v): "" for v in I_end_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "I_end",
                        },
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
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "lambda_I"
                        },
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
                        value=state.get("midpoint_I"),
                        marks={float(
                            v): "" for v in midpoint_I_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "midpoint_I"
                        },
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
                        value=state.get("w_v_A"),
                        marks={float(
                            v): "" for v in w_v_A_vals},
                        tooltip={
                            "always_visible": True, "placement": "bottom"},
                        updatemode="drag",
                        id={
                            "type": "control",
                            "page": "simulation200",
                            "name": "w_v_A"
                        },
                        persistence=True,
                    ),
                ], style={"width": "100%",
                          "display": "inline-block",
                          "padding": "0 10px"}),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
                      "align-items": "center", "gap": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id={
                        "type": "control",
                        "page": "simulation200",
                        "name": "iteration",
                    },
                    options=iteration_vals,
                    value=state.get("iteration"),
                    clearable=False,
                    style={
                        "width": "200px"},
                    persistence=True,
                ),
                dcc.Dropdown(
                    id={
                        "type": "control",
                        "page": "simulation200",
                        "name": "environment_type",
                    },
                    options=environment_type_vals,
                    value=state.get("environment_type"),
                    clearable=False,
                    style={
                        "width": "200px"},
                    persistence=True,
                ),
            ], style={"display": "flex", "gap": "12px", "paddingTop": "20px"}),
            html.Div([
                dcc.Graph(
                    id={
                        "type": "graph",
                        "page": "simulation200",
                        "name": "content"
                    },
                    config={"responsive": True})
            ], style={"width": "60%", "height": "80vh"})
        ])
    ]

    return layout


@callback(
    Output("main-url", "hash", allow_duplicate=True),
    Input("main-url", "pathname"),
    Input({"type": "control", "page": ALL, "name": ALL}, "value"),
    State({"type": "control", "page": ALL, "name": ALL}, "id"),
    State("main-url", "hash"),
    prevent_initial_call="initial_duplicate",
)
def update_hash(pathname, values, ids, current_hash):
    """Update the URL hash with the current app state."""

    page = pathname.strip("/")

    if not page:
        print("update_hash prevents update because no inputs at home page!")
        raise PreventUpdate

    print(f"update_hash: at page '{page}'")

    # only keep controls of current page
    state = {
        comp_id["name"]: value
        for comp_id, value in zip(ids, values)
        if comp_id["page"] == page
    }

    if not state:
        print("update_hash prevents Update because state is empty!")
        raise PreventUpdate

    print(f"update_hash: encoding state: {state}")

    new_hash = "#" + base64.b64encode(json.dumps(state).encode()).decode()

    if new_hash == current_hash:
        print("update_hash prevents Update because new_has == current_hash!")
        raise PreventUpdate

    return new_hash


@callback(
    Output(f"{page_prefix + page_id}-store", "data"),
    Input("main-url", "hash"),
    State(f"{page_prefix + page_id}-store", "data"),
    prevent_initial_call=False,
)
def load_hash_to_store(
    hash_value,
    current_store,
):
    """
    Restore sliders/selectors from URL hash if present.
    If no hash or invalid, fallback to persisted values.
    """

    if not hash_value:
        # No hash → use persisted values
        print("load_hash_to_store() prevents Update! no hash")
        raise PreventUpdate

    try:
        decoded = json.loads(base64.b64decode(hash_value[1:]))
    except Exception:
        # Invalid hash → fallback to persisted values
        print("load_hash_to_store() prevents Update! invalid hash")
        raise PreventUpdate

    print(f"load_hash_to_store decoded: {decoded}")

    new_store = {**defaults, **decoded}

    if new_store == current_store:
        raise dash.exceptions.PreventUpdate
    return new_store


@callback(
    Output({"type": "control", "page": f"{page_prefix + page_id}",
           "name": ALL}, "value"),
    Input(f"{page_prefix + page_id}-store", "data"),
)
def sync_sliders(store):
    print(f"sync_sliders() store: {store}")

    if not store:
        print("sync_sliders() prevents update because no store!")
        raise dash.exceptions.PreventUpdate

    # Make sure the order is correct
    # names = [item["id"]["name"] for item in ctx.outputs_list]
    # print(names)

    return [
        store["lambda_A"],
        store["eta"],
        store["gamma"],
        store["alpha"],
        store["kappa"],
        store["C_start"],
        store["C_end"],
        store["lambda_C"],
        store["midpoint_C"],
        store["I_start"],
        store["I_end"],
        store["lambda_I"],
        store["midpoint_I"],
        store["w_v_A"],
        store["iteration"],
        store["environment_type"],
    ]


@callback(
    Output({"type": "graph", "name": "content", "page": "simulation200"},
           "figure"),
    Input(f"{page_prefix + page_id}-store", "data"),
    prevent_initial_call=True,
)
def update_graph(store):

    lambda_A, eta, gamma, alpha, kappa, C_start, C_end, lambda_C, midpoint_C, \
        I_start, I_end, lambda_I, midpoint_I, w_v_A, iteration, \
        environment_type = (
            store[k] for k in [
                "lambda_A", "eta", "gamma", "alpha", "kappa",
                "C_start", "C_end", "lambda_C", "midpoint_C",
                "I_start", "I_end", "lambda_I", "midpoint_I",
                "w_v_A", "iteration", "environment_type"
            ]
        )

    print("updating graph")

    if iteration == "expected":
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
                       "M_S_mean", "M_S_std", "C", "v_mean", "v_std", "I",
                       "p_A_mean", "p_A_std"]

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

        # shaded bounds
        V_upper = dff["V_mean"] + dff["V_std"]
        V_lower = dff["V_mean"] - dff["V_std"]
        M_A_upper = dff["M_A_mean"] + dff["M_A_std"]
        M_A_lower = dff["M_A_mean"] - dff["M_A_std"]
        M_S_upper = dff["M_S_mean"] + dff["M_S_std"]
        M_S_lower = dff["M_S_mean"] - dff["M_S_std"]
        v_upper = dff["v_mean"] + dff["v_std"]
        v_lower = dff["v_mean"] - dff["v_std"]
        # p_A_upper = dff["p_A_mean"] + dff["p_A_std"]
        # p_A_lower = dff["p_A_mean"] - dff["p_A_std"]

        fig = make_subplots(
            rows=5,
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
            title_text="C and I",
            range=[0, 1],
            row=2,
            col=1,
        )

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

        # --- Probability aggressive ---

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["p_A_mean"],
                mode="lines",
                name="Probability aggressive",
                line=dict(color="magenta"),
            ),
            row=5,
            col=1,
        )

        # fig.add_trace(
        #     go.Scatter(
        #         x=np.concatenate([dff["Step"], dff["Step"][::-1]]),
        #         y=np.concatenate([p_A_upper, p_A_lower[::-1]]),
        #         fill="toself",
        #         fillcolor="rgba(255,0,255,0.2)",
        #         line=dict(color="rgba(255,0,255,0)"),
        #         hoverinfo="skip",
        #         showlegend=False,
        #     ),
        #     row=5,
        #     col=1,
        # )

        fig.update_yaxes(
            title_text="Probability aggressive",
            range=[0, 1],
            row=5,
            col=1,
        )

        fig.update_xaxes(title_text="Episode", row=5, col=1)

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

        cols_needed = ["Step", "V", "M_A", "M_S", "C", "v", "I", "p_A"]

        # read only filtered rows and selected columns
        table = pq.read_table(
            os.path.join(
                config.DATA_DIR,
                "200_development",
                f"200_development_{iteration}.parquet"
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

        fig = make_subplots(
            rows=5,
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
            title_text="C and I",
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

        # --- Probability aggressive ---

        fig.add_trace(
            go.Scatter(
                x=dff["Step"],
                y=dff["p_A"],
                mode="lines",
                name="Probability aggressive",
                line=dict(color="magenta"),
            ),
            row=5,
            col=1,
        )

        fig.update_yaxes(
            title_text="Probability aggressive",
            range=[0, 1],
            row=5,
            col=1,
        )

        fig.update_xaxes(title_text="Episode", row=5, col=1)

        return fig
