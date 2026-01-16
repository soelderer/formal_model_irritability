from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import numpy as np
import pandas as pd

df = pd.read_csv('data/001_fnr_impulse_response.csv')

theta_vals = sorted(df["theta_A_w1"].unique())
C_vals = sorted(df["C"].unique())

app = Dash()
app.server.config["PROPAGATE_EXCEPTIONS"] = True


# Requires Dash 2.17.0 or later
app.layout = [
    html.H1(children='Formal Model of Irritability',
            style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Div([
                html.Label("theta_A_w1", style={"textAlign": "center"}),
                dcc.Slider(
                    min=min(theta_vals),
                    max=max(theta_vals),
                    step=None,
                    value=theta_vals[0],
                    marks={int(v): "" for v in theta_vals},
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                    dots=False,
                    id="theta_A_w1-slider"
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
                    id="C-slider"
                ),
            ], style={"width": "20%",
                      "display": "inline-block",
                      "padding": "0 10px"}),
        ], style={"display": "flex", "align-items": "center", "gap": "10px"}),
        html.Div([
            dcc.Graph(id='graph-content')
        ], style={"width": "45%"})
    ])
]

@callback(
    Output("graph-content", "figure"),
    Input("theta_A_w1-slider", "value"),
    Input("C-slider", "value")
)

def update_graph(theta_A_w1, C):
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

if __name__ == '__main__':
    app.run(debug=True)
