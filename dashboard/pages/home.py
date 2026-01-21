import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    html.H1(children="Formal Model of Irritability",
            style={"textAlign": "center"}),
    html.Div([
        html.P(children="Welcome to the model explorer dashboard.")
    ])
])
