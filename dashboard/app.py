from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
import webbrowser
import threading
import config

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.server.config["PROPAGATE_EXCEPTIONS"] = True


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Simulations", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    page["name"],
                    href=page["path"],
                    active="exact",
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(dash.page_container, style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])

# Expose the Flask server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run(debug=False,
            host="127.0.0.1",
            port=8050,
            )
