import os

DATA_DIR = os.getenv(
    "APP_DATA_DIR",
    os.path.join(os.path.dirname(__file__),
                 # "/app/data")
                 "data")
)

# CSS style

table_style = {
    "borderCollapse": "collapse",
    "borderSpacing": "0 6px",
}

toprule = {"borderTop": "2px solid black"}
midrule = {"borderBottom": "1px solid black"}
bottomrule = {"borderBottom": "2px solid black"}


plot_and_param_box_style = {
    "display": "flex",
    "alignItems": "flex-start",
    "gap": "80px",
}

slider_div_style = {
    "width": "100%",
    "display": "inline-block",
    "paddingBottom": "20px",
}

hidden_style = {
    "display": "none",
}

param_config_box_style = {
    "width": "20%",
    "marginTop": "80px",
}
