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
    "borderSpacing": "0 6px"
}

toprule = {"borderTop": "2px solid black"}
midrule = {"borderBottom": "1px solid black"}
bottomrule = {"borderBottom": "2px solid black"}
