import os

DATA_DIR = os.getenv(
    "APP_DATA_DIR",
    os.path.join(os.path.dirname(__file__),
                 "/app/data")
)
