#!/usr/bin/env bash
set -e

# Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "Python3 not found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv

    echo "Activating venv and installing requirements..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # Create a marker file to indicate requirements are installed
    touch venv/.requirements_installed
else
    echo "Virtual environment exists. Activating..."
    source venv/bin/activate

    # Optional: check if marker exists
    if [ ! -f venv/.requirements_installed ]; then
        echo "Installing requirements (first run after venv creation)..."
        pip install --upgrade pip
        pip install -r requirements.txt
        touch venv/.requirements_installed
    fi
fi

# Launch the app
echo "Starting dashboard..."
python app.py
