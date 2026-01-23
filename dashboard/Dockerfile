# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install build dependencies for NumPy
RUN apt-get update && \
    apt-get install -y build-essential gfortran git curl python3-dev \
    libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel build

# Copy requirements
COPY requirements.txt ./requirements.txt

# Install numpy first from source with baseline=none
RUN pip install --no-cache-dir --no-binary=:all: numpy

# Install the rest of requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . ./

# Expose the port Dash will run on
EXPOSE 80

# Use gunicorn to serve the Dash app
# app.py must define `server = app.server`
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:server", "--workers=2", "--threads=4", "--timeout=120"]
