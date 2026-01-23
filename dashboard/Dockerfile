# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel build

RUN pip install numpy==2.3.5

# Copy requirements
COPY requirements.txt ./requirements.txt

# Install the rest of requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . ./

# Expose the port Dash will run on
EXPOSE 80

# Use gunicorn to serve the Dash app
# app.py must define `server = app.server`
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:server", "--workers=4", "--threads=4", "--timeout=120"]
