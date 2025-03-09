#!/bin/bash

# Ensure the data directory exists
mkdir -p data/uploads

# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Generate a random secret key
export FLASK_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(16))')

# Run the Flask application
echo "Starting Lipid AI Platform..."
echo "Access the application at: http://127.0.0.1:5003"
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py