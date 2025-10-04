#!/bin/bash

# Stock Price Prediction Web Application Startup Script

echo "=============================================="
echo "Stock Price Prediction Web App"
echo "=============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ All dependencies installed"
echo ""
echo "=============================================="
echo "Starting Flask Web Application..."
echo "=============================================="
echo ""
echo "🌐 Web Interface: http://localhost:5000"
echo "🔌 API Health Check: http://localhost:5000/api/health"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Run the application
python app.py
