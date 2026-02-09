#!/bin/bash
# Start script for Data Insight Platform

echo "Starting Data Insight Platform..."
echo ""

if [ -d "ENV" ]; then
    echo "Activating virtual environment..."
    source ENV/bin/activate
fi

if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r platform_requirements.txt
fi

echo "Starting platform..."
echo "The platform will open at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run data_insight_platform.py
