#!/bin/bash
# ECHO HANDS — One-Click Streamlit Launcher

cd "$(dirname "$0")"

# Set Streamlit config directory to local project .streamlit folder to bypass global permission errors and prompts
export STREAMLIT_CONFIG_DIR="$(pwd)/.streamlit"
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"

# Free port 8501 if an old instance is hanging
lsof -ti :8501 | xargs kill -9 2>/dev/null || true

echo "Starting ECHO HANDS on http://localhost:8501 ..."
.venv/bin/streamlit run app.py --browser.gatherUsageStats=false
