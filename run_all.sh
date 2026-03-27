#!/bin/bash

# ============================================
# Network Anomaly Detection Project
# Complete Run Script
# ============================================

set -e  # Exit on error

echo "============================================"
echo "  Network Anomaly Detection Project"
echo "============================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "[Step 1/4] Installing requirements..."
echo "--------------------------------------------"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "Requirements installed successfully!"
echo ""

echo "[Step 2/4] Running Jupyter Notebook..."
echo "--------------------------------------------"
jupyter nbconvert --to notebook --execute "Network_Anomaly_Detection.ipynb" \
    --output "Network_Anomaly_Detection_executed.ipynb" \
    --ExecutePreprocessor.timeout=600
echo "Notebook executed successfully!"
echo ""

echo "[Step 3/4] Verifying output files..."
echo "--------------------------------------------"
if [ -d "outputs" ]; then
    echo "Output files generated:"
    ls -la outputs/
else
    echo "ERROR: outputs directory not found!"
    exit 1
fi
echo ""

echo "[Step 4/4] Launching Streamlit Dashboard..."
echo "--------------------------------------------"
echo "Starting dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""
streamlit run app.py --server.headless true
