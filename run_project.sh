#!/bin/bash

################################################################################
# Network Performance Anomaly Detection Project
# Master Run Script
# 
# This script runs the complete project:
# 1. Check/Install dependencies
# 2. Run the analysis (data processing + anomaly detection)
# 3. Launch the Streamlit dashboard
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   Network Performance Anomaly Detection Project${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${GREEN}>>> $1${NC}"
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================
# STEP 1: Check Dependencies
# ============================================================
print_section "STEP 1: Checking Dependencies"

# Check Python
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python not found. Please install Python 3.x${NC}"
    exit 1
fi

echo -e "Python found: $PYTHON_CMD"
$PYTHON_CMD --version

# Check pip
if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo -e "${YELLOW}pip not found, attempting to install...${NC}"
    curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
fi

# Install required packages
print_section "STEP 2: Installing Required Packages"

REQUIRED_PACKAGES=(
    "pandas"
    "numpy"
    "matplotlib"
    "seaborn"
    "scikit-learn"
    "streamlit"
    "jupyter"
)

echo "Installing packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    echo -e "  - Installing ${YELLOW}$package${NC}..."
    $PYTHON_CMD -m pip install "$package" --quiet --break-system-packages 2>/dev/null || \
    $PYTHON_CMD -m pip install "$package" --quiet 2>/dev/null || \
    pip install "$package" --quiet --break-system-packages 2>/dev/null || \
    pip install "$package" --quiet 2>/dev/null || {
        echo -e "${RED}Failed to install $package${NC}"
        echo "Trying to continue anyway..."
    }
done

echo -e "${GREEN}Package installation completed!${NC}"

# ============================================================
# STEP 3: Check Data File
# ============================================================
print_section "STEP 3: Checking Data File"

DATA_FILE="Performance.csv"

if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}ERROR: Data file '$DATA_FILE' not found!${NC}"
    echo "Please ensure Performance.csv is in the project directory."
    exit 1
fi

echo -e "${GREEN}Data file found: $DATA_FILE${NC}"
echo "File size: $(du -h "$DATA_FILE" | cut -f1)"
echo "Line count: $(wc -l < "$DATA_FILE")"

# ============================================================
# STEP 4: Run Analysis
# ============================================================
print_section "STEP 4: Running Anomaly Detection Analysis"

ANALYSIS_SCRIPT="run_analysis.py"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo -e "${RED}ERROR: Analysis script '$ANALYSIS_SCRIPT' not found!${NC}"
    exit 1
fi

echo "Running analysis script..."
echo "This may take a few minutes..."
echo ""

$PYTHON_CMD "$ANALYSIS_SCRIPT"

# Check if outputs were created
if [ ! -f "processed_data_with_anomalies.csv" ] || [ ! -f "anomaly_log.csv" ]; then
    echo -e "${RED}ERROR: Analysis did not produce expected output files!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Analysis completed successfully!${NC}"
echo ""
echo "Generated files:"
ls -lh processed_data_with_anomalies.csv anomaly_log.csv method_comparison.csv summary_visualization.png 2>/dev/null || true

# ============================================================
# STEP 5: Summary Statistics
# ============================================================
print_section "STEP 5: Analysis Summary"

echo "Reading results..."
$PYTHON_CMD -c "
import pandas as pd
df = pd.read_csv('processed_data_with_anomalies.csv')
anomaly_log = pd.read_csv('anomaly_log.csv')

print('='*60)
print('ANALYSIS SUMMARY')
print('='*60)
print(f'Total records analyzed: {len(df):,}')
print(f'Total anomalies detected: {len(anomaly_log):,}')
print(f'High confidence (3 methods agree): {(df[\"all_methods\"] == 3).sum():,}')
print('')
print('ANOMALIES BY METHOD:')
print(f'  - Isolation Forest: {df[\"iso_anomaly\"].sum():,}')
print(f'  - One-Class SVM: {df[\"ocsvm_anomaly\"].sum():,}')
print(f'  - Autoencoder: {df[\"ae_anomaly\"].sum():,}')
print('')
print('TOP 5 MOST VARIABLE REGIONS:')
regional = df.groupby('region').agg({'mean_download_kbps': ['mean', 'std']})
regional.columns = ['mean', 'std']
regional['cv'] = regional['std'] / regional['mean'] * 100
regional = regional.sort_values('cv', ascending=False)
for i, (region, row) in enumerate(regional.head(5).iterrows(), 1):
    print(f'  {i}. {region}: CV = {row[\"cv\"]:.1f}%')
print('='*60)
"

# ============================================================
# STEP 6: Launch Dashboard
# ============================================================
print_section "STEP 6: Launch Streamlit Dashboard"

DASHBOARD_FILE="app.py"

if [ ! -f "$DASHBOARD_FILE" ]; then
    echo -e "${RED}ERROR: Dashboard file '$DASHBOARD_FILE' not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Starting Streamlit dashboard...${NC}"
echo ""
echo "============================================================"
echo "  DASHBOARD STARTING"
echo "============================================================"
echo ""
echo "  The dashboard will open in your web browser."
echo "  Press Ctrl+C to stop the server."
echo ""
echo "  If it doesn't open automatically, go to:"
echo -e "  ${BLUE}http://localhost:8501${NC}"
echo ""
echo "============================================================"
echo ""

# Launch Streamlit
streamlit run "$DASHBOARD_FILE" --server.headless=true --browser.gatherUsageStats=false

echo ""
echo -e "${GREEN}Dashboard session ended.${NC}"
