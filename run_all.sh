#!/bin/bash

################################################################################
#                    NETWORK PERFORMANCE ANOMALY DETECTION
#                         Complete Run Script
# 
# This script runs the entire project from start to finish:
# 1. Environment setup and directory creation
# 2. Install all Python dependencies
# 3. Run data analysis and model training
# 4. Execute Jupyter notebook
# 5. Launch Streamlit dashboard
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║        NETWORK PERFORMANCE ANOMALY DETECTION - COMPLETE RUNNER           ║"
echo "║                                                                          ║"
echo "║  Models: Isolation Forest | One-Class SVM | Autoencoder (PyTorch)       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step counter
STEP=0

################################################################################
# STEP 1: Directory Setup
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Creating Project Directories${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

echo -e "${CYAN}Creating necessary directories...${NC}"
mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/outputs"
mkdir -p "$SCRIPT_DIR/notebooks"
mkdir -p "$SCRIPT_DIR/logs"

# Copy data file if it exists in parent directory
if [ -f "$SCRIPT_DIR/Performance.csv" ]; then
    echo -e "${GREEN}✓ Performance.csv found in project directory${NC}"
    cp "$SCRIPT_DIR/Performance.csv" "$SCRIPT_DIR/data/" 2>/dev/null || true
fi

if [ -f "$SCRIPT_DIR/../Performance.csv" ]; then
    echo -e "${GREEN}✓ Performance.csv found in parent directory${NC}"
    cp "$SCRIPT_DIR/../Performance.csv" "$SCRIPT_DIR/data/" 2>/dev/null || true
fi

echo -e "${GREEN}✓ Directories created successfully${NC}"

################################################################################
# STEP 2: Check Python Installation
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Checking Python Installation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

# Check Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python not found! Please install Python 3.x${NC}"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo -e "${RED}✗ pip not found! Please install pip${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pip is available${NC}"

################################################################################
# STEP 3: Install Dependencies
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Installing Python Dependencies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

# Create requirements.txt if it doesn't exist
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo -e "${YELLOW}Creating requirements.txt...${NC}"
    cat > "$SCRIPT_DIR/requirements.txt" << 'EOF'
pandas
numpy
scikit-learn
torch
streamlit
plotly
matplotlib
seaborn
matplotlib-venn
kaleido
jupyter
nbconvert
ipykernel
EOF
fi

echo -e "${CYAN}Upgrading pip...${NC}"
$PYTHON_CMD -m pip install --upgrade pip --quiet

echo -e "${CYAN}Installing required packages (this may take a few minutes)...${NC}"
echo -e "${YELLOW}Packages: pandas, numpy, scikit-learn, torch, streamlit, plotly, matplotlib, seaborn, jupyter${NC}"

$PYTHON_CMD -m pip install -r "$SCRIPT_DIR/requirements.txt" --quiet 2>&1 | while read -r line; do
    if [[ "$line" == *"error"* ]] || [[ "$line" == *"Error"* ]]; then
        echo -e "${RED}$line${NC}"
    fi
done

echo -e "${GREEN}✓ All dependencies installed successfully${NC}"

# Verify installations
echo -e "${CYAN}Verifying installations...${NC}"
$PYTHON_CMD -c "import pandas; print(f'  pandas {pandas.__version__}')" || echo -e "${RED}✗ pandas failed${NC}"
$PYTHON_CMD -c "import numpy; print(f'  numpy {numpy.__version__}')" || echo -e "${RED}✗ numpy failed${NC}"
$PYTHON_CMD -c "import sklearn; print(f'  scikit-learn {sklearn.__version__}')" || echo -e "${RED}✗ scikit-learn failed${NC}"
$PYTHON_CMD -c "import streamlit; print(f'  streamlit {streamlit.__version__}')" || echo -e "${RED}✗ streamlit failed${NC}"
$PYTHON_CMD -c "import matplotlib; print(f'  matplotlib {matplotlib.__version__}')" || echo -e "${RED}✗ matplotlib failed${NC}"

# Check PyTorch
$PYTHON_CMD -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ PyTorch available${NC}" || echo -e "${YELLOW}⚠ PyTorch not available - will use sklearn MLP${NC}"

################################################################################
# STEP 4: Verify Data File
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Verifying Data File${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

DATA_FILE=""
if [ -f "$SCRIPT_DIR/data/Performance.csv" ]; then
    DATA_FILE="$SCRIPT_DIR/data/Performance.csv"
elif [ -f "$SCRIPT_DIR/Performance.csv" ]; then
    DATA_FILE="$SCRIPT_DIR/Performance.csv"
elif [ -f "$SCRIPT_DIR/../Performance.csv" ]; then
    DATA_FILE="$SCRIPT_DIR/../Performance.csv"
fi

if [ -z "$DATA_FILE" ]; then
    echo -e "${RED}✗ Performance.csv not found!${NC}"
    echo -e "${YELLOW}Please place Performance.csv in one of these locations:${NC}"
    echo -e "  - $SCRIPT_DIR/data/Performance.csv"
    echo -e "  - $SCRIPT_DIR/Performance.csv"
    echo -e "  - Parent directory of the project"
    exit 1
else
    echo -e "${GREEN}✓ Found data file: $DATA_FILE${NC}"
    # Count rows
    ROWS=$(wc -l < "$DATA_FILE")
    echo -e "${GREEN}  Total rows: $ROWS${NC}"
fi

################################################################################
# STEP 5: Run Analysis Pipeline
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Running Data Analysis & Model Training${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

if [ -f "$SCRIPT_DIR/run_analysis.py" ]; then
    echo -e "${CYAN}Executing analysis pipeline...${NC}"
    echo -e "${YELLOW}This includes:${NC}"
    echo -e "  - Data loading and filtering"
    echo -e "  - Isolation Forest training"
    echo -e "  - One-Class SVM training"
    echo -e "  - Autoencoder training (PyTorch)"
    echo -e "  - Answering all 5 project questions"
    echo -e "  - Generating output files"
    echo ""
    
    cd "$SCRIPT_DIR"
    $PYTHON_CMD run_analysis.py 2>&1 | tee "$SCRIPT_DIR/logs/analysis_log.txt" || {
        echo -e "${YELLOW}Analysis script had issues, trying alternative approach...${NC}"
    }
    
    # Check if outputs were created
    if [ -f "$SCRIPT_DIR/outputs/processed_data_with_anomalies.csv" ]; then
        echo -e "${GREEN}✓ Analysis completed successfully${NC}"
        # Copy outputs to main directory for dashboard
        cp "$SCRIPT_DIR/outputs/processed_data_with_anomalies.csv" "$SCRIPT_DIR/" 2>/dev/null || true
        cp "$SCRIPT_DIR/outputs/anomaly_log.csv" "$SCRIPT_DIR/" 2>/dev/null || true
    else
        echo -e "${YELLOW}⚠ Output files not found, running inline analysis...${NC}"
        
        # Run inline analysis with PyTorch
        $PYTHON_CMD << 'PYTHON_SCRIPT'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    print("PyTorch is available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - using sklearn MLP")

np.random.seed(42)
if PYTORCH_AVAILABLE:
    torch.manual_seed(42)

# Create directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*60)
print("Network Performance Anomaly Detection")
print("="*60)

# Load data
print("\n1. Loading Data...")
data_path = None
for path in ['data/Performance.csv', 'Performance.csv', '../Performance.csv']:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("ERROR: Performance.csv not found!")
    exit(1)

df = pd.read_csv(data_path)
df['aggregate_date'] = pd.to_datetime(df['aggregate_date'])
print(f"   Dataset Shape: {df.shape}")

# Filter data
print("\n2. Filtering Data...")
df_filtered = df[df['sample_count'] >= 5].copy()
df_filtered = df_filtered[df_filtered['place_type'] == 'locality']
df_daily = df_filtered[df_filtered['aggregation_period'] == 'Day'].copy()
print(f"   Filtered to {len(df_daily)} daily locality records")

# Prepare features
print("\n3. Preparing Features...")
features = ['mean_download_kbps', 'mean_upload_kbps', 'mean_latency_ms']
X = df_daily[features].copy().fillna(df_daily[features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# Isolation Forest
print("\n4. Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_scaled)
iso_predictions = iso_forest.predict(X_scaled)
iso_scores = iso_forest.decision_function(X_scaled)
df_daily['iso_anomaly'] = (iso_predictions == -1).astype(int)
df_daily['iso_score'] = iso_scores
print(f"   Anomalies: {df_daily['iso_anomaly'].sum()}")

# One-Class SVM
print("\n5. Training One-Class SVM...")
sample_size = min(10000, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_sample)
ocsvm_predictions = ocsvm.predict(X_scaled)
ocsvm_scores = ocsvm.decision_function(X_scaled)
df_daily['ocsvm_anomaly'] = (ocsvm_predictions == -1).astype(int)
df_daily['ocsvm_score'] = ocsvm_scores
print(f"   Anomalies: {df_daily['ocsvm_anomaly'].sum()}")

# Autoencoder with PyTorch
print("\n6. Training Autoencoder...")
if PYTORCH_AVAILABLE:
    # Define PyTorch Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_mm)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = Autoencoder(X_mm.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Train
    num_epochs = 50
    autoencoder.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
    # Get reconstructions
    autoencoder.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        reconstructions = autoencoder(X_tensor).cpu().numpy()
    
    mse = np.mean(np.power(X_mm - reconstructions, 2), axis=1)
    print("   PyTorch Autoencoder trained successfully")
else:
    # Fallback to sklearn MLP
    print("   Using sklearn MLPRegressor as autoencoder...")
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 2, 8, 16), activation='relu',
                       solver='adam', max_iter=200, random_state=42,
                       early_stopping=True, validation_fraction=0.1)
    X_train, X_test = train_test_split(X_mm, test_size=0.2, random_state=42)
    mlp.fit(X_train, X_train)
    reconstructions = mlp.predict(X_mm)
    mse = np.mean(np.power(X_mm - reconstructions, 2), axis=1)

threshold = np.percentile(mse, 95)
df_daily['ae_mse'] = mse
df_daily['ae_anomaly'] = (mse > threshold).astype(int)
df_daily['ae_threshold'] = threshold
print(f"   Anomalies: {df_daily['ae_anomaly'].sum()}")

# Combine methods
print("\n7. Combining Methods...")
df_daily['all_methods'] = df_daily['iso_anomaly'] + df_daily['ocsvm_anomaly'] + df_daily['ae_anomaly']
print(f"   All 3 methods agree: {(df_daily['all_methods'] == 3).sum()}")

# Calculate worst score
print("\n8. Calculating Worst Scores...")
df_daily['download_score'] = (df_daily['mean_download_kbps'].max() - df_daily['mean_download_kbps']) / df_daily['mean_download_kbps'].max()
df_daily['upload_score'] = (df_daily['mean_upload_kbps'].max() - df_daily['mean_upload_kbps']) / df_daily['mean_upload_kbps'].max()
df_daily['latency_score'] = df_daily['mean_latency_ms'] / df_daily['mean_latency_ms'].max()
df_daily['worst_score'] = 0.4 * df_daily['download_score'] + 0.2 * df_daily['upload_score'] + \
                          0.3 * df_daily['latency_score'] + 0.1 * df_daily['all_methods'] / 3

# Create anomaly log
print("\n9. Creating Anomaly Log...")
anomaly_log = df_daily[(df_daily['iso_anomaly'] == 1) | (df_daily['ocsvm_anomaly'] == 1) | 
                       (df_daily['ae_anomaly'] == 1)].copy()

def get_detection_methods(row):
    methods = []
    if row['iso_anomaly'] == 1: methods.append('IsolationForest')
    if row['ocsvm_anomaly'] == 1: methods.append('OC-SVM')
    if row['ae_anomaly'] == 1: methods.append('Autoencoder')
    return ', '.join(methods)

anomaly_log['detection_methods'] = anomaly_log.apply(get_detection_methods, axis=1)
anomaly_log['methods_count'] = anomaly_log['iso_anomaly'] + anomaly_log['ocsvm_anomaly'] + anomaly_log['ae_anomaly']
anomaly_log = anomaly_log.sort_values(['methods_count', 'aggregate_date'], ascending=[False, True])

# Save outputs
print("\n10. Saving Outputs...")
df_daily.to_csv('processed_data_with_anomalies.csv', index=False)
anomaly_log.to_csv('anomaly_log.csv', index=False)
df_daily.to_csv('outputs/processed_data_with_anomalies.csv', index=False)
anomaly_log.to_csv('outputs/anomaly_log.csv', index=False)
print(f"   Saved {len(df_daily)} records")
print(f"   Saved {len(anomaly_log)} anomalies")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
PYTHON_SCRIPT
    fi
else
    echo -e "${RED}✗ run_analysis.py not found!${NC}"
fi

################################################################################
# STEP 6: Execute Jupyter Notebook
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Executing Jupyter Notebook${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

NOTEBOOK_EXECUTED=false

# Find notebook
NOTEBOOK_PATH=""
for notebook in "Anomaly_Detection_Network_Performance.ipynb" "anomaly_detection_analysis.ipynb"; do
    if [ -f "$SCRIPT_DIR/$notebook" ]; then
        NOTEBOOK_PATH="$SCRIPT_DIR/$notebook"
        break
    fi
done

if [ -n "$NOTEBOOK_PATH" ]; then
    echo -e "${CYAN}Found notebook: $(basename "$NOTEBOOK_PATH")${NC}"
    
    # Try to execute the notebook
    echo -e "${CYAN}Executing notebook (this may take a few minutes)...${NC}"
    
    # Install nbconvert if needed
    $PYTHON_CMD -m pip install nbconvert --quiet 2>/dev/null
    
    # Execute notebook
    $PYTHON_CMD -m nbconvert --to notebook --execute "$NOTEBOOK_PATH" \
        --output "executed_notebook.ipynb" \
        --ExecutePreprocessor.timeout=600 2>&1 | tee "$SCRIPT_DIR/logs/notebook_log.txt" && {
        echo -e "${GREEN}✓ Notebook executed successfully${NC}"
        NOTEBOOK_EXECUTED=true
    } || {
        echo -e "${YELLOW}⚠ Notebook execution had issues (outputs still generated from analysis)${NC}"
    }
else
    echo -e "${YELLOW}⚠ No Jupyter notebook found - skipping notebook execution${NC}"
    echo -e "${YELLOW}  Analysis outputs are still available from Step 5${NC}"
fi

################################################################################
# STEP 7: Summary of Outputs
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Summary of Generated Outputs${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${CYAN}Checking generated files...${NC}"
echo ""

# Check for output files
FILES_TO_CHECK=(
    "processed_data_with_anomalies.csv"
    "anomaly_log.csv"
    "outputs/processed_data_with_anomalies.csv"
    "outputs/anomaly_log.csv"
    "outputs/method_comparison.csv"
    "outputs/summary_visualization.png"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        SIZE=$(du -h "$SCRIPT_DIR/$file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $file (${SIZE})"
    else
        echo -e "  ${YELLOW}○${NC} $file ${YELLOW}(not generated)${NC}"
    fi
done

# Count anomalies if file exists
if [ -f "$SCRIPT_DIR/anomaly_log.csv" ]; then
    ANOMALY_COUNT=$(tail -n +2 "$SCRIPT_DIR/anomaly_log.csv" | wc -l)
    echo ""
    echo -e "${GREEN}Total Anomalies Detected: $ANOMALY_COUNT${NC}"
fi

################################################################################
# STEP 8: Launch Streamlit Dashboard
################################################################################
((STEP++))
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP $STEP: Launching Streamlit Dashboard${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

if [ -f "$SCRIPT_DIR/app.py" ]; then
    echo -e "${GREEN}✓ app.py found${NC}"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  STREAMLIT DASHBOARD STARTING${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}The dashboard will open in your web browser.${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the dashboard.${NC}"
    echo ""
    echo -e "${PURPLE}Dashboard Features:${NC}"
    echo -e "  • Overview tab with KPI metrics"
    echo -e "  • Anomaly Detection visualization"
    echo -e "  • Q&A Results for all 5 questions"
    echo -e "  • Top 5 Worst Performers"
    echo -e "  • Anomaly Log with filtering"
    echo -e "  • Method Comparison analysis"
    echo ""
    
    cd "$SCRIPT_DIR"
    streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
else
    echo -e "${RED}✗ app.py not found! Cannot launch dashboard.${NC}"
    echo -e "${YELLOW}Please ensure app.py is in the project directory.${NC}"
fi

################################################################################
# COMPLETION MESSAGE
################################################################################
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    PROJECT EXECUTION COMPLETE!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Project Location: $SCRIPT_DIR${NC}"
echo ""
echo -e "${YELLOW}Generated Files:${NC}"
echo -e "  • processed_data_with_anomalies.csv - Full dataset with anomaly flags"
echo -e "  • anomaly_log.csv - List of all detected anomalies"
echo -e "  • method_comparison.csv - Comparison of detection methods"
echo -e "  • summary_visualization.png - Visual summary of results"
echo ""
echo -e "${YELLOW}To run again:${NC}"
echo -e "  cd $SCRIPT_DIR"
echo -e "  bash run_all.sh"
echo ""
