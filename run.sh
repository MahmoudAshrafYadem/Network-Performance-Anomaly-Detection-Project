#!/bin/bash

################################################################################
#                    NETWORK PERFORMANCE ANOMALY DETECTION
#                         Quick Run Script
# 
# Usage: bash run.sh [option]
#   --all        - Run complete pipeline and launch dashboard (default)
#   --setup      - Install dependencies only
#   --analyze    - Run analysis pipeline only
#   --notebook   - Execute Jupyter notebook only
#   --dashboard  - Launch Streamlit dashboard only
#   --help       - Show this help message
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Python not found!${NC}"
    exit 1
fi

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    # Create requirements.txt if needed
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << 'EOF'
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
    
    $PYTHON_CMD -m pip install --upgrade pip --quiet
    $PYTHON_CMD -m pip install -r requirements.txt --quiet
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to run analysis
run_analysis() {
    echo -e "${BLUE}Running analysis pipeline...${NC}"
    
    mkdir -p outputs data logs
    
    # Check for data file
    DATA_FOUND=false
    for path in "data/Performance.csv" "Performance.csv" "../Performance.csv"; do
        if [ -f "$path" ]; then
            DATA_FOUND=true
            break
        fi
    done
    
    if [ "$DATA_FOUND" = false ]; then
        echo -e "${RED}✗ Performance.csv not found!${NC}"
        exit 1
    fi
    
    # Run analysis
    if [ -f "run_analysis.py" ]; then
        $PYTHON_CMD run_analysis.py 2>&1 | tee logs/analysis_log.txt
    else
        # Inline analysis with PyTorch
        $PYTHON_CMD << 'PYTHON_SCRIPT'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

os.makedirs('outputs', exist_ok=True)

print("Loading data...")
data_path = None
for p in ['data/Performance.csv', 'Performance.csv', '../Performance.csv']:
    if os.path.exists(p):
        data_path = p
        break

df = pd.read_csv(data_path)
df['aggregate_date'] = pd.to_datetime(df['aggregate_date'])

# Filter
df = df[df['sample_count'] >= 5]
df = df[df['place_type'] == 'locality']
df = df[df['aggregation_period'] == 'Day']
print(f"Filtered to {len(df)} records")

# Features
features = ['mean_download_kbps', 'mean_upload_kbps', 'mean_latency_ms']
X = df[features].fillna(df[features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# Isolation Forest
print("Training Isolation Forest...")
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_scaled)
df['iso_anomaly'] = (iso.predict(X_scaled) == -1).astype(int)
df['iso_score'] = iso.decision_function(X_scaled)

# One-Class SVM
print("Training One-Class SVM...")
sample_idx = np.random.choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_scaled[sample_idx])
df['ocsvm_anomaly'] = (ocsvm.predict(X_scaled) == -1).astype(int)
df['ocsvm_score'] = ocsvm.decision_function(X_scaled)

# Autoencoder with PyTorch
print("Training Autoencoder...")
if PYTORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.ReLU()
            )
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
    
    X_tensor = torch.FloatTensor(X_mm)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = Autoencoder(X_mm.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    autoencoder.train()
    for epoch in range(50):
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
    
    autoencoder.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        recon = autoencoder(X_tensor).cpu().numpy()
    
    print("PyTorch Autoencoder trained")
else:
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8, 2, 8, 16), activation='relu',
                       max_iter=200, random_state=42)
    X_train, X_test = train_test_split(X_mm, test_size=0.2, random_state=42)
    mlp.fit(X_train, X_train)
    recon = mlp.predict(X_mm)

mse = np.mean(np.power(X_mm - recon, 2), axis=1)
threshold = np.percentile(mse, 95)
df['ae_mse'] = mse
df['ae_anomaly'] = (mse > threshold).astype(int)
df['ae_threshold'] = threshold

# Combine
df['all_methods'] = df['iso_anomaly'] + df['ocsvm_anomaly'] + df['ae_anomaly']

# Worst score
df['download_score'] = (df['mean_download_kbps'].max() - df['mean_download_kbps']) / df['mean_download_kbps'].max()
df['upload_score'] = (df['mean_upload_kbps'].max() - df['mean_upload_kbps']) / df['mean_upload_kbps'].max()
df['latency_score'] = df['mean_latency_ms'] / df['mean_latency_ms'].max()
df['worst_score'] = 0.4*df['download_score'] + 0.2*df['upload_score'] + 0.3*df['latency_score'] + 0.1*df['all_methods']/3

# Anomaly log
anomaly_log = df[(df['iso_anomaly']==1) | (df['ocsvm_anomaly']==1) | (df['ae_anomaly']==1)].copy()
def get_methods(r):
    m = []
    if r['iso_anomaly']==1: m.append('IsolationForest')
    if r['ocsvm_anomaly']==1: m.append('OC-SVM')
    if r['ae_anomaly']==1: m.append('Autoencoder')
    return ', '.join(m)
anomaly_log['detection_methods'] = anomaly_log.apply(get_methods, axis=1)
anomaly_log['methods_count'] = anomaly_log['iso_anomaly'] + anomaly_log['ocsvm_anomaly'] + anomaly_log['ae_anomaly']

# Save
df.to_csv('processed_data_with_anomalies.csv', index=False)
df.to_csv('outputs/processed_data_with_anomalies.csv', index=False)
anomaly_log.to_csv('anomaly_log.csv', index=False)
anomaly_log.to_csv('outputs/anomaly_log.csv', index=False)

print(f"✓ Analysis complete: {len(df)} records, {len(anomaly_log)} anomalies")
PYTHON_SCRIPT
    fi
    
    echo -e "${GREEN}✓ Analysis complete${NC}"
}

# Function to run notebook
run_notebook() {
    echo -e "${BLUE}Executing Jupyter notebook...${NC}"
    
    NOTEBOOK=""
    for nb in "Anomaly_Detection_Network_Performance.ipynb" "anomaly_detection_analysis.ipynb"; do
        if [ -f "$nb" ]; then
            NOTEBOOK="$nb"
            break
        fi
    done
    
    if [ -n "$NOTEBOOK" ]; then
        $PYTHON_CMD -m pip install nbconvert --quiet 2>/dev/null
        $PYTHON_CMD -m nbconvert --to notebook --execute "$NOTEBOOK" \
            --output "executed_notebook.ipynb" \
            --ExecutePreprocessor.timeout=600
        echo -e "${GREEN}✓ Notebook executed${NC}"
    else
        echo -e "${YELLOW}⚠ No notebook found${NC}"
    fi
}

# Function to launch dashboard
launch_dashboard() {
    echo -e "${BLUE}Launching Streamlit dashboard...${NC}"
    
    if [ -f "app.py" ]; then
        echo -e "${CYAN}Dashboard will open in your browser. Press Ctrl+C to stop.${NC}"
        streamlit run app.py --server.headless=true
    else
        echo -e "${RED}✗ app.py not found${NC}"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo -e "${CYAN}Network Performance Anomaly Detection - Run Script${NC}"
    echo ""
    echo "Usage: bash run.sh [option]"
    echo ""
    echo "Options:"
    echo "  --all        Run complete pipeline and launch dashboard (default)"
    echo "  --setup      Install dependencies only"
    echo "  --analyze    Run analysis pipeline only"
    echo "  --notebook   Execute Jupyter notebook only"
    echo "  --dashboard  Launch Streamlit dashboard only"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash run.sh              # Run everything"
    echo "  bash run.sh --setup      # Just install dependencies"
    echo "  bash run.sh --analyze    # Run analysis without dashboard"
    echo "  bash run.sh --dashboard  # Launch dashboard (after running analysis)"
}

# Main logic
case "${1:---all}" in
    --all)
        install_dependencies
        run_analysis
        run_notebook
        launch_dashboard
        ;;
    --setup)
        install_dependencies
        ;;
    --analyze)
        run_analysis
        run_notebook
        ;;
    --notebook)
        run_notebook
        ;;
    --dashboard)
        launch_dashboard
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac
