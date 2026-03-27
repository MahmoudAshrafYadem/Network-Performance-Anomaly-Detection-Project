"""
Network Performance Anomaly Detection
Main Analysis Script

This script performs the complete anomaly detection analysis including:
1. Data loading and filtering
2. EDA
3. Isolation Forest, OC-SVM, and Autoencoder (PyTorch) models
4. Answer all 5 project questions
5. Generate outputs for the Streamlit dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
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
    print("PyTorch is available - using PyTorch Autoencoder")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - using sklearn MLP Autoencoder")

# Set random seeds
np.random.seed(42)
if PYTORCH_AVAILABLE:
    torch.manual_seed(42)

# Create output directory
os.makedirs('outputs', exist_ok=True)

print("="*60)
print("Network Performance Anomaly Detection")
print("="*60)

# ===========================================
# 1. DATA LOADING
# ===========================================
print("\n1. Loading Data...")

# Find data file
data_path = None
for path in ['Performance.csv', 'data/Performance.csv', '../data/Performance.csv']:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("ERROR: Performance.csv not found!")
    exit(1)

df = pd.read_csv(data_path)
df['aggregate_date'] = pd.to_datetime(df['aggregate_date'])

print(f"   Dataset Shape: {df.shape}")
print(f"   Date Range: {df['aggregate_date'].min()} to {df['aggregate_date'].max()}")

# ===========================================
# 2. FILTERING DECISIONS
# ===========================================
print("\n2. Applying Filters...")

# Sample count threshold
SAMPLE_COUNT_THRESHOLD = 5
df_filtered = df[df['sample_count'] >= SAMPLE_COUNT_THRESHOLD].copy()
print(f"   Sample count filter: {len(df)} -> {len(df_filtered)} rows")

# Place type filter
df_filtered = df_filtered[df_filtered['place_type'] == 'locality'].copy()
print(f"   Locality filter: {len(df_filtered)} rows")

# Aggregation period filter
df_daily = df_filtered[df_filtered['aggregation_period'] == 'Day'].copy()
print(f"   Daily filter: {len(df_daily)} rows")

# ===========================================
# 3. PREPARE DATA FOR ANOMALY DETECTION
# ===========================================
print("\n3. Preparing Data for Anomaly Detection...")

features = ['mean_download_kbps', 'mean_upload_kbps', 'mean_latency_ms']
X = df_daily[features].copy().fillna(df_daily[features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Also create MinMax scaled version for autoencoder
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

print(f"   Feature matrix shape: {X_scaled.shape}")

# ===========================================
# 4. ISOLATION FOREST
# ===========================================
print("\n4. Training Isolation Forest...")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

iso_predictions = iso_forest.predict(X_scaled)
iso_scores = iso_forest.decision_function(X_scaled)

df_daily['iso_anomaly'] = (iso_predictions == -1).astype(int)
df_daily['iso_score'] = iso_scores

print(f"   Anomalies detected: {df_daily['iso_anomaly'].sum()}")
print(f"   Anomaly rate: {df_daily['iso_anomaly'].mean()*100:.2f}%")

# ===========================================
# 5. ONE-CLASS SVM
# ===========================================
print("\n5. Training One-Class SVM...")

# Use sample for training (OC-SVM is computationally expensive)
sample_size = min(10000, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_sample)

ocsvm_predictions = ocsvm.predict(X_scaled)
ocsvm_scores = ocsvm.decision_function(X_scaled)

df_daily['ocsvm_anomaly'] = (ocsvm_predictions == -1).astype(int)
df_daily['ocsvm_score'] = ocsvm_scores

print(f"   Anomalies detected: {df_daily['ocsvm_anomaly'].sum()}")
print(f"   Anomaly rate: {df_daily['ocsvm_anomaly'].mean()*100:.2f}%")

# ===========================================
# 6. AUTOENCODER (PyTorch)
# ===========================================
print("\n6. Training Autoencoder...")

if PYTORCH_AVAILABLE:
    # Define PyTorch Autoencoder Model
    class Autoencoder(nn.Module):
        """
        PyTorch Autoencoder for Anomaly Detection
        
        Architecture:
        - Encoder: input -> 16 -> 8 -> 2 (bottleneck)
        - Decoder: 2 -> 8 -> 16 -> output
        """
        def __init__(self, input_dim, encoding_dim=2):
            super(Autoencoder, self).__init__()
            
            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, encoding_dim),
                nn.ReLU()
            )
            
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim),
                nn.Sigmoid()  # Output between 0 and 1 (MinMax scaled data)
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
        def get_encoding(self, x):
            """Get the encoded representation"""
            return self.encoder(x)
    
    # Prepare data for PyTorch
    X_tensor = torch.FloatTensor(X_mm)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    autoencoder = Autoencoder(X_mm.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    autoencoder.train()
    
    print(f"   Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            # Forward pass
            outputs = autoencoder(batch_x)
            loss = criterion(outputs, batch_x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Get reconstructions for all data
    autoencoder.eval()
    with torch.no_grad():
        X_tensor_device = X_tensor.to(device)
        reconstructions = autoencoder(X_tensor_device).cpu().numpy()
    
    print("   PyTorch Autoencoder trained successfully")
    
else:
    # Use sklearn MLPRegressor as autoencoder fallback
    print("   Using sklearn MLPRegressor as autoencoder...")
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(16, 8, 2, 8, 16),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    X_train, X_test = train_test_split(X_mm, test_size=0.2, random_state=42)
    mlp.fit(X_train, X_train)
    
    reconstructions = mlp.predict(X_mm)

# Calculate reconstruction error (MSE)
mse = np.mean(np.power(X_mm - reconstructions, 2), axis=1)

# Set threshold at 95th percentile
threshold = np.percentile(mse, 95)

df_daily['ae_mse'] = mse
df_daily['ae_anomaly'] = (mse > threshold).astype(int)
df_daily['ae_threshold'] = threshold

print(f"   Anomalies detected: {df_daily['ae_anomaly'].sum()}")
print(f"   Anomaly rate: {df_daily['ae_anomaly'].mean()*100:.2f}%")

# ===========================================
# 7. METHOD COMBINATION
# ===========================================
print("\n7. Combining Methods...")

df_daily['all_methods'] = (df_daily['iso_anomaly'] + 
                           df_daily['ocsvm_anomaly'] + 
                           df_daily['ae_anomaly'])

print(f"   All 3 methods agree: {(df_daily['all_methods'] == 3).sum()}")
print(f"   Exactly 2 methods: {(df_daily['all_methods'] == 2).sum()}")
print(f"   Exactly 1 method: {(df_daily['all_methods'] == 1).sum()}")

# ===========================================
# 8. Q1: REGIONAL VARIABILITY
# ===========================================
print("\n8. Q1: Regional Variability Analysis...")

regional_stats = df_daily.groupby('region').agg({
    'mean_download_kbps': ['mean', 'std', 'count']
})
regional_stats.columns = ['mean_download', 'std_download', 'count']
regional_stats['cv'] = regional_stats['std_download'] / regional_stats['mean_download'] * 100
regional_stats = regional_stats.sort_values('cv', ascending=False)

print("   Top 5 most variable regions:")
for region, row in regional_stats.head(5).iterrows():
    print(f"   - {region}: CV = {row['cv']:.1f}%")

# ===========================================
# 9. Q2: DAILY ANOMALIES
# ===========================================
print("\n9. Q2: Daily Anomaly Analysis...")

daily_anomalies = df_daily.groupby('aggregate_date').agg({
    'iso_anomaly': 'sum',
    'ocsvm_anomaly': 'sum',
    'ae_anomaly': 'sum',
    'mean_download_kbps': 'mean',
    'mean_latency_ms': 'mean',
    'all_methods': 'sum'
}).reset_index()

print(f"   Analyzed {len(daily_anomalies)} days")

# ===========================================
# 10. Q3: CARRIER CONSISTENCY
# ===========================================
print("\n10. Q3: Carrier Consistency Analysis...")

carrier_region = df_daily.groupby(['carrier_name', 'region'])['mean_download_kbps'].mean().unstack()
print(f"   Analyzed {carrier_region.shape[0]} carriers across {carrier_region.shape[1]} regions")

# ===========================================
# 11. Q4: LTE vs 5G
# ===========================================
print("\n11. Q4: LTE vs 5G Comparison...")

df_lte = df_daily[df_daily['technology_type'] == 'LTE']
df_5g = df_daily[df_daily['technology_type'] == '5G']

lte_anomalies = df_lte['iso_anomaly'].sum() if len(df_lte) > 0 else 0
g5_anomalies = df_5g['iso_anomaly'].sum() if len(df_5g) > 0 else 0

print(f"   LTE anomalies: {lte_anomalies}")
print(f"   5G anomalies: {g5_anomalies}")

# ===========================================
# 12. Q5: WORST PERFORMERS
# ===========================================
print("\n12. Q5: Top 5 Worst Performers...")

# Calculate worst score
df_daily['download_score'] = (df_daily['mean_download_kbps'].max() - df_daily['mean_download_kbps']) / df_daily['mean_download_kbps'].max()
df_daily['upload_score'] = (df_daily['mean_upload_kbps'].max() - df_daily['mean_upload_kbps']) / df_daily['mean_upload_kbps'].max()
df_daily['latency_score'] = df_daily['mean_latency_ms'] / df_daily['mean_latency_ms'].max()

df_daily['worst_score'] = (
    0.4 * df_daily['download_score'] +
    0.2 * df_daily['upload_score'] +
    0.3 * df_daily['latency_score'] +
    0.1 * df_daily['all_methods'] / 3
)

top5_worst = df_daily.nlargest(5, 'worst_score')

print("   Top 5 Worst:")
for i, (_, row) in enumerate(top5_worst.iterrows(), 1):
    print(f"   {i}. {row['region']} - {row['carrier_name']} - {row['aggregate_date'].strftime('%Y-%m-%d')}")
    print(f"      Download: {row['mean_download_kbps']/1000:.1f} Mbps, Latency: {row['mean_latency_ms']:.1f} ms")

# ===========================================
# 13. CREATE ANOMALY LOG
# ===========================================
print("\n13. Creating Anomaly Log...")

anomaly_log = df_daily[
    (df_daily['iso_anomaly'] == 1) | 
    (df_daily['ocsvm_anomaly'] == 1) | 
    (df_daily['ae_anomaly'] == 1)
].copy()

def get_detection_methods(row):
    methods = []
    if row['iso_anomaly'] == 1:
        methods.append('IsolationForest')
    if row['ocsvm_anomaly'] == 1:
        methods.append('OC-SVM')
    if row['ae_anomaly'] == 1:
        methods.append('Autoencoder')
    return ', '.join(methods)

anomaly_log['detection_methods'] = anomaly_log.apply(get_detection_methods, axis=1)
anomaly_log['methods_count'] = anomaly_log['iso_anomaly'] + anomaly_log['ocsvm_anomaly'] + anomaly_log['ae_anomaly']
anomaly_log = anomaly_log.sort_values(['methods_count', 'aggregate_date'], ascending=[False, True])

# Save anomaly log
anomaly_log.to_csv('outputs/anomaly_log.csv', index=False)
print(f"   Saved {len(anomaly_log)} anomalies to anomaly_log.csv")

# ===========================================
# 14. SAVE PROCESSED DATA
# ===========================================
print("\n14. Saving Processed Data...")

df_daily.to_csv('outputs/processed_data_with_anomalies.csv', index=False)
print(f"   Saved {len(df_daily)} records to processed_data_with_anomalies.csv")

# ===========================================
# 15. CREATE SUMMARY VISUALIZATIONS
# ===========================================
print("\n15. Creating Summary Visualizations...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Method Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Anomaly counts
methods = ['Isolation Forest', 'OC-SVM', 'Autoencoder']
anomaly_cols = ['iso_anomaly', 'ocsvm_anomaly', 'ae_anomaly']
anomaly_counts = [df_daily[col].sum() for col in anomaly_cols]

colors = ['blue', 'orange', 'green']
axes[0, 0].bar(methods, anomaly_counts, color=colors, alpha=0.7)
axes[0, 0].set_ylabel('Number of Anomalies')
axes[0, 0].set_title('Anomalies Detected by Each Method')
for i, v in enumerate(anomaly_counts):
    axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Agreement
agreement_counts = [
    (df_daily['all_methods'] == 3).sum(),
    (df_daily['all_methods'] == 2).sum(),
    (df_daily['all_methods'] == 1).sum()
]
agreement_labels = ['All 3 Methods', 'Exactly 2 Methods', 'Exactly 1 Method']
axes[0, 1].bar(agreement_labels, agreement_counts, color=['red', 'orange', 'gray'], alpha=0.7)
axes[0, 1].set_ylabel('Number of Records')
axes[0, 1].set_title('Anomaly Agreement Between Methods')

# Feature space
scatter = axes[1, 0].scatter(df_daily['mean_download_kbps']/1000, 
                              df_daily['mean_latency_ms'],
                              c=df_daily['iso_anomaly'], cmap='coolwarm', alpha=0.6)
axes[1, 0].set_xlabel('Download Speed (Mbps)')
axes[1, 0].set_ylabel('Latency (ms)')
axes[1, 0].set_title('Anomalies in Feature Space (Isolation Forest)')

# Top 5 worst
labels = [f"{row['region'][:8]}\n{row['carrier_name']}" for _, row in top5_worst.iterrows()]
axes[1, 1].barh(range(5), top5_worst['worst_score'].values, color='red', alpha=0.7)
axes[1, 1].set_yticks(range(5))
axes[1, 1].set_yticklabels(labels)
axes[1, 1].set_xlabel('Worst Score')
axes[1, 1].set_title('Top 5 Worst Performers')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/summary_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print("   Created summary_visualization.png")

# ===========================================
# 16. SAVE METHOD COMPARISON
# ===========================================
comparison_report = pd.DataFrame({
    'Metric': ['Total Anomalies', 'Anomaly Rate (%)', 'True Positives (Agreed by 3)',
               'Potential False Positives (Single Method)'],
    'Isolation Forest': [df_daily['iso_anomaly'].sum(), 
                         f"{df_daily['iso_anomaly'].mean()*100:.2f}",
                         (df_daily['all_methods'] == 3).sum(),
                         ((df_daily['iso_anomaly'] == 1) & (df_daily['all_methods'] == 1)).sum()],
    'One-Class SVM': [df_daily['ocsvm_anomaly'].sum(),
                      f"{df_daily['ocsvm_anomaly'].mean()*100:.2f}",
                      (df_daily['all_methods'] == 3).sum(),
                      ((df_daily['ocsvm_anomaly'] == 1) & (df_daily['all_methods'] == 1)).sum()],
    'Autoencoder (PyTorch)': [df_daily['ae_anomaly'].sum(),
                   f"{df_daily['ae_anomaly'].mean()*100:.2f}",
                   (df_daily['all_methods'] == 3).sum(),
                   ((df_daily['ae_anomaly'] == 1) & (df_daily['all_methods'] == 1)).sum()]
})

comparison_report.to_csv('outputs/method_comparison.csv', index=False)
print("\n   Saved method_comparison.csv")

# ===========================================
# COMPLETION
# ===========================================
print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print(f"\nSummary:")
print(f"  - Total records analyzed: {len(df_daily):,}")
print(f"  - Total anomalies detected: {len(anomaly_log):,}")
print(f"  - High confidence anomalies (3 methods): {(df_daily['all_methods'] == 3).sum()}")
print(f"\nOutputs saved to 'outputs/' directory:")
print(f"  - processed_data_with_anomalies.csv")
print(f"  - anomaly_log.csv")
print(f"  - method_comparison.csv")
print(f"  - summary_visualization.png")
