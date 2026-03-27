# Network Performance Anomaly Detection Project

## PROJECT 05: Daily Signal - Anomaly Detection on Daily Network Performance Data (Egypt Regions)

---

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)

**Linux/Mac - Complete Pipeline:**
```bash
cd anomaly_detection_project
bash run_all.sh
```

**Windows:**
```batch
run_project.bat
```

**Cross-platform (Python):**
```bash
python run_project.py
```

### Option 2: Run with Options

**Linux/Mac - Selective Run:**
```bash
# Show available options
bash run.sh --help

# Install dependencies only
bash run.sh --setup

# Run analysis only (no dashboard)
bash run.sh --analyze

# Launch dashboard only
bash run.sh --dashboard

# Run everything (same as run_all.sh)
bash run.sh --all
```

---

## 📁 Project Structure

```
anomaly_detection_project/
├── run_all.sh              # ★ COMPLETE RUN SCRIPT (Linux/Mac) - Runs everything
├── run.sh                  # ★ SELECTIVE RUN SCRIPT (Linux/Mac) - With options
├── run_project.bat         # Master run script (Windows)
├── run_project.py          # Master run script (Cross-platform Python)
├── run_analysis.py         # Analysis script
├── app.py                  # Streamlit dashboard
├── Anomaly_Detection_Network_Performance.ipynb  # Jupyter notebook (PyTorch)
├── anomaly_detection_analysis.ipynb            # Detailed Jupyter notebook
├── Performance.csv         # Dataset
├── processed_data_with_anomalies.csv           # Analysis results
├── anomaly_log.csv         # Anomaly log
├── method_comparison.csv   # Method comparison
├── summary_visualization.png                    # Summary plots
├── requirements.txt        # Dependencies
├── data/                   # Data directory
├── outputs/                # Output files directory
├── logs/                   # Execution logs
└── README.md               # This file
```

---

## 🔧 Available Commands

### Run Complete Project (bash run_all.sh)

The `run_all.sh` script performs ALL steps automatically:

1. ✓ Creates project directories (data/, outputs/, logs/)
2. ✓ Checks Python installation
3. ✓ Installs ALL dependencies from requirements.txt
4. ✓ Verifies data file (Performance.csv)
5. ✓ Runs complete analysis pipeline
6. ✓ Executes Jupyter notebook
7. ✓ Shows summary of outputs
8. ✓ Launches Streamlit dashboard

### Run Selective Steps (bash run.sh)

```bash
# Show all options
bash run.sh --help

# Available options:
bash run.sh --all        # Run everything and launch dashboard (default)
bash run.sh --setup      # Install dependencies only
bash run.sh --analyze    # Run analysis pipeline only
bash run.sh --notebook   # Execute Jupyter notebook only
bash run.sh --dashboard  # Launch Streamlit dashboard only
```

### Run Individual Components

```bash
# Run analysis only
python run_analysis.py

# Launch dashboard only
streamlit run app.py

# Open Jupyter notebook
jupyter notebook Anomaly_Detection_Network_Performance.ipynb
```

---

## 📦 Dependencies

The script will automatically install these packages:

```
pandas
numpy
scikit-learn
torch           # PyTorch for Autoencoder
streamlit
plotly
matplotlib
seaborn
matplotlib-venn
kaleido
jupyter
nbconvert
ipykernel
```

---

## 🤖 Anomaly Detection Methods

| Method | Description | Speed |
|--------|-------------|-------|
| **Isolation Forest** | Tree-based ensemble that isolates anomalies | Fast |
| **One-Class SVM** | Learns decision boundary around normal data | Slow |
| **Autoencoder (PyTorch)** | Neural network reconstruction error | Moderate |

**Consensus Approach**: Anomalies detected by 2+ methods are considered high-confidence.

### PyTorch Autoencoder Architecture
```
Encoder: input_dim -> 16 -> 8 -> 2 (bottleneck)
Decoder: 2 -> 8 -> 16 -> input_dim
Activation: ReLU (hidden), Sigmoid (output)
Loss: MSELoss
Optimizer: Adam (lr=0.001)
```

---

## 📊 Project Questions Answered

### Q1: Most Variable Download Speed Regions
Regions ranked by Coefficient of Variation (CV) in download speed.

### Q2: Days with Abnormal Cross-Region Performance
Daily analysis of anomaly counts across all regions.

### Q3: Carrier Performance Consistency
Heatmap of carrier performance across regions.

### Q4: LTE vs 5G Comparison
Comparison of anomaly rates between technologies.

### Q5: Top 5 Worst Performing Combinations
Ranked by worst score (download, upload, latency, anomaly flags).

---

## 📋 Filtering Decisions

| Filter | Value | Justification |
|--------|-------|---------------|
| Sample Count | ≥ 5 | Removes unreliable single measurements |
| Place Type | locality | Focus on city-level data |
| Aggregation Period | Day | Daily granularity analysis |

---

## 📈 Results Summary

```
Total records analyzed:     8,086
Total anomalies detected:   ~742
High confidence anomalies:  ~118
Anomaly rate:              ~5%
```

---

## 🌐 Dashboard Features

The Streamlit dashboard includes:
- 📊 **Overview Tab**: Dataset statistics and distributions
- 🔍 **Anomaly Detection Tab**: Results from all 3 methods
- 📊 **Q&A Results Tab**: Answers to all 5 questions
- 🏆 **Top 5 Worst Tab**: Worst performing combinations
- 📋 **Anomaly Log Tab**: Searchable/filterable anomaly list
- ℹ️ **Method Comparison Tab**: Method agreement analysis

**Dashboard Filters:**
- Date range selection
- Region filter
- Carrier filter
- Technology filter (LTE/5G)

---

## ⚠️ Troubleshooting

### Data File Not Found
```
Error: Performance.csv not found!
```
**Solution**: Place `Performance.csv` in one of these locations:
- `anomaly_detection_project/data/Performance.csv`
- `anomaly_detection_project/Performance.csv`
- Parent directory of the project

### PyTorch Not Available
The script will automatically use sklearn's MLPRegressor instead.

### Permission Denied (Linux/Mac)
```bash
# Run with bash explicitly
bash run_all.sh
```

### CUDA/GPU Support
PyTorch will automatically use GPU if available, otherwise CPU.

---

## 👤 Author

Data Science Engineer - March 2026

---

## 📄 License

ITI - Internal Training Material
