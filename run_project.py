#!/usr/bin/env python3
"""
Network Performance Anomaly Detection Project
Master Run Script (Python Version)

This script runs the complete project:
1. Check/Install dependencies
2. Run the analysis (data processing + anomaly detection)
3. Launch the Streamlit dashboard

Usage:
    python run_project.py [--skip-analysis] [--skip-dashboard] [--install-only]

Options:
    --skip-analysis   Skip running analysis (use existing results)
    --skip-dashboard  Skip launching dashboard
    --install-only    Only install dependencies, don't run anything
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_section(text):
    """Print a section header."""
    print(f"\n>>> {text}\n")

def run_command(cmd, description="Running command"):
    """Run a shell command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        return False

def check_python():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ is required!")
        return False
    return True

def install_packages():
    """Install required packages."""
    print_section("Installing Required Packages")
    
    packages = [
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "torch",
        "streamlit",
        "jupyter",
    ]
    
    for package in packages:
        print(f"  - Installing {package}...")
        
        # Try different pip install methods
        commands = [
            f'"{sys.executable}" -m pip install {package} --quiet',
            f'pip install {package} --quiet --break-system-packages',
            f'pip install {package} --quiet',
        ]
        
        installed = False
        for cmd in commands:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                installed = True
                break
        
        if not installed:
            print(f"    Warning: Could not install {package}, trying to continue...")
        else:
            print(f"    ✓ {package} installed")
    
    print("\nPackage installation completed!")

def check_data_file():
    """Check if data file exists."""
    data_file = SCRIPT_DIR / "Performance.csv"
    
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return False
    
    file_size = data_file.stat().st_size / (1024 * 1024)  # MB
    print(f"Data file found: {data_file.name}")
    print(f"File size: {file_size:.2f} MB")
    return True

def run_analysis():
    """Run the main analysis script."""
    print_section("Running Anomaly Detection Analysis")
    
    analysis_script = SCRIPT_DIR / "run_analysis.py"
    
    if not analysis_script.exists():
        print(f"ERROR: Analysis script not found: {analysis_script}")
        return False
    
    print("Running analysis (this may take a few minutes)...\n")
    
    # Run the analysis
    result = subprocess.run(
        f'"{sys.executable}" "{analysis_script}"',
        shell=True,
        text=True
    )
    
    if result.returncode != 0:
        print("ERROR: Analysis failed!")
        return False
    
    # Check outputs
    outputs = [
        "processed_data_with_anomalies.csv",
        "anomaly_log.csv",
    ]
    
    for output in outputs:
        if not (SCRIPT_DIR / output).exists():
            print(f"ERROR: Expected output not found: {output}")
            return False
    
    print("\n✓ Analysis completed successfully!")
    
    # Print summary
    print_summary()
    
    return True

def print_summary():
    """Print analysis summary."""
    print_section("Analysis Summary")
    
    try:
        import pandas as pd
        
        df = pd.read_csv(SCRIPT_DIR / "processed_data_with_anomalies.csv")
        anomaly_log = pd.read_csv(SCRIPT_DIR / "anomaly_log.csv")
        
        print("="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total records analyzed: {len(df):,}")
        print(f"Total anomalies detected: {len(anomaly_log):,}")
        print(f"High confidence (3 methods agree): {(df['all_methods'] == 3).sum():,}")
        print("")
        print("ANOMALIES BY METHOD:")
        print(f"  - Isolation Forest: {df['iso_anomaly'].sum():,}")
        print(f"  - One-Class SVM: {df['ocsvm_anomaly'].sum():,}")
        print(f"  - Autoencoder (PyTorch): {df['ae_anomaly'].sum():,}")
        print("")
        print("TOP 5 MOST VARIABLE REGIONS:")
        
        regional = df.groupby('region').agg({'mean_download_kbps': ['mean', 'std']})
        regional.columns = ['mean', 'std']
        regional['cv'] = regional['std'] / regional['mean'] * 100
        regional = regional.sort_values('cv', ascending=False)
        
        for i, (region, row) in enumerate(regional.head(5).iterrows(), 1):
            print(f"  {i}. {region}: CV = {row['cv']:.1f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"Could not generate summary: {e}")

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print_section("Launching Streamlit Dashboard")
    
    dashboard_file = SCRIPT_DIR / "app.py"
    
    if not dashboard_file.exists():
        print(f"ERROR: Dashboard file not found: {dashboard_file}")
        return False
    
    print("="*60)
    print("  DASHBOARD STARTING")
    print("="*60)
    print("")
    print("  The dashboard will open in your web browser.")
    print("  Press Ctrl+C to stop the server.")
    print("")
    print("  If it doesn't open automatically, go to:")
    print("  http://localhost:8501")
    print("")
    print("="*60)
    print("")
    
    # Launch Streamlit
    subprocess.run(
        f'streamlit run "{dashboard_file}" --server.headless=true --browser.gatherUsageStats=false',
        shell=True
    )
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Network Performance Anomaly Detection Project Runner"
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip running analysis (use existing results)'
    )
    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip launching dashboard'
    )
    parser.add_argument(
        '--install-only',
        action='store_true',
        help='Only install dependencies'
    )
    
    args = parser.parse_args()
    
    print_header("Network Performance Anomaly Detection Project")
    
    # Check Python version
    if not check_python():
        sys.exit(1)
    
    # Install packages
    install_packages()
    
    if args.install_only:
        print("\n✓ Dependencies installed. Run without --install-only to execute the project.")
        sys.exit(0)
    
    # Check data file
    if not check_data_file():
        sys.exit(1)
    
    # Run analysis
    if not args.skip_analysis:
        if not run_analysis():
            sys.exit(1)
    else:
        print("\nSkipping analysis (using existing results)")
        print_summary()
    
    # Launch dashboard
    if not args.skip_dashboard:
        launch_dashboard()
    
    print("\n✓ Project completed!")

if __name__ == "__main__":
    main()
