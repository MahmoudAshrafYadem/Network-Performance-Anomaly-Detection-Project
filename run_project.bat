@echo off
REM ============================================================
REM Network Performance Anomaly Detection Project
REM Master Run Script for Windows
REM ============================================================

setlocal EnableDelayedExpansion

echo ============================================================
echo   Network Performance Anomaly Detection Project
echo ============================================================
echo.

REM Get script directory
cd /d "%~dp0"

REM Step 1: Check Python
echo [Step 1] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.x
    pause
    exit /b 1
)
python --version
echo.

REM Step 2: Install packages
echo [Step 2] Installing required packages...
pip install pandas numpy matplotlib seaborn scikit-learn streamlit jupyter --quiet 2>nul
if errorlevel 1 (
    pip install pandas numpy matplotlib seaborn scikit-learn streamlit jupyter --break-system-packages --quiet 2>nul
)
echo Packages installed.
echo.

REM Step 3: Check data file
echo [Step 3] Checking data file...
if not exist "Performance.csv" (
    echo ERROR: Performance.csv not found!
    pause
    exit /b 1
)
echo Data file found: Performance.csv
echo.

REM Step 4: Run analysis
echo [Step 4] Running analysis...
python run_analysis.py
if errorlevel 1 (
    echo ERROR: Analysis failed!
    pause
    exit /b 1
)
echo Analysis completed.
echo.

REM Step 5: Show summary
echo [Step 5] Analysis Summary:
echo ============================================================
python -c "import pandas as pd; df=pd.read_csv('processed_data_with_anomalies.csv'); a=pd.read_csv('anomaly_log.csv'); print(f'Records: {len(df):,}'); print(f'Anomalies: {len(a):,}'); print(f'High confidence: {(df[\"all_methods\"]==3).sum():,}')"
echo ============================================================
echo.

REM Step 6: Launch dashboard
echo [Step 6] Launching Dashboard...
echo.
echo ============================================================
echo   DASHBOARD STARTING
echo ============================================================
echo.
echo   The dashboard will open in your web browser.
echo   Press Ctrl+C to stop the server.
echo.
echo   If it doesn't open, go to: http://localhost:8501
echo.
echo ============================================================
echo.

streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

echo.
echo Dashboard session ended.
pause
