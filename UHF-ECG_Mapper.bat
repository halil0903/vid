@echo off
title UHF-ECG Activation Mapper
echo ============================================================
echo   UHF-ECG Ventricular Activation Mapper
echo   Research ^& Education Tool
echo ============================================================
echo.
echo   Starting application...
echo   Browser will open automatically at http://localhost:8501
echo.
echo   Press Ctrl+C to stop.
echo ============================================================

cd /d "%~dp0"
start "" http://localhost:8501
streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false --theme.base dark --theme.primaryColor "#FF6B6B" --theme.backgroundColor "#0a0a2e" --theme.secondaryBackgroundColor "#12122e" --theme.textColor "#e0e0ff"
pause
