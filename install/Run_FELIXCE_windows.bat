@echo off
setlocal enabledelayedexpansion

:: 1. Define your environment name
set "TARGET_ENV=FELIXCE_v2026.02.12"

:: 2. Dynamically find the path for this environment
for /f "tokens=2" %%i in ('conda env list ^| findstr /C:"%TARGET_ENV%"') do (
    set "ENV_PATH=%%i"
)

:: Check if the environment was actually found
if not defined ENV_PATH (
    echo [ERROR] Could not find conda environment: %TARGET_ENV%
    pause
    exit /b
)

:: 3. Set the PATH to prioritize your Conda DLLs (fixes the MSMPI warning)
set "PATH=%ENV_PATH%;%ENV_PATH%\Library\bin;%ENV_PATH%\Scripts;%PATH%"

:: 4. Launch Streamlit directly using the environment's Python
:: Uses -m streamlit to bypass the slow activation handshake
cd /d "%~dp0\.."
"%ENV_PATH%\python.exe" -m streamlit run main.py

pause