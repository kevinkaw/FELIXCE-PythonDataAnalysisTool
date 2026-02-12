@echo off
:: Change directory to the parent folder (project root)
cd /d "%~dp0\.."

:: Activate the environment
call conda activate FELIXCE_v2026.02.12

:: Run the app from the root folder
streamlit run main.py

pause