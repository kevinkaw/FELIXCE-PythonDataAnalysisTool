@echo off
setlocal EnableDelayedExpansion

:: --- Configuration ---
set "ENV_FILE=environment.yml"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "DEFAULT_INSTALL=%USERPROFILE%\miniconda3"
set "ENV_NAME=FELIXCE_v2026.02.12"

echo --- Checking for Conda ---

:: 1. Check if 'conda' command is in PATH
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo Conda found in PATH.
    :: Get the full path to ensuring consistent execution
    for /f "tokens=*" %%i in ('where conda') do set "CONDA_EXE=%%i"
    goto :CondaFound
)

:: 2. Check default paths if not in PATH
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
    goto :CondaFound
)
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
    goto :CondaFound
)
if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" (
    set "CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe"
    goto :CondaFound
)

:: 3. Install if still not found
echo Conda not found. Downloading Miniconda...
curl -o miniconda.exe "%MINICONDA_URL%"
if %errorlevel% neq 0 (
    echo Error: Failed to download Miniconda.
    pause
    exit /b 1
)

echo Installing Miniconda...
start /wait "" miniconda.exe /S /InstallationType=JustMe /D=%DEFAULT_INSTALL%
del miniconda.exe

set "CONDA_EXE=%DEFAULT_INSTALL%\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
    echo Error: Conda installation seemed to fail. Could not find conda.exe.
    pause
    exit /b 1
)

:CondaFound
echo.
echo Using Conda at: "%CONDA_EXE%"

:: 4. Initialize and Create/Update Environment
echo Checking existing environments...

:: We need to use 'call' because conda might be a batch file.
:: Also quoting the executable path is safer.
call "%CONDA_EXE%" env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% equ 0 (
    echo Environment '%ENV_NAME%' exists. Updating and pruning...
    call "%CONDA_EXE%" env update -f "%ENV_FILE%" --prune
) else (
    echo Creating new environment '%ENV_NAME%'...
    call "%CONDA_EXE%" env create -f "%ENV_FILE%"
)

if %errorlevel% neq 0 (
    echo.
    echo Error: Environment setup failed.
    pause
    exit /b %errorlevel%
)

echo.
echo --- Setup Complete ---
pause