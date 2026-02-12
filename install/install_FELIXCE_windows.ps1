$EnvFile = "environment.yml"
$MinicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$DefaultInstall = "$env:USERPROFILE\miniconda3"

Write-Host "--- Checking for Conda ---" -ForegroundColor Cyan

# 1. Check if 'conda' command works
$CondaExe = Get-Command conda -ErrorAction SilentlyContinue

if ($CondaExe) {
    Write-Host "Conda found in PATH."
} else {
    # 2. Check default paths if not in PATH
    $PossiblePaths = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )

    foreach ($Path in $PossiblePaths) {
        if (Test-Path $Path) {
            $CondaExe = $Path
            Write-Host "Conda found at $Path"
            break
        }
    }
}

# 3. Install if still not found
if (!$CondaExe) {
    Write-Host "Conda not found. Downloading Miniconda..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $MinicondaUrl -OutFile "miniconda.exe"
    Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S", "/InstallationType=JustMe", "/D=$DefaultInstall" -Wait
    Remove-Item ".\miniconda.exe"
    $CondaExe = "$DefaultInstall\Scripts\conda.exe"
}

# 4. Initialize and Create Environment
# Get a list of environments
$ExistingEnvs = & $CondaExe env list

if ($ExistingEnvs -match "FELIXCE_v2026.02.12") {
    Write-Host "Environment exists. Updating and pruning..." -ForegroundColor Yellow
    & $CondaExe env update -f $EnvFile --prune
} else {
    Write-Host "Creating new environment..." -ForegroundColor Yellow
    & $CondaExe env create -f $EnvFile
}

Write-Host "--- Setup Complete ---" -ForegroundColor Green