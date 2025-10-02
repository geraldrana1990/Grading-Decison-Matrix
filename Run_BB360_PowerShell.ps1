# === BB360 one-click launcher (PowerShell) ===
# Right-click this file > Run with PowerShell.
# It will create a virtual env, install deps, and start the app in your browser.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Detect Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $python) { Write-Host "Python not found. Please install Python 3.10+."; exit 1 }

# Create venv if missing
if (-not (Test-Path ".\.venv")) {
  & $python.Source -m venv .venv
}

# Activate venv
$activate = ".\.venv\Scripts\Activate.ps1"
. $activate

# Install requirements
python -m pip install --upgrade pip | Out-Null
pip install -r requirements.txt

# Launch Streamlit
Start-Process "http://localhost:8501/"
streamlit run BB360_app.py --server.port=8501 --server.address=0.0.0.0
