# setup.ps1 - run once from the project root.
# Usage:  cd C:\dev\credit-risk-copilot ;  .\setup.ps1

Write-Host "Creating folder tree..." -ForegroundColor Cyan
$dirs = @(
    "data\raw", "data\processed",
    "ecl", "agent", "evals",
    "notebooks", "tests", "docs"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "Installing packages..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Next:" -ForegroundColor Green
Write-Host "  1. Put kaggle.json at C:\Users\rudra\.kaggle\kaggle.json"
Write-Host "  2. cd data\raw"
Write-Host "     kaggle datasets download -d wordsforthewise/lending-club"
Write-Host "     Expand-Archive .\lending-club.zip -DestinationPath ."
Write-Host "  3. cd ..\.. ; python notebooks\00_schema_audit.py"
