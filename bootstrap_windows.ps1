param(
    [switch]$SkipPythonInstall,
    [switch]$NoVenv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Get-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @("py", "-3.12")
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }

    return $null
}

function Invoke-PythonCommand {
    param(
        [string[]]$PythonCmd,
        [string[]]$Args
    )

    if ($PythonCmd.Length -gt 1) {
        $launcher = $PythonCmd[0]
        $prefix = $PythonCmd[1..($PythonCmd.Length - 1)]
        & $launcher @prefix @Args
    }
    else {
        & $PythonCmd[0] @Args
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Step "IR Project bootstrap started"
Write-Host "Project root: $projectRoot"

$pythonCmd = Get-PythonCommand

if (-not $pythonCmd) {
    if ($SkipPythonInstall) {
        throw "Python not found and -SkipPythonInstall was provided."
    }

    Write-Step "Python not found. Attempting install via winget"
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget is not available. Install Python 3.12 manually, then re-run this script."
    }

    winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements

    # Refresh local PATH candidates for current session.
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312",
        "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            if (-not ($env:PATH -split ";" | Where-Object { $_ -eq $candidate })) {
                $env:PATH = "$candidate;$env:PATH"
            }
        }
    }

    $pythonCmd = Get-PythonCommand
    if (-not $pythonCmd) {
        throw "Python install appears incomplete. Open a new terminal and run this script again."
    }
}

Write-Step "Using Python launcher"
Write-Host ($pythonCmd -join " ")

if (-not $NoVenv) {
    Write-Step "Creating virtual environment (.venv)"
    Invoke-PythonCommand -PythonCmd $pythonCmd -Args @("-m", "venv", ".venv")

    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment creation failed (.venv\\Scripts\\python.exe not found)."
    }

    Write-Step "Upgrading pip in virtual environment"
    & $venvPython -m pip install --upgrade pip

    Write-Step "Installing project dependencies"
    & $venvPython -m pip install -r requirements.txt

    Write-Step "Bootstrap complete"
    Write-Host "Activate venv: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
    Write-Host "Run GUI: python -m streamlit run src/product/app.py" -ForegroundColor Green
    Write-Host "Run pipeline from CLI:" -ForegroundColor Green
    Write-Host "  python src/scraper/scrape_complaints.py --target-records 80 --max-sitemaps 2 --crawl-delay 0.5"
    Write-Host "  python src/pipeline/preprocess.py"
    Write-Host "  python src/analysis/ai_issue_classifier.py"
    Write-Host "  python src/analysis/eda.py"
}
else {
    Write-Step "No virtual environment requested (-NoVenv)"
    Invoke-PythonCommand -PythonCmd $pythonCmd -Args @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-PythonCommand -PythonCmd $pythonCmd -Args @("-m", "pip", "install", "-r", "requirements.txt")

    Write-Step "Bootstrap complete"
    Write-Host "Run GUI: python -m streamlit run src/product/app.py" -ForegroundColor Green
}
