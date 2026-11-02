# One-time setup on Windows (PowerShell, from the repository root):
#   powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
#
# python.org publishes no Windows installer for Python 3.12.13 (3.12 receives
# source-only security releases after 3.12.10). This script therefore uses uv
# (https://docs.astral.sh/uv/), which installs a standalone CPython 3.12.13 for
# this project without touching any other Python on the computer. It then
# creates .venv, installs the solver with Numba plus pytest, pandas, matplotlib
# and SciPy, runs the tests and prints the experiment plan.

$ErrorActionPreference = "Stop"
$required = "3.12.13"

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($path in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $path) { return $path }
    }
    return $null
}

$uv = Find-Uv
if ($null -eq $uv) {
    Write-Host "Installing uv (per-user, no administrator rights needed)..." -ForegroundColor Cyan
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = Find-Uv
    if ($null -eq $uv) { throw "uv could not be installed. Install it manually (https://docs.astral.sh/uv/) and run this script again." }
}
Write-Host "Using uv: $uv ($(& $uv --version))" -ForegroundColor Cyan

# Older uv versions do not know recent Python patch releases: update uv first.
& $uv self update
if ($LASTEXITCODE -ne 0) {
    Write-Host "uv self update is not available for this uv installation; reinstalling uv..." -ForegroundColor Yellow
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = "$env:USERPROFILE\.local\bin\uv.exe"
    if (-not (Test-Path $uv)) { throw "uv could not be updated. Update it manually and run this script again." }
}
Write-Host "uv version: $(& $uv --version)" -ForegroundColor Cyan

& $uv python install $required
if ($LASTEXITCODE -ne 0) { throw "uv could not install Python $required." }
if (Test-Path ".venv\Scripts\python.exe") {
    $current = & ".venv\Scripts\python.exe" --version
    if ("$current" -notmatch [regex]::Escape($required)) {
        throw ".venv uses $current. Delete the .venv folder and run this script again."
    }
} else {
    & $uv venv --python $required .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv." }
}
$py = ".venv\Scripts\python.exe"
& $uv pip install --python $py -e ".[fast]" pytest pandas matplotlib scipy openpyxl
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
& $py -c "import sys, numpy, numba; print('Python', sys.version.split()[0], 'NumPy', numpy.__version__, 'Numba', numba.__version__)"
& $py -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
& $py tools\campaigns.py plan
Write-Host "`nSetup complete. Next: powershell -ExecutionPolicy Bypass -File tools\run_experiments.ps1" -ForegroundColor Green
