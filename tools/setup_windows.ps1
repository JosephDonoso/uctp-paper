# One-time setup on Windows (PowerShell, from the repository root):
#   powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
# Creates .venv with Python 3.12.13, installs the solver with Numba and the
# analysis libraries, and runs the test suite.

$ErrorActionPreference = "Stop"
$required = "3.12.13"
$launcher = $null
foreach ($candidate in @(@{ Exe = "py"; Args = @("-3.12") }, @{ Exe = "python"; Args = @() })) {
    try {
        $version = & $candidate.Exe @($candidate.Args + "--version") 2>&1
        if ("$version" -match [regex]::Escape($required)) { $launcher = $candidate; break }
    } catch { }
}
if ($null -eq $launcher) {
    throw "Python $required was not found. Install it from python.org (Windows installer, 64-bit) and run this script again."
}
Write-Host "Using $($launcher.Exe) $($launcher.Args -join ' ') ($required)" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $launcher.Exe @($launcher.Args + @("-m", "venv", ".venv"))
}
$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[fast]" pytest pandas matplotlib scipy
& $py -m pytest -q tests
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
& $py tools\campaigns.py plan
Write-Host "`nSetup complete. Next: powershell -ExecutionPolicy Bypass -File tools\run_experiments.ps1" -ForegroundColor Green
