# Concurrency pilot on its own (PowerShell, from the repository root):
#   powershell -ExecutionPolicy Bypass -File tools\concurrency_test.ps1          # first time
#   powershell -ExecutionPolicy Bypass -File tools\concurrency_test.ps1 -Force   # repeat it
# Runs the same 30-second real-instance run alone, then 4 and then 6 simultaneous
# copies (about 2 minutes). Keep the computer otherwise idle. The result is saved
# in results\01_pilot_concurrency and reused by run_experiments.ps1.

param([switch]$Force)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run tools\setup_windows.ps1 first." }
$arguments = @("tools\campaigns.py", "concurrency")
if ($Force) { $arguments += "--force" }
& $py @arguments
