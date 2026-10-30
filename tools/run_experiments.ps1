# Run or continue all experiments (PowerShell, from the repository root):
#   powershell -ExecutionPolicy Bypass -File tools\run_experiments.ps1
# Each call is a session of at most -MaxHours (default 8) and continues where the
# previous session stopped. Repeat the same command until it reports that all
# campaigns are complete. To stop early and cleanly, create an empty file named
# STOP inside the results folder (running shards finish first), or press Ctrl+C
# (the runs in progress are repeated next time).

param(
    [int]$Workers = 0,
    [double]$MaxHours = 8,
    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run tools\setup_windows.ps1 first." }
$arguments = @("tools\campaigns.py", "run", "--max-hours", "$MaxHours")
if ($Workers -gt 0) { $arguments += @("--workers", "$Workers") }
if ($Only.Count -gt 0) { $arguments += @("--only") + $Only }
& $py @arguments
$code = $LASTEXITCODE
& $py tools\campaigns.py status
switch ($code) {
    0 { Write-Host "`nAll campaigns complete. Next: $py tools\campaigns.py package" -ForegroundColor Green }
    2 { Write-Host "`nSession ended during the convergence pilot. Run this script again." -ForegroundColor Yellow }
    3 { Write-Host "`nSession finished. Run this script again to continue." -ForegroundColor Yellow }
    default { Write-Host "`nStopped (exit $code). Run this script again to continue." -ForegroundColor Yellow }
}
