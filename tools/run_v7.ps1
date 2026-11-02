# Experiments of the v7 revision (PowerShell, from the repository root):
#   powershell -ExecutionPolicy Bypass -File tools\run_v7.ps1 -Round 1
#   powershell -ExecutionPolicy Bypass -File tools\run_v7.ps1 -Round 2   (after decision.json exists)
# Each call is a session of at most -MaxHours (default 8) and continues where the
# previous one stopped; repeat it until it reports that the round is complete.
# Stop early: create an empty file STOP inside results_v7 (running shards finish),
# or press Ctrl+C (the runs in progress are repeated next time).

param(
    [Parameter(Mandatory = $true)][ValidateSet(1, 2)][int]$Round,
    [int]$Workers = 4,
    [double]$MaxHours = 8
)

$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run tools\setup_windows.ps1 first." }
& $py -c "import openpyxl" 2>$null
if ($LASTEXITCODE -ne 0) { throw "openpyxl is missing: run tools\setup_windows.ps1 again first." }
& $py tools\campaigns_v7.py run --round $Round --workers $Workers --max-hours $MaxHours
$code = $LASTEXITCODE
& $py tools\campaigns_v7.py status --round $Round
switch ($code) {
    0 { Write-Host "`nRound $Round complete. Next: $py tools\campaigns_v7.py package" -ForegroundColor Green }
    2 { Write-Host "`nSession ended during the Decoupled pilot. Run this script again." -ForegroundColor Yellow }
    3 { Write-Host "`nSession finished. Run this script again to continue." -ForegroundColor Yellow }
    default { Write-Host "`nStopped (exit $code). Run this script again to continue." -ForegroundColor Yellow }
}
