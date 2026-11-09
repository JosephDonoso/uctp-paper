# Pending runs of the v9 revision (run from the repository root, PowerShell).
#
#   .\tools\run_v9_pending.ps1 -Base "C:\ruta\datasets\Synthetic\BASE" -ResultsV7 "results_v7" -Seeds "results_v7\SEEDS.json"
#
# 1. LS ablation under the main protocol (2500 iterations, common random numbers, checkpoints,
#    50 re-evaluation orders), on the real instance (50 seeds) and on BASE (seed 703).
#    About 9 h of CPU in total; resumable (--resume).
# 2. Crossed evaluation of the new ablation timetables (with / without LS).
# 3. Partial adjustment (p = 0.6) on the final alpha-0.25 timetables of E5a (2500 iterations).
#
# Outputs: results_v9\LS_v9 (runs), extract\ (compact solutions), reeval_v9\ (CSV for the analysis).

param(
    [Parameter(Mandatory = $true)][string]$Base,
    [Parameter(Mandatory = $true)][string]$ResultsV7,
    [Parameter(Mandatory = $true)][string]$Seeds,
    [string]$Python = ".venv\Scripts\python.exe",
    [int]$Workers = 4
)
$ErrorActionPreference = "Stop"
function Run([string[]]$Arguments) {
    Write-Host "`n>>> $Python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE)." }
}
$Common = @("--approaches", "Embedded", "--alphas", "0.25", "1", "--local-search", "on", "off",
            "--iteration-budgets", "2500", "--checkpoints", "25", "50", "100", "200", "500", "1000", "1500", "2000",
            "--common-random-numbers", "--save-solutions", "--reevaluate", "50", "--resume")

# 1. Runs (layout results_v9\LS_v9\shards\<name>\records, as expected by extract_solutions.py)
Run (@("-m", "uctp.experiments", "run", "--inputs", "datasets\Real\real.json", "--input-format", "instance",
       "--output", "results_v9\LS_v9\shards\real", "--runs", "50") + $Common)
Run (@("-m", "uctp.experiments", "run", "--inputs", $Base, "--input-format", "legacy", "--group", "BASE",
       "--output", "results_v9\LS_v9\shards\BASE", "--seeds", "703") + $Common)

# 2. Crossed evaluation
Run @("tools\extract_solutions.py", "results_v9", "LS_v9", "extract\LS_v9.jsonl.gz")
Run @("tools\reevaluate_solutions.py", "--solutions", "extract\LS_v9.jsonl.gz", "--orders", "50",
      "--ls", "on", "--out", "reeval_v9\reeval_ls_on.csv", "--workers", "$Workers")
Run @("tools\reevaluate_solutions.py", "--solutions", "extract\LS_v9.jsonl.gz", "--orders", "50",
      "--ls", "off", "--out", "reeval_v9\reeval_ls_off.csv", "--workers", "$Workers")

# 3. Partial adjustment on the 2500-iteration timetables
Run @("tools\extract_solutions.py", $ResultsV7, "E5a_real_converged", "extract\E5a.jsonl.gz")
Run @("tools\v7_noise_ls.py", "--solutions", "extract\E5a.jsonl.gz", "--seeds", $Seeds,
      "--out", "reeval_v9", "--workers", "$Workers", "--partial-stage", "E5a_real_converged")

Write-Host "`nDone. Copy reeval_v9\reeval_ls_on.csv, reeval_ls_off.csv, e4b_calibration.csv and e4b_partial.csv" -ForegroundColor Green
Write-Host "into paper_results_v8\data (or a v9 copy) and rerun paper_analysis_v8.py and revision_checks.py." -ForegroundColor Green
