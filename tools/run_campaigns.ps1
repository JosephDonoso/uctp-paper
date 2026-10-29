# Re-run every time-budget campaign of the manuscript and regenerate its tables and figures.
#
# Usage (from the repository root, in PowerShell):
#   .\tools\run_campaigns.ps1 -Synthetic "C:\ruta\a\datasets_sinteticos"
#
# -Synthetic must contain one folder per group (Dispersion-Low, BASE, Dispersion-High,
# Offer-Low, Offer-High) with the SAME synthetic instances used before (legacy format).
# Every campaign can be stopped and resumed: run the script again and finished runs are
# skipped (--resume). Results go to .\results\<campaign>.

param(
    [Parameter(Mandatory = $true)][string]$Synthetic,
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$Out = "results",
    [int]$RealSeeds = 50,
    [switch]$SkipSynthetic
)

$ErrorActionPreference = "Stop"
$Alphas = @("0", "0.125", "0.25", "0.5", "1", "2", "4")
$Groups = @("Dispersion-Low", "BASE", "Dispersion-High", "Offer-Low", "Offer-High")
$Real = "datasets\Real\real.json"

function Run([string[]]$Arguments) {
    Write-Host "`n>>> $Python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE)." }
}

# 0. Environment check: tests (includes exact Numba vs pure-Python equivalence).
Run @("-m", "pytest", "-q", "tests")
Run @("-c", "import numba, numpy, sys; print('Python', sys.version.split()[0], 'NumPy', numpy.__version__, 'Numba', numba.__version__)")

# 1. Real instance, 30 s, all weights (main real-instance results).
Run (@("-m", "uctp.experiments", "run", "--inputs", $Real, "--output", "$Out\real_30s",
       "--approaches", "Embedded", "Decoupled-50", "--alphas") + $Alphas +
     @("--runs", "$RealSeeds", "--time-budgets", "30", "--save-solutions", "--resume"))

# 2. Real instance, quality-time curve (30, 120, 300 s) with traces.
Run @("-m", "uctp.experiments", "run", "--inputs", $Real, "--output", "$Out\real_curve",
      "--approaches", "Embedded", "Decoupled-50", "--alphas", "0", "0.25", "1",
      "--runs", "10", "--time-budgets", "30", "120", "300", "--save-solutions", "--trace", "--resume")

# 3. Local-search ablation (real instance and BASE).
Run @("-m", "uctp.experiments", "run", "--inputs", $Real, "--output", "$Out\ls_ablation_real",
      "--approaches", "Embedded", "--alphas", "0.25", "1", "--local-search", "on", "off",
      "--runs", "$RealSeeds", "--time-budgets", "30", "--save-solutions", "--resume")

if (-not $SkipSynthetic) {
    $SynInputs = $Groups | ForEach-Object { Join-Path $Synthetic $_ }
    foreach ($path in $SynInputs) { if (-not (Test-Path $path)) { throw "Missing synthetic group: $path" } }

    # 4. Synthetic approach comparison: Embedded and Decoupled-25/50/100, 7 weights, 30 s.
    Run (@("-m", "uctp.experiments", "run", "--inputs") + $SynInputs +
         @("--input-format", "legacy", "--output", "$Out\synthetic_30s",
           "--approaches", "Embedded", "Decoupled-25", "Decoupled-50", "Decoupled-100",
           "--alphas") + $Alphas + @("--seeds", "703", "--time-budgets", "30", "--save-solutions", "--resume"))

    # 5. LS ablation on BASE.
    Run @("-m", "uctp.experiments", "run", "--inputs", (Join-Path $Synthetic "BASE"),
          "--input-format", "legacy", "--output", "$Out\ls_ablation_base",
          "--approaches", "Embedded", "--alphas", "0.25", "1", "--local-search", "on", "off",
          "--seeds", "703", "--time-budgets", "30", "--save-solutions", "--resume")

    # 6. Five-seed variability on BASE.
    Run @("-m", "uctp.experiments", "run", "--inputs", (Join-Path $Synthetic "BASE"),
          "--input-format", "legacy", "--output", "$Out\base_five_seeds",
          "--approaches", "Embedded", "--alphas", "1",
          "--seeds", "703", "215303898", "1366270267", "664565471", "2054631744",
          "--time-budgets", "30", "--save-solutions", "--resume")
}

# 7. Tables and figures (requires pandas, matplotlib, scipy).
$Art = "$Out\paper_artifacts"
Run @("tools\paper_artifacts.py", "real-weights", "--raw", "$Out\real_30s\raw.csv", "--budget", "30",
      "--baselines", "tools\real_baselines.json", "--alpha-compare", "0.25", "--out", $Art)
Run @("tools\paper_artifacts.py", "quality-time", "--raw", "$Out\real_curve\raw.csv", "--out", $Art)
if (-not $SkipSynthetic) {
    Run @("tools\paper_artifacts.py", "synthetic-changes", "--raw", "$Out\synthetic_30s\raw.csv",
          "--budget", "30", "--approaches", "Embedded", "Decoupled-50", "--out", $Art)
}
Write-Host "`nDone. Tables and figures in $Art" -ForegroundColor Green
