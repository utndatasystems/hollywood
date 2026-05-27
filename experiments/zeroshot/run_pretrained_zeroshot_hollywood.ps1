param(
  [string]$Python = "python",
  [Parameter(Mandatory=$true)][string]$LcmSrc,
  [Parameter(Mandatory=$true)][string]$InputRoot,
  [Parameter(Mandatory=$true)][string]$ModelDir,
  [string]$OutRoot = "runs/zeroshot_hollywood_200k",
  [int[]]$Seeds = @(0, 1, 2),
  [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$env:PYTHON_DOTENV_DISABLED = "1"
foreach ($node in @("NODE00", "NODE01", "NODE02", "NODE03", "NODE04", "NODE05")) {
  Set-Item -Path "env:$node" -Value "{'hostname':'local','python':'3.11'}"
}

$stats = Join-Path $InputRoot "statistics_workload_combined_hollywood.json"
$jobLight = Join-Path $InputRoot "parsed_plans\hollywood_job_light.json"
$job = Join-Path $InputRoot "parsed_plans\hollywood_job.json"
$jobComplex = Join-Path $InputRoot "parsed_plans\hollywood_job_complex.json"
$hyperparams = Join-Path $LcmSrc "conf\zeroshot_hyperparameters\tune_est_best_config.json"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

foreach ($seed in $Seeds) {
  $targetDir = Join-Path $OutRoot "seed_$seed"
  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

  Push-Location $LcmSrc
  try {
    & $Python main.py `
      --mode predict `
      --model_type zeroshot `
      --test_workload_runs $jobLight $job $jobComplex `
      --statistics_file $stats `
      --model_dir $ModelDir `
      --target_dir $targetDir `
      --device $Device `
      --hyperparameter_path $hyperparams `
      --seed $seed `
      --num_workers 0
  }
  finally {
    Pop-Location
  }
}
