param(
  [int]$Movies = 200000,
  [int]$UntilStep = 130,
  [string]$Model = "gemini-3.1-flash-lite",
  [switch]$RunCalibration
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$python = $env:PYTHON_EXE
if ([string]::IsNullOrWhiteSpace($python)) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) {
    $python = $cmd.Source
  }
}
if ([string]::IsNullOrWhiteSpace($python)) {
  throw "Python executable not found. Set PYTHON_EXE or add python to PATH."
}
$script = Join-Path $PSScriptRoot 'run_full_api_pipeline.py'

$cmd = @(
  $script,
  '--n-movies', $Movies,
  '--until-step', $UntilStep,
  '--model', $Model
)

if ($RunCalibration) {
  $cmd += '--run-calibration'
}

& $python -u @cmd
