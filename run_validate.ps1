param(
    [string]$InputDir = '',
    [string]$OutputDir = '',
    [string]$Video = ''
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first.' }
if (-not $InputDir) { $InputDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'new_100_samp_720p' }
if (-not $OutputDir) { $OutputDir = Join-Path $PSScriptRoot 'outputs\energy_gcc_100ms_summary' }
$taskArgs = @((Join-Path $PSScriptRoot 'evaluate_spatial_cues.py'),'--input',$InputDir,'--out',$OutputDir)
if ($Video) { $taskArgs += @('--video',$Video) }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Validation failed with exit code $LASTEXITCODE" }
