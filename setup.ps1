param(
    [string]$BasePython = 'python',
    [string]$PackageIndex = 'https://pypi.org/simple'
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    & $BasePython -m venv --system-site-packages (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $taskPython -m pip install --disable-pip-version-check --no-cache-dir --index-url $PackageIndex 'numpy>=1.26,<3' 'scipy>=1.11' 'soundfile>=0.12' 'requests>=2.28'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $taskPython (Join-Path $PSScriptRoot 'download_assets.py') --tools-only
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg download failed.' }
