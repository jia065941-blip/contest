$pythonCandidates = @(
    $env:COMPETITION_PYTHON
    $(if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' })
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
    $(Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }

$python = $pythonCandidates | Select-Object -First 1
if (-not $python) {
    throw 'CPython 3.11 x64 runtime not found. Create .venv with CPython 3.11, activate a virtual environment, or set COMPETITION_PYTHON.'
}

$env:PYTHONDONTWRITEBYTECODE = '1'
& $python (Join-Path $PSScriptRoot 'run.py') @args
exit $LASTEXITCODE
