param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$python = 'C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe'
$runtime = 'C:\tmp\competition_platform_runtime'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}
if (-not (Test-Path -LiteralPath $runtime)) {
    throw "Project dependencies not found: $runtime"
}

$env:PYTHONPATH = $runtime
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python (Join-Path $PSScriptRoot 'run.py') @Arguments
exit $LASTEXITCODE
