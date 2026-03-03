param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$python = $env:SPIKEINTERFACE_PYTHON
if (-not $python -or -not (Test-Path $python)) {
    $python = "C:\Users\ryoi\AppData\Local\anaconda3\envs\spikeinterface\python.exe"
}

if (-not (Test-Path $python)) {
    Write-Error "SpikeInterface Python not found. Set SPIKEINTERFACE_PYTHON or update run_spikeinterface.ps1."
    exit 1
}

if (-not $Args -or $Args.Count -eq 0) {
    Write-Host "Usage: .\run_spikeinterface.ps1 <python args>"
    Write-Host "Example: .\run_spikeinterface.ps1 -c `"import spikeinterface as si; print(si.__version__)`""
    exit 2
}

& $python @Args
exit $LASTEXITCODE

