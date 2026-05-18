param(
    [int]$IntervalSeconds = 600,
    [int]$MaxChecks = 24
)

$ErrorActionPreference = "Stop"

$collector = Join-Path $PSScriptRoot "collect_adapter_quality_remote_metrics.ps1"
$gate = Join-Path $PSScriptRoot "evaluate_adapter_quality_gate.ps1"

for ($i = 1; $i -le $MaxChecks; $i++) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "===== check $i/$MaxChecks at $stamp ====="
    $output = & powershell -ExecutionPolicy Bypass -File $collector -Tail 200
    $output | ForEach-Object { Write-Host $_ }

    if (($output -join "`n") -match "Average-mAP") {
        Write-Host "===== gate evaluation ====="
        $output | & powershell -ExecutionPolicy Bypass -File $gate
        Write-Host "Average-mAP detected; stopping watcher."
        exit 0
    }

    if ($i -lt $MaxChecks) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}

Write-Host "No Average-mAP detected after $MaxChecks checks."
