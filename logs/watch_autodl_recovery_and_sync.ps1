param(
    [int]$IntervalSeconds = 600,
    [int]$MaxAttempts = 24,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Continue"

if (-not $LogPath) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogPath = Join-Path $PSScriptRoot "autodl_recovery_watch_$stamp.log"
}

$syncScript = Join-Path $PSScriptRoot "sync_adapter_followup_guards_after_ssh.ps1"
$collector = Join-Path $PSScriptRoot "collect_adapter_quality_remote_metrics.ps1"
$gate = Join-Path $PSScriptRoot "evaluate_adapter_quality_gate.ps1"

function Write-Log {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $LogPath -Value $line
    Write-Host $line
}

Write-Log "AutoDL recovery watcher start; interval=${IntervalSeconds}s max_attempts=$MaxAttempts"
Write-Log "No approval sentinel will be created by this watcher."

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    Write-Log "attempt $attempt/${MaxAttempts}: sync follow-up guards"
    $syncOutput = & powershell -ExecutionPolicy Bypass -File $syncScript 2>&1
    $syncExit = $LASTEXITCODE
    $syncOutput | ForEach-Object { Add-Content -LiteralPath $LogPath -Value $_ }

    if ($syncExit -eq 0) {
        Write-Log "sync succeeded; collecting metrics and gate"
        $metricOutput = & powershell -ExecutionPolicy Bypass -File $collector -Tail 2600 -RetryCount 2 -RetryDelaySeconds 20 2>&1
        $gateOutput = $metricOutput | & powershell -ExecutionPolicy Bypass -File $gate 2>&1
        Add-Content -LiteralPath $LogPath -Value "===== collector ====="
        $metricOutput | ForEach-Object { Add-Content -LiteralPath $LogPath -Value $_ }
        Add-Content -LiteralPath $LogPath -Value "===== gate ====="
        $gateOutput | ForEach-Object { Add-Content -LiteralPath $LogPath -Value $_ }
        $gateOutput | ForEach-Object { Write-Host $_ }
        Write-Log "watcher complete after successful sync"
        exit 0
    }

    Write-Log "sync failed with exit=$syncExit"
    if ($attempt -lt $MaxAttempts) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}

Write-Log "watcher exhausted without successful sync"
exit 1
