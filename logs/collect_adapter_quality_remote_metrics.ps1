param(
    [int]$Tail = 200,
    [int]$RetryCount = 2,
    [int]$RetryDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$RemoteDir = "/root/autodl-tmp/OpenTAD_Back_check"

$Runs = @(
    @{
        Name = "neutral"
        Port = 35407
        Pattern = "input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0"
    },
    @{
        Name = "neg025"
        Port = 25876
        Pattern = "input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0"
    }
)

foreach ($run in $Runs) {
    $remoteScript = @"
set -u
cd $RemoteDir
latest=`$(ls -t logs/$($run.Pattern)_*.log 2>/dev/null | head -1)
echo LOG=`$latest
if [ -z "`$latest" ]; then
  echo NO_LOG
  exit 0
fi
echo METRICS
grep -E 'mAP at tIoU|Average-mAP|Testing Over|Training Over' "`$latest" | tail -$Tail || true
echo LATEST_TRAIN
grep -E '\[Train\]: \[[0-9]+\]\[[0-9]+/[0-9]+\]' "`$latest" | tail -5 || true
"@

    Write-Host "===== $($run.Name) / $($run.Port) ====="
    Write-Host "SERVER_PORT=$($run.Port)"
    Write-Host "RUN=$($run.Name)"

    $sshOutput = @()
    $sshExit = 1

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
        Write-Host "REMOTE_ATTEMPT=$attempt/$RetryCount"

        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $sshOutput = ($remoteScript -replace "`r", "") | & $Ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -p $run.Port root@connect.cqa1.seetacloud.com "bash -s" 2>&1
            $sshExit = $LASTEXITCODE
        } catch {
            $sshOutput = @($_.Exception.Message)
            $sshExit = 1
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }

        if ($sshExit -eq 0) {
            break
        }

        if ($attempt -lt $RetryCount -and $RetryDelaySeconds -gt 0) {
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }

    if ($sshExit -eq 0) {
        Write-Host "REMOTE_STATUS=OK"
    } else {
        Write-Host "REMOTE_STATUS=SSH_FAILED"
        Write-Host "REMOTE_EXIT=$sshExit"
    }

    $sshOutput | ForEach-Object { Write-Host $_ }
}
