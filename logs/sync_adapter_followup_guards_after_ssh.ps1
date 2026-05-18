param(
    [int]$PseudoPort = 35407,
    [int]$ReglossPort = 25876,
    [string]$HostName = "connect.cqa1.seetacloud.com",
    [string]$RemoteDir = "/root/autodl-tmp/OpenTAD_Back_check",
    [string]$LocalRepo = "OpenTAD_Back"
)

$ErrorActionPreference = "Stop"

$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$Scp = "C:\Windows\System32\OpenSSH\scp.exe"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$repo = Resolve-Path (Join-Path $root $LocalRepo)

function Invoke-Remote {
    param(
        [int]$Port,
        [string]$Script
    )
    ($Script -replace "`r", "") | & $Ssh -o BatchMode=yes -o ConnectTimeout=20 -p $Port "root@$HostName" "bash -s"
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed on port $Port with exit code $LASTEXITCODE"
    }
}

function Copy-RemoteFile {
    param(
        [int]$Port,
        [string]$LocalPath,
        [string]$RemotePath
    )
    & $Scp -P $Port $LocalPath "root@${HostName}:$RemotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed on port $Port with exit code $LASTEXITCODE for $LocalPath"
    }
}

$queueGuard = Join-Path $repo "scripts\wait_for_screen_and_gate_then_run.sh"
$pseudoLauncher = Join-Path $repo "scripts\run_adapter_pseudo_boundary_snap_pair.sh"
$reglossLauncher = Join-Path $repo "scripts\run_adapter_actionformer_regloss.sh"
$simotaLauncher = Join-Path $repo "scripts\run_adapter_simota_iou_sum.sh"
$simotaAssigner = Join-Path $repo "opentad\models\losses\assigner\anchor_free_simota_assigner.py"
$simotaPlainConfig = Join-Path $repo "configs\adatad\thumos\input_random_fixed_50pct_adapter_simota_mink4_w1.py"

$script:SyncFailures = New-Object 'System.Collections.Generic.List[string]'

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host "== $Name =="
    try {
        & $Action
        Write-Host "OK: $Name"
    } catch {
        $message = $_.Exception.Message
        Write-Host "FAILED: $Name"
        Write-Host $message
        $script:SyncFailures.Add("${Name}: $message")
    }
}

Invoke-Step -Name "Sync and check q64 guard on 35407" -Action {
    Copy-RemoteFile -Port $PseudoPort -LocalPath $queueGuard -RemotePath "$RemoteDir/scripts/wait_for_screen_and_gate_then_run.sh"
    Copy-RemoteFile -Port $PseudoPort -LocalPath $pseudoLauncher -RemotePath "$RemoteDir/scripts/run_adapter_pseudo_boundary_snap_pair.sh"
    $pseudoCheck = @"
set -euo pipefail
cd $RemoteDir
chmod +x scripts/wait_for_screen_and_gate_then_run.sh scripts/run_adapter_pseudo_boundary_snap_pair.sh
bash -n scripts/wait_for_screen_and_gate_then_run.sh scripts/run_adapter_pseudo_boundary_snap_pair.sh
CHECK_ONLY=1 START_INDEX=1 END_INDEX=1 SKIP_CACHE_BUILD=1 bash scripts/run_adapter_pseudo_boundary_snap_pair.sh
test ! -e gate_approvals/adapter_pseudo_snap_q64_after_quality.ok
echo Q64_GUARD_SYNC_AND_CHECK_OK
"@
    Invoke-Remote -Port $PseudoPort -Script $pseudoCheck
}

Invoke-Step -Name "Sync and check regloss/SimOTA guard on 25876" -Action {
    Copy-RemoteFile -Port $ReglossPort -LocalPath $queueGuard -RemotePath "$RemoteDir/scripts/wait_for_screen_and_gate_then_run.sh"
    Copy-RemoteFile -Port $ReglossPort -LocalPath $reglossLauncher -RemotePath "$RemoteDir/scripts/run_adapter_actionformer_regloss.sh"
    Copy-RemoteFile -Port $ReglossPort -LocalPath $simotaLauncher -RemotePath "$RemoteDir/scripts/run_adapter_simota_iou_sum.sh"
    Copy-RemoteFile -Port $ReglossPort -LocalPath $simotaAssigner -RemotePath "$RemoteDir/opentad/models/losses/assigner/anchor_free_simota_assigner.py"
    Copy-RemoteFile -Port $ReglossPort -LocalPath $simotaPlainConfig -RemotePath "$RemoteDir/configs/adatad/thumos/input_random_fixed_50pct_adapter_simota_mink4_w1.py"
    $reglossCheck = @"
set -euo pipefail
cd $RemoteDir
chmod +x scripts/wait_for_screen_and_gate_then_run.sh scripts/run_adapter_actionformer_regloss.sh scripts/run_adapter_simota_iou_sum.sh
bash -n scripts/wait_for_screen_and_gate_then_run.sh scripts/run_adapter_actionformer_regloss.sh scripts/run_adapter_simota_iou_sum.sh
CHECK_ONLY=1 bash scripts/run_adapter_actionformer_regloss.sh
CHECK_ONLY=1 bash scripts/run_adapter_simota_iou_sum.sh
test ! -e gate_approvals/adapter_regloss15_after_quality.ok
echo REGLOSS_GUARD_CHECK_OK
echo SIMOTA_BACKUP_CHECK_OK
"@
    Invoke-Remote -Port $ReglossPort -Script $reglossCheck
}

if ($script:SyncFailures.Count -gt 0) {
    Write-Host "Follow-up guard sync finished with failures. Approval sentinels were not created."
    $script:SyncFailures | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host "Follow-up guard sync complete. Approval sentinels were not created."
