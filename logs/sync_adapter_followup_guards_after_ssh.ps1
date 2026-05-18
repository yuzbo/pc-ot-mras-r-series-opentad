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
}

function Copy-RemoteFile {
    param(
        [int]$Port,
        [string]$LocalPath,
        [string]$RemotePath
    )
    & $Scp -P $Port $LocalPath "root@${HostName}:$RemotePath"
}

$queueGuard = Join-Path $repo "scripts\wait_for_screen_and_gate_then_run.sh"
$pseudoLauncher = Join-Path $repo "scripts\run_adapter_pseudo_boundary_snap_pair.sh"

Write-Host "== Sync queue guard to both servers =="
Copy-RemoteFile -Port $PseudoPort -LocalPath $queueGuard -RemotePath "$RemoteDir/scripts/wait_for_screen_and_gate_then_run.sh"
Copy-RemoteFile -Port $ReglossPort -LocalPath $queueGuard -RemotePath "$RemoteDir/scripts/wait_for_screen_and_gate_then_run.sh"

Write-Host "== Sync pseudo-boundary launcher to 35407 =="
Copy-RemoteFile -Port $PseudoPort -LocalPath $pseudoLauncher -RemotePath "$RemoteDir/scripts/run_adapter_pseudo_boundary_snap_pair.sh"

Write-Host "== Remote syntax and q64 check-only on 35407 =="
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

Write-Host "== Remote queue guard syntax on 25876 =="
$reglossCheck = @"
set -euo pipefail
cd $RemoteDir
chmod +x scripts/wait_for_screen_and_gate_then_run.sh
bash -n scripts/wait_for_screen_and_gate_then_run.sh
test ! -e gate_approvals/adapter_regloss15_after_quality.ok
echo REGLOSS_GUARD_CHECK_OK
"@
Invoke-Remote -Port $ReglossPort -Script $reglossCheck

Write-Host "Follow-up guard sync complete. Approval sentinels were not created."
