param(
    [string]$SourceRoot = "E:\tmp\thumos14_raw",
    [string]$RemoteHostName = "ssh.cn-zhongwei-1.paracloud.com",
    [string]$RemoteUser = "sczc063@BSCC-N16R4",
    [string]$IdentityFile = "C:\Users\skywalker\.ssh\id_rsa",
    [string]$RemoteRoot = "~/run/yuzibo/thumos14",
    [string]$LogFile = "E:\DeskTop\TAD\temrefuse-tad\logs\sync_thumos_n16r4_transfer.log"
)

$ErrorActionPreference = "Stop"

$sshCommon = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=30",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=10",
    "-o", "IdentitiesOnly=yes",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    "-o", "HostkeyAlgorithms=+ssh-rsa",
    "-i", $IdentityFile
)

$scpCommon = $sshCommon + @("-o", "User=$RemoteUser", "-P", "22")
$sshLogin = $sshCommon + @("-p", "22", "-l", $RemoteUser, $RemoteHostName)

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $LogFile -Value $line
}

function Invoke-Remote {
    param([string]$Command)
    & ssh @sshLogin $Command
    if ($LASTEXITCODE -ne 0) {
        throw "remote command failed with exit code $LASTEXITCODE"
    }
}

function Get-RemoteManifest {
    param([string]$Split)

    $cmd = "cd ~/run/yuzibo && mkdir -p thumos14/$Split && find thumos14/$Split -maxdepth 1 -type f -printf '%f`t%s`n' 2>/dev/null"
    $lines = & ssh @sshLogin $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "failed to read remote manifest for $Split"
    }

    $manifest = @{}
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $parts = $line -split "`t", 2
        if ($parts.Count -eq 2) {
            $manifest[$parts[0]] = [int64]$parts[1]
        }
    }
    return $manifest
}

function Sync-Split {
    param(
        [string]$LocalDir,
        [string]$Split
    )

    if (-not (Test-Path -LiteralPath $LocalDir -PathType Container)) {
        throw "missing local directory: $LocalDir"
    }

    Write-Log "reading remote manifest for $Split"
    $remote = Get-RemoteManifest -Split $Split
    $files = Get-ChildItem -LiteralPath $LocalDir -File -Filter "*.mp4" | Sort-Object Name
    $total = $files.Count
    $index = 0
    $copied = 0
    $skipped = 0
    $bytesCopied = [int64]0

    Write-Log "sync $Split start: local_files=$total local_dir=$LocalDir"
    foreach ($file in $files) {
        $index += 1
        $remoteSize = $remote[$file.Name]
        if ($null -ne $remoteSize -and [int64]$remoteSize -eq [int64]$file.Length) {
            $skipped += 1
            if ($index % 50 -eq 0) {
                Write-Log "sync $Split progress: $index/$total copied=$copied skipped=$skipped"
            }
            continue
        }

        $dest = "${RemoteHostName}:${RemoteRoot}/$Split/$($file.Name)"
        Write-Log "copy $Split $index/$total $($file.Name) size=$($file.Length)"
        & scp @scpCommon -- $file.FullName $dest
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed for $($file.FullName) with exit code $LASTEXITCODE"
        }

        $copied += 1
        $bytesCopied += [int64]$file.Length
    }

    Write-Log "sync $Split done: copied=$copied skipped=$skipped bytes_copied=$bytesCopied"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null
Write-Log "THUMOS14 N16R4 sync start"
Invoke-Remote "cd ~/run/yuzibo && mkdir -p thumos14/train thumos14/test thumos14/annotations"

Sync-Split `
    -LocalDir (Join-Path $SourceRoot "Validation Data\validation") `
    -Split "train"

Sync-Split `
    -LocalDir (Join-Path $SourceRoot "Test Data\TH14_test_set_mp4") `
    -Split "test"

Invoke-Remote "cd ~/run/yuzibo && find thumos14/train -maxdepth 1 -type f -name '*.mp4' | wc -l && find thumos14/test -maxdepth 1 -type f -name '*.mp4' | wc -l && du -sh thumos14"
Write-Log "THUMOS14 N16R4 sync complete"
