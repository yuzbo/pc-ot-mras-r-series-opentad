param(
    [string]$InputPath = "",
    [double]$BaselineAverageMap = 61.95,
    [double]$PreservationGate = 60.0,
    [double]$StopGate = 30.0
)

$ErrorActionPreference = "Stop"

if ($InputPath) {
    if (-not (Test-Path -LiteralPath $InputPath)) {
        throw "Input file not found: $InputPath"
    }
    $raw = Get-Content -LiteralPath $InputPath -Raw
} else {
    $raw = ($input | Out-String)
    if ([string]::IsNullOrWhiteSpace($raw)) {
        $raw = [Console]::In.ReadToEnd()
    }
}

if ([string]::IsNullOrWhiteSpace($raw)) {
    Write-Host "No collector output provided."
    exit 0
}

function New-RunState {
    param([string]$Name)
    [pscustomobject]@{
        Name = $Name
        Log = ""
        Blocks = New-Object System.Collections.Generic.List[object]
        Current = [ordered]@{}
    }
}

function Add-CurrentBlock {
    param($Run)
    if ($null -eq $Run) {
        return
    }
    if ($Run.Current.Contains("Average-mAP")) {
        $Run.Blocks.Add([pscustomobject]$Run.Current)
        $Run.Current = [ordered]@{}
    }
}

$runs = [ordered]@{}
$currentRun = $null

foreach ($line in ($raw -split "`r?`n")) {
    if ($line -match "^RUN=(.+)$") {
        Add-CurrentBlock -Run $currentRun
        $name = $matches[1].Trim()
        if (-not $runs.Contains($name)) {
            $runs[$name] = New-RunState -Name $name
        }
        $currentRun = $runs[$name]
        continue
    }
    if ($line -match "^LOG=(.*)$") {
        if ($null -ne $currentRun) {
            $currentRun.Log = $matches[1].Trim()
        }
        continue
    }
    if ($line -match "mAP at tIoU ([0-9.]+) is ([0-9.]+)%") {
        if ($null -ne $currentRun) {
            $currentRun.Current["mAP@$($matches[1])"] = [double]$matches[2]
        }
        continue
    }
    if ($line -match "Average-mAP:\s*([0-9.]+)") {
        if ($null -ne $currentRun) {
            $currentRun.Current["Average-mAP"] = [double]$matches[1]
            Add-CurrentBlock -Run $currentRun
        }
        continue
    }
}

Add-CurrentBlock -Run $currentRun

if ($runs.Count -eq 0) {
    Write-Host "No RUN sections found in collector output."
    exit 0
}

$neutralFirst = $null
$neutralLatest = $null
$negFirst = $null
$negLatest = $null

foreach ($run in $runs.Values) {
    Write-Host "===== gate / $($run.Name) ====="
    if ($run.Log) {
        Write-Host "LOG=$($run.Log)"
    }

    if ($run.Blocks.Count -eq 0) {
        Write-Host "STATUS=NO_EVAL_YET"
        continue
    }

    $first = $run.Blocks[0]
    $latest = $run.Blocks[$run.Blocks.Count - 1]
    $firstMap = [double]$first.'Average-mAP'
    $latestMap = [double]$latest.'Average-mAP'

    Write-Host ("EVAL_COUNT={0}" -f $run.Blocks.Count)
    Write-Host ("FIRST_AVERAGE_MAP={0:N2}" -f $firstMap)
    Write-Host ("LATEST_AVERAGE_MAP={0:N2}" -f $latestMap)
    if ($latest.PSObject.Properties.Name -contains "mAP@0.70") {
        Write-Host ("LATEST_MAP_070={0:N2}" -f ([double]$latest.'mAP@0.70'))
    }

    if ($run.Name -eq "neutral") {
        $neutralFirst = $firstMap
        $neutralLatest = $latestMap
        if ($firstMap -lt $StopGate) {
            Write-Host "GATE=STOP_NOW_BELOW_FIRST_EVAL_FLOOR"
        } elseif ($latestMap -lt $PreservationGate) {
            Write-Host "GATE=QUALITY_BRANCH_PRESERVATION_NOT_PROVEN"
        } elseif ($latestMap -ge $BaselineAverageMap) {
            Write-Host "GATE=QUALITY_BRANCH_PRESERVATION_PASS"
        } else {
            Write-Host "GATE=BORDERLINE_RECOVERY_NEEDS_NEXT_EVAL"
        }
        continue
    }

    if ($run.Name -eq "neg025") {
        $negFirst = $firstMap
        $negLatest = $latestMap
        if ($firstMap -lt $StopGate) {
            Write-Host "GATE=STOP_NOW_BELOW_FIRST_EVAL_FLOOR"
        } elseif ($latestMap -lt $PreservationGate) {
            Write-Host "GATE=QUALITY_SUPERVISION_NOT_RECOVERED"
        } elseif ($latestMap -ge $BaselineAverageMap) {
            Write-Host "GATE=CHECKPOINT_BASELINE_RECOVERED_ALPHA_SWEEP_CANDIDATE"
        } else {
            Write-Host "GATE=BORDERLINE_RECOVERY_NO_ALPHA_SWEEP_YET"
        }
        continue
    }

    Write-Host "GATE=UNCLASSIFIED_RUN"
}

Write-Host "===== combined decision ====="
if ($null -eq $neutralLatest) {
    Write-Host "DECISION=WAIT_FOR_NEUTRAL_FIRST_EVAL"
    exit 0
}

if ($neutralFirst -lt $StopGate) {
    Write-Host "DECISION=STOP_NEUTRAL_AND_AUDIT_IMPLEMENTATION"
    exit 0
}

if ($neutralLatest -lt $PreservationGate) {
    Write-Host "DECISION=DO_NOT_INTERPRET_NEG025_OR_RUN_ALPHA_SWEEP"
    exit 0
}

if ($neutralLatest -lt $BaselineAverageMap) {
    Write-Host "DECISION=WAIT_FOR_STRONGER_NEUTRAL_RECOVERY"
    exit 0
}

if ($null -eq $negLatest) {
    Write-Host "DECISION=NEUTRAL_PASS_WAIT_FOR_NEG025"
    exit 0
}

if ($negFirst -lt $StopGate) {
    Write-Host "DECISION=STOP_NEG025_AND_AUDIT_QUALITY_LOSS"
} elseif ($negLatest -ge $BaselineAverageMap) {
    Write-Host "DECISION=NEG025_BASELINE_RECOVERED_PREPARE_INFERENCE_ALPHA_SWEEP"
} else {
    Write-Host "DECISION=QUALITY_BRANCH_SAFE_BUT_NEG025_NOT_BASELINE"
}
