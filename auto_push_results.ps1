param(
    [string]$RepoPath = $PSScriptRoot,
    [string]$ResultsPath = "examples/results",
    [int]$PollIntervalMinutes = 10,
    [int]$QuietMinutes = 20,
    [string]$CommitPrefix = "auto-push results",
    [int]$MaxRunHours = 6
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $Message"
}

Set-Location $RepoPath
$startedAt = Get-Date
Write-Log "Auto-push watcher started. Repo: $RepoPath"
Write-Log "Monitoring: $ResultsPath | Poll every $PollIntervalMinutes min | Quiet threshold: $QuietMinutes min | Max runtime: $MaxRunHours h"

while ($true) {
    $elapsedHours = ((Get-Date) - $startedAt).TotalHours
    if ($elapsedHours -ge $MaxRunHours) {
        Write-Log "Max runtime reached ($MaxRunHours h). Stopping watcher."
        break
    }

    try {
        Set-Location $RepoPath

        $statusOutput = git status --porcelain -- "$ResultsPath"
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git status failed. Retrying in $PollIntervalMinutes minutes."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        if ([string]::IsNullOrWhiteSpace(($statusOutput | Out-String))) {
            Write-Log "No changes under $ResultsPath."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        $latestFile = Get-ChildItem -Path $ResultsPath -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if (-not $latestFile) {
            Write-Log "Changes detected but no files found under $ResultsPath yet."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        $minutesSinceWrite = ((Get-Date) - $latestFile.LastWriteTime).TotalMinutes

        if ($minutesSinceWrite -lt $QuietMinutes) {
            Write-Log "Changes detected, but files are still being written (latest: $($latestFile.FullName), $([math]::Round($minutesSinceWrite, 1)) min ago). Waiting."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        Write-Log "Changes are quiet enough. Staging and pushing results."

        git add -- "$ResultsPath"
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git add failed. Retrying later."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        $stagedOutput = git diff --cached --name-only -- "$ResultsPath"
        if ([string]::IsNullOrWhiteSpace(($stagedOutput | Out-String))) {
            Write-Log "No staged result changes after git add."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $commitMessage = "$CommitPrefix $timestamp"

        git commit -m "$commitMessage"
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git commit failed or nothing to commit. Retrying later."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        git push
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git push failed. Commit is local; will retry push on next cycle."
            Start-Sleep -Seconds ($PollIntervalMinutes * 60)
            continue
        }

        Write-Log "Auto-push successful."
    }
    catch {
        Write-Log "Unexpected error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds ($PollIntervalMinutes * 60)
}
