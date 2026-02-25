# Solver Progress Poller
# Polls the solver log file every 5 minutes and displays key metrics

param(
    [int]$IntervalMinutes = 5,
    [string]$LogDir = "logs"
)

$host.UI.RawUI.WindowTitle = "Solver Progress Monitor"

function Get-LatestLogFile {
    $paths = @(
        "c:\Users\c3205\Documents\Code\python\draw\logs",
        (Join-Path $PSScriptRoot $LogDir),
        (Join-Path (Split-Path $PSScriptRoot -Parent) $LogDir),
        "logs",
        "refactored\logs"
    )
    
    foreach ($logPath in $paths) {
        if (Test-Path $logPath) {
            $latestLog = Get-ChildItem -Path $logPath -Filter "solver_*.log" -ErrorAction SilentlyContinue | 
                         Sort-Object LastWriteTime -Descending | 
                         Select-Object -First 1
            if ($latestLog) {
                return $latestLog
            }
        }
    }
    return $null
}

function Get-SolverMetrics {
    param([string]$LogPath)
    
    if (-not (Test-Path $LogPath)) {
        return $null
    }
    
    $content = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $null }
    
    $metrics = @{
        CurrentStage = "Unknown"
        BestObjective = "N/A"
        BoundRange = "N/A"
        MemoryUsed = "N/A"
        MemoryAvailable = "N/A"
        ProcessMemory = "N/A"
        CPU = "N/A"
        Workers = "N/A"
        Variables = "N/A"
        Constraints = "N/A"
        SolveTime = "N/A"
        Status = "Running"
        LastUpdate = (Get-Item $LogPath).LastWriteTime
    }
    
    # Get current stage
    $stageMatches = [regex]::Matches($content, "Starting solve for stage: (\w+)")
    if ($stageMatches.Count -gt 0) {
        $metrics.CurrentStage = $stageMatches[$stageMatches.Count - 1].Groups[1].Value
    }
    
    # Get completed stages
    $completedMatches = [regex]::Matches($content, "Stage (\w+) completed")
    $metrics.CompletedStages = @()
    foreach ($match in $completedMatches) {
        $metrics.CompletedStages += $match.Groups[1].Value
    }
    
    # Get best objective and bounds from solver output
    $bestMatches = [regex]::Matches($content, "#\d+\s+[\d.]+s\s+best:(\d+)\s+next:\[(\d+),(\d+)\]")
    if ($bestMatches.Count -gt 0) {
        $lastMatch = $bestMatches[$bestMatches.Count - 1]
        $metrics.BestObjective = $lastMatch.Groups[1].Value
        $metrics.BoundRange = "[$($lastMatch.Groups[2].Value), $($lastMatch.Groups[3].Value)]"
    }
    
    # Get bound-only updates
    $boundMatches = [regex]::Matches($content, "#Bound\s+[\d.]+s\s+best:(\d+)\s+next:\[(\d+),(\d+)\]")
    if ($boundMatches.Count -gt 0) {
        $lastMatch = $boundMatches[$boundMatches.Count - 1]
        if ($metrics.BestObjective -eq "N/A") {
            $metrics.BestObjective = $lastMatch.Groups[1].Value
        }
        $metrics.BoundRange = "[$($lastMatch.Groups[2].Value), $($lastMatch.Groups[3].Value)]"
    }
    
    # Get latest memory/CPU from monitor
    $monitorMatches = [regex]::Matches($content, "MONITOR \| Memory: ([\d.]+)% \((\d+)MB used, (\d+)MB available\) \| Process: (\d+)MB \| CPU: ([\d.]+)%")
    if ($monitorMatches.Count -gt 0) {
        $lastMatch = $monitorMatches[$monitorMatches.Count - 1]
        $metrics.MemoryUsed = "$($lastMatch.Groups[1].Value)% ($($lastMatch.Groups[2].Value) MB)"
        $metrics.MemoryAvailable = "$($lastMatch.Groups[3].Value) MB"
        $metrics.ProcessMemory = "$($lastMatch.Groups[4].Value) MB"
        $metrics.CPU = "$($lastMatch.Groups[5].Value)%"
    }
    
    # Get workers config
    $workerMatch = [regex]::Match($content, "workers=(\d+)")
    if ($workerMatch.Success) {
        $metrics.Workers = $workerMatch.Groups[1].Value
    }
    
    # Get model size
    $varMatch = [regex]::Match($content, "#Variables: ([\d']+)")
    if ($varMatch.Success) {
        $metrics.Variables = $varMatch.Groups[1].Value
    }
    
    $constMatch = [regex]::Match($content, "constraints:(\d+)/(\d+)")
    if ($constMatch.Success) {
        $metrics.Constraints = $constMatch.Groups[2].Value
    }
    
    # Get solve time from solver output
    $timeMatches = [regex]::Matches($content, "([\d.]+)s\s+best:")
    if ($timeMatches.Count -gt 0) {
        $lastTime = $timeMatches[$timeMatches.Count - 1].Groups[1].Value
        $metrics.SolveTime = "$lastTime seconds"
    }
    
    # Check if completed or failed
    if ($content -match "OPTIMAL|FEASIBLE") {
        $metrics.Status = "Completed"
    }
    if ($content -match "INFEASIBLE") {
        $metrics.Status = "INFEASIBLE - No solution exists"
    }
    if ($content -match "error|Error|ERROR|exception|Exception") {
        $metrics.Status = "Error detected"
    }
    
    return $metrics
}

function Show-Dashboard {
    param($Metrics, $LogFile, $PollCount)
    
    Clear-Host
    
    $separator = "=" * 60
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Host ""
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "       HOCKEY DRAW SOLVER - PROGRESS MONITOR" -ForegroundColor Cyan
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Poll #$PollCount | Current Time: $now" -ForegroundColor Gray
    Write-Host "  Log File: $($LogFile.Name)" -ForegroundColor Gray
    Write-Host "  Last Modified: $($Metrics.LastUpdate)" -ForegroundColor Gray
    Write-Host ""
    Write-Host $separator -ForegroundColor Yellow
    Write-Host "  SOLVER STATUS" -ForegroundColor Yellow
    Write-Host $separator -ForegroundColor Yellow
    Write-Host ""
    
    # Status with color
    $statusColor = switch ($Metrics.Status) {
        "Running" { "Green" }
        "Completed" { "Cyan" }
        "Error detected" { "Red" }
        "INFEASIBLE - No solution exists" { "Red" }
        default { "White" }
    }
    Write-Host "  Status:           " -NoNewline; Write-Host $Metrics.Status -ForegroundColor $statusColor
    Write-Host "  Current Stage:    " -NoNewline; Write-Host $Metrics.CurrentStage -ForegroundColor White
    
    if ($Metrics.CompletedStages.Count -gt 0) {
        Write-Host "  Completed Stages: " -NoNewline; Write-Host ($Metrics.CompletedStages -join ", ") -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host $separator -ForegroundColor Yellow
    Write-Host "  OPTIMIZATION PROGRESS" -ForegroundColor Yellow
    Write-Host $separator -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Best Objective:   " -NoNewline; Write-Host $Metrics.BestObjective -ForegroundColor Green
    Write-Host "  Bound Range:      " -NoNewline; Write-Host $Metrics.BoundRange -ForegroundColor White
    Write-Host "  Solve Time:       " -NoNewline; Write-Host $Metrics.SolveTime -ForegroundColor White
    Write-Host ""
    Write-Host $separator -ForegroundColor Yellow
    Write-Host "  RESOURCE USAGE" -ForegroundColor Yellow
    Write-Host $separator -ForegroundColor Yellow
    Write-Host ""
    
    # Memory with color coding
    $memPercent = 0
    if ($Metrics.MemoryUsed -match "([\d.]+)%") {
        $memPercent = [float]$Matches[1]
    }
    $memColor = if ($memPercent -gt 90) { "Red" } elseif ($memPercent -gt 80) { "Yellow" } else { "Green" }
    
    Write-Host "  Memory Used:      " -NoNewline; Write-Host $Metrics.MemoryUsed -ForegroundColor $memColor
    Write-Host "  Memory Available: " -NoNewline; Write-Host $Metrics.MemoryAvailable -ForegroundColor White
    Write-Host "  Process Memory:   " -NoNewline; Write-Host $Metrics.ProcessMemory -ForegroundColor White
    Write-Host "  CPU Usage:        " -NoNewline; Write-Host $Metrics.CPU -ForegroundColor White
    Write-Host ""
    Write-Host $separator -ForegroundColor Yellow
    Write-Host "  MODEL INFO" -ForegroundColor Yellow
    Write-Host $separator -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Workers:          " -NoNewline; Write-Host $Metrics.Workers -ForegroundColor White
    Write-Host "  Variables:        " -NoNewline; Write-Host $Metrics.Variables -ForegroundColor White
    Write-Host "  Constraints:      " -NoNewline; Write-Host $Metrics.Constraints -ForegroundColor White
    Write-Host ""
    Write-Host $separator -ForegroundColor Cyan
    Write-Host "  Next update in $IntervalMinutes minutes... (Press Ctrl+C to stop)" -ForegroundColor Gray
    Write-Host $separator -ForegroundColor Cyan
    Write-Host ""
}

# Main loop
$pollCount = 0
Write-Host "Starting Solver Progress Monitor..." -ForegroundColor Green
Write-Host "Polling every $IntervalMinutes minutes" -ForegroundColor Gray
Write-Host ""

while ($true) {
    $pollCount++
    
    $logFile = Get-LatestLogFile
    
    if ($logFile) {
        $metrics = Get-SolverMetrics -LogPath $logFile.FullName
        if ($metrics) {
            Show-Dashboard -Metrics $metrics -LogFile $logFile -PollCount $pollCount
        } else {
            Write-Host "[$pollCount] $(Get-Date -Format 'HH:mm:ss') - Could not parse log file" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[$pollCount] $(Get-Date -Format 'HH:mm:ss') - No solver log files found in 'logs' directory" -ForegroundColor Yellow
        Write-Host "Waiting for solver to start..." -ForegroundColor Gray
    }
    
    # Sleep for interval
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
