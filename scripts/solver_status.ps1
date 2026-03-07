# Quick Solver Status Check
# Shows current solver status without polling

param(
    [string]$LogDir = "logs"
)

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
            if ($latestLog) { return $latestLog }
        }
    }
    return $null
}

function Get-SolverMetrics {
    param([string]$LogPath)
    
    $content = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $null }
    
    $metrics = @{
        CurrentStage = "Unknown"
        BestObjective = "N/A"
        BoundRange = "N/A"
        Gap = "N/A"
        MemoryPercent = "N/A"
        MemoryAvailable = "N/A"
        ProcessMemory = "N/A"
        CPU = "N/A"
        Workers = "N/A"
        Variables = "N/A"
        Constraints = "N/A"
        SolveTime = "N/A"
        Status = "Running"
        SolutionsFound = 0
        LastUpdate = (Get-Item $LogPath).LastWriteTime
        RunDir = "N/A"
    }
    
    # Get run directory
    $runMatch = [regex]::Match($content, "Run directory: (.+)")
    if ($runMatch.Success) {
        $metrics.RunDir = $runMatch.Groups[1].Value.Trim()
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
    
    # Count solutions found
    $solutionMatches = [regex]::Matches($content, "#(\d+)\s+[\d.]+s\s+best:")
    $metrics.SolutionsFound = $solutionMatches.Count
    
    # Get best objective and bounds
    $bestMatches = [regex]::Matches($content, "#\d+\s+[\d.]+s\s+best:(\d+)\s+next:\[(\d+),(\d+)\]")
    if ($bestMatches.Count -gt 0) {
        $lastMatch = $bestMatches[$bestMatches.Count - 1]
        $metrics.BestObjective = $lastMatch.Groups[1].Value
        $lower = [int]$lastMatch.Groups[2].Value
        $upper = [int]$lastMatch.Groups[3].Value
        $metrics.BoundRange = "[$lower, $upper]"
        $metrics.Gap = $upper - $lower
    }
    
    # Get bound-only updates
    $boundMatches = [regex]::Matches($content, "#Bound\s+([\d.]+)s\s+best:(\d+)\s+next:\[(\d+),(\d+)\]")
    if ($boundMatches.Count -gt 0) {
        $lastMatch = $boundMatches[$boundMatches.Count - 1]
        $metrics.SolveTime = "$($lastMatch.Groups[1].Value) seconds"
        if ($metrics.BestObjective -eq "N/A") {
            $metrics.BestObjective = $lastMatch.Groups[2].Value
        }
        $lower = [int]$lastMatch.Groups[3].Value
        $upper = [int]$lastMatch.Groups[4].Value
        $metrics.BoundRange = "[$lower, $upper]"
        $metrics.Gap = $upper - $lower
    }
    
    # Get latest memory/CPU
    $monitorMatches = [regex]::Matches($content, "MONITOR \| Memory: ([\d.]+)% \((\d+)MB used, (\d+)MB available\) \| Process: (\d+)MB \| CPU: ([\d.]+)%")
    if ($monitorMatches.Count -gt 0) {
        $lastMatch = $monitorMatches[$monitorMatches.Count - 1]
        $metrics.MemoryPercent = "$($lastMatch.Groups[1].Value)%"
        $metrics.MemoryAvailable = "$($lastMatch.Groups[3].Value) MB"
        $metrics.ProcessMemory = "$($lastMatch.Groups[4].Value) MB"
        $metrics.CPU = "$($lastMatch.Groups[5].Value)%"
    }
    
    # Get workers
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
    
    # Check status
    if ($content -match "OPTIMAL") {
        $metrics.Status = "OPTIMAL"
    } elseif ($content -match "FEASIBLE") {
        $metrics.Status = "FEASIBLE"
    } elseif ($content -match "INFEASIBLE") {
        $metrics.Status = "INFEASIBLE"
    }
    
    return $metrics
}

# Main
$logFile = Get-LatestLogFile

if (-not $logFile) {
    Write-Host "`n  No solver log files found!`n" -ForegroundColor Red
    exit 1
}

$metrics = Get-SolverMetrics -LogPath $logFile.FullName

if (-not $metrics) {
    Write-Host "`n  Could not parse log file!`n" -ForegroundColor Red
    exit 1
}

# Display compact status
$now = Get-Date -Format "HH:mm:ss"
$age = [math]::Round(((Get-Date) - $metrics.LastUpdate).TotalSeconds)

Write-Host ""
Write-Host "  ====== SOLVER STATUS @ $now ======" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Stage:      $($metrics.CurrentStage)" -ForegroundColor White
Write-Host "  Run:        $($metrics.RunDir)" -ForegroundColor Gray

# Status color
$statusColor = switch ($metrics.Status) {
    "Running" { "Green" }
    "OPTIMAL" { "Cyan" }
    "FEASIBLE" { "Yellow" }
    "INFEASIBLE" { "Red" }
    default { "White" }
}
Write-Host "  Status:     " -NoNewline; Write-Host $metrics.Status -ForegroundColor $statusColor

Write-Host ""
Write-Host "  --- Optimization ---" -ForegroundColor Yellow
Write-Host "  Best:       $($metrics.BestObjective)" -ForegroundColor Green
Write-Host "  Bounds:     $($metrics.BoundRange)" -ForegroundColor White
Write-Host "  Gap:        $($metrics.Gap)" -ForegroundColor White
Write-Host "  Solutions:  $($metrics.SolutionsFound)" -ForegroundColor White
Write-Host "  Time:       $($metrics.SolveTime)" -ForegroundColor White

Write-Host ""
Write-Host "  --- Resources ---" -ForegroundColor Yellow

# Memory color
$memColor = "Green"
if ($metrics.MemoryPercent -match "([\d.]+)") {
    $memVal = [float]$Matches[1]
    if ($memVal -gt 90) { $memColor = "Red" }
    elseif ($memVal -gt 80) { $memColor = "Yellow" }
}
Write-Host "  Memory:     " -NoNewline; Write-Host "$($metrics.MemoryPercent) (Process: $($metrics.ProcessMemory))" -ForegroundColor $memColor
Write-Host "  CPU:        $($metrics.CPU)" -ForegroundColor White
Write-Host "  Workers:    $($metrics.Workers)" -ForegroundColor White

Write-Host ""
Write-Host "  Log updated ${age}s ago" -ForegroundColor Gray
Write-Host ""
