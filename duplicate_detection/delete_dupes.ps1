Write-Host "Starting deletion of 397,324 files..."
$start = Get-Date
$files = Get-Content "near_duplicate_deletions.txt"
$total = $files.Count
$deleted = 0
$errors = 0

foreach ($f in $files) {
    try {
        if (Test-Path $f -PathType Leaf) {
            Remove-Item $f -Force
            $deleted++
        }
    } catch {
        $errors++
    }
    if (($deleted + $errors) % 10000 -eq 0) {
        $pct = [math]::Round((($deleted + $errors) / $total) * 100, 1)
        Write-Host "Progress: $($deleted + $errors) / $total ($pct%) - Deleted: $deleted, Errors: $errors"
    }
}

$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "=== COMPLETE ==="
Write-Host "Deleted: $deleted files"
Write-Host "Errors: $errors"
Write-Host "Time: $($elapsed.Minutes)m $($elapsed.Seconds)s"
