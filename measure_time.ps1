# Measure Pipeline Execution Time (Cache Speed Verification)
$sw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "Running Phase 5 Pipeline..." -ForegroundColor Cyan
python scripts/phase5_full_pipeline.py

$sw.Stop()
$elapsed = $sw.Elapsed.TotalSeconds

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Execution Time: $($elapsed.ToString('F2')) seconds" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($elapsed -lt 30) {
    Write-Host "PASS: Cache Speed criteria met (< 30s)" -ForegroundColor Green
} else {
    Write-Host "WARNING: Cache Speed criteria not met (> 30s)" -ForegroundColor Yellow
}
