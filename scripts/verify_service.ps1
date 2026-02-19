$ErrorActionPreference = 'Stop'

$passes = 0
$fails = 0

function Pass($msg) {
    $script:passes++
    Write-Host "[PASS] $msg" -ForegroundColor Green
}

function Fail($msg) {
    $script:fails++
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

Write-Host "=== Sentiment MLOps Service Verification ===" -ForegroundColor Cyan

# 1) Container status
try {
    $ps = docker compose ps --format json | ConvertFrom-Json
    if (-not $ps) {
        Fail "No docker compose services found"
    } else {
        $ml = $ps | Where-Object { $_.Service -eq 'mlflow-server' }
        $api = $ps | Where-Object { $_.Service -eq 'fastapi-app' }

        if ($ml -and $ml.State -eq 'running') { Pass "mlflow-server is running" } else { Fail "mlflow-server not running" }
        if ($api -and $api.State -eq 'running') { Pass "fastapi-app is running" } else { Fail "fastapi-app not running" }
    }
}
catch {
    Fail "Failed to query docker compose status: $($_.Exception.Message)"
}

# 2) Health endpoint
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    if ($health.status -eq 'healthy') { Pass "/health returns healthy" } else { Fail "/health status is $($health.status)" }
    if ($health.model_loaded -eq $true) { Pass "Model is loaded" } else { Fail "Model not loaded" }
}
catch {
    Fail "Health endpoint failed: $($_.Exception.Message)"
}

# 3) Info endpoint
try {
    $info = Invoke-RestMethod -Uri "http://localhost:8000/info" -Method Get
    if ($info.model_name -eq 'SentimentClassifier') { Pass "Model name is SentimentClassifier" } else { Fail "Unexpected model name: $($info.model_name)" }
    if ($info.model_stage -eq 'production') { Pass "Model stage is production" } else { Fail "Unexpected model stage: $($info.model_stage)" }
}
catch {
    Fail "Info endpoint failed: $($_.Exception.Message)"
}

# 4) Positive/Negative prediction checks
try {
    $posBody = @{ text = 'This movie was amazing, emotional and beautifully directed.' } | ConvertTo-Json -Compress
    $pos = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $posBody
    if ($pos.sentiment -eq 'positive') { Pass "Positive sample predicted as positive" } else { Fail "Positive sample predicted as $($pos.sentiment)" }
}
catch {
    Fail "Positive prediction failed: $($_.Exception.Message)"
}

try {
    $negBody = @{ text = 'Terrible movie, boring plot and bad acting. Waste of time.' } | ConvertTo-Json -Compress
    $neg = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $negBody
    if ($neg.sentiment -eq 'negative') { Pass "Negative sample predicted as negative" } else { Fail "Negative sample predicted as $($neg.sentiment)" }
}
catch {
    Fail "Negative prediction failed: $($_.Exception.Message)"
}

# 5) Validation/error behavior
try {
    $emptyBody = @{ text = '   ' } | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $emptyBody | Out-Null
    Fail "Whitespace input unexpectedly accepted"
}
catch {
    Pass "Whitespace input rejected (expected behavior)"
}

try {
    $badBody = @{ wrong = 'field' } | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $badBody | Out-Null
    Fail "Invalid schema unexpectedly accepted"
}
catch {
    Pass "Invalid schema rejected (expected behavior)"
}

# 6) Dataset sample accuracy (random 50 rows)
try {
    if (Test-Path ".\IMDB Dataset.csv") {
        $rows = Import-Csv ".\IMDB Dataset.csv" | Get-Random -Count 50
        $correct = 0
        $count = 0
        $skipped = 0

        foreach ($r in $rows) {
            try {
                $reviewText = [string]$r.review
                if ([string]::IsNullOrWhiteSpace($reviewText)) {
                    $skipped++
                    continue
                }

                if ($reviewText.Length -gt 5000) {
                    $reviewText = $reviewText.Substring(0, 5000)
                }

                $body = @{ text = $reviewText } | ConvertTo-Json -Compress
                $pred = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
                if ($pred.sentiment -eq $r.sentiment) { $correct++ }
                $count++
            }
            catch {
                $skipped++
            }
        }

        $acc = if ($count -gt 0) { [math]::Round(($correct / $count) * 100, 2) } else { 0 }
        Write-Host "Sample accuracy (50 random reviews): $acc% ($correct/$count), skipped: $skipped" -ForegroundColor Yellow
        if ($count -lt 10) {
            Fail "Too many invalid/skipped rows in dataset batch check"
        } elseif ($acc -ge 75) {
            Pass "Sample accuracy is acceptable (>= 75%)"
        } else {
            Fail "Sample accuracy is low (< 75%)"
        }
    } else {
        Fail "IMDB Dataset.csv not found in repo root"
    }
}
catch {
    Fail "Dataset batch check failed: $($_.Exception.Message)"
}

# 7) Pytest smoke
try {
    docker compose exec fastapi-app pytest tests -q | Out-Null
    if ($LASTEXITCODE -eq 0) { Pass "Pytest suite passed" } else { Fail "Pytest suite failed" }
}
catch {
    Fail "Pytest execution failed: $($_.Exception.Message)"
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Passed: $passes"
Write-Host "Failed: $fails"

if ($fails -gt 0) {
    exit 1
}

exit 0
