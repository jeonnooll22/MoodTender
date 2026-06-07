param(
  [switch]$Restart
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root "LivePortrait\venv\Scripts\python.exe"
$App = Join-Path $Root "backend\main.py"
$EnvFile = Join-Path $Root ".env"
$Port = 7862

if (-not (Test-Path $Python)) {
  throw "Python venv not found: $Python"
}

if (-not (Test-Path $App)) {
  throw "Backend entrypoint not found: $App"
}

if (Test-Path $EnvFile) {
  Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
      return
    }

    $parts = $line.Split("=", 2)
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($name) {
      Set-Item -Path "Env:$name" -Value $value
    }
  }
}

$listeners = @(Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" })
if ($listeners.Count -gt 0) {
  $pids = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
  if ($Restart) {
    foreach ($processId in $pids) {
      Stop-Process -Id $processId -Force
    }
    Start-Sleep -Seconds 1
  } else {
    Write-Host "MoodTender is already running at http://127.0.0.1:$Port/"
    Write-Host "Use '.\run.ps1 -Restart' to stop the existing server and start a new one."
    exit 0
  }
}

if (-not $env:OPENAI_API_KEY) {
  Write-Warning "OPENAI_API_KEY is not set. The page will open, but LLM replies will fail until you set it."
  Write-Host "Create a .env file with: OPENAI_API_KEY=sk-..."
}

Write-Host "Starting MoodTender at http://127.0.0.1:$Port/"
Set-Location (Join-Path $Root "backend")
& $Python $App
