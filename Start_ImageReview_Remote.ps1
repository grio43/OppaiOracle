[CmdletBinding()]
param(
  [string]$TensorBoardDir = "",
  [int]$Port = 8080,
  [switch]$KeepOpenOnError,
  [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
if ([string]::IsNullOrWhiteSpace($TensorBoardDir)) {
  $TensorBoardDir = Join-Path $scriptRoot "tensorboard"
}

# Check for cloudflared
$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared) {
  # Try common install locations
  $commonPaths = @(
    "C:\Program Files (x86)\cloudflared\cloudflared.exe",
    "C:\Program Files\cloudflared\cloudflared.exe",
    "$env:LOCALAPPDATA\cloudflared\cloudflared.exe"
  )
  foreach ($path in $commonPaths) {
    if (Test-Path $path) {
      $cloudflared = @{ Source = $path }
      break
    }
  }
}
if (-not $cloudflared) {
  Write-Host ""
  Write-Host "cloudflared is not installed." -ForegroundColor Red
  Write-Host ""
  Write-Host "Install it using one of these methods:" -ForegroundColor Yellow
  Write-Host ""
  Write-Host "  Option 1 - winget (recommended):" -ForegroundColor Cyan
  Write-Host "    winget install Cloudflare.cloudflared"
  Write-Host ""
  Write-Host "  Option 2 - Direct download:" -ForegroundColor Cyan
  Write-Host "    https://github.com/cloudflare/cloudflared/releases/latest"
  Write-Host "    Download cloudflared-windows-amd64.exe, rename to cloudflared.exe,"
  Write-Host "    and add to your PATH."
  Write-Host ""
  if ($KeepOpen -or $KeepOpenOnError) {
    [void](Read-Host "Press Enter to close")
  }
  exit 1
}

. (Join-Path $scriptRoot "payton_env.ps1")

if (-not (Test-Path $TensorBoardDir)) {
  throw "TensorBoard log directory not found: $TensorBoardDir"
}

if (-not $PaytonPython) {
  throw "PaytonPython not set. Ensure payton_env.ps1 completed successfully."
}

$serverJob = $null
$tunnelProcess = $null
$exitCode = 0

try {
  Write-Host ""
  Write-Host "Starting Image Review with Cloudflare Tunnel..." -ForegroundColor Cyan
  Write-Host "TensorBoard directory: $TensorBoardDir" -ForegroundColor DarkGray
  Write-Host ""

  # Start the image review server in background
  Write-Host "Starting local server on port $Port..." -ForegroundColor Yellow
  $serverJob = Start-Job -ScriptBlock {
    param($python, $tbDir, $port)
    & $python -m image_review --tensorboard-dir $tbDir --port $port --host "127.0.0.1"
  } -ArgumentList $PaytonPython, $TensorBoardDir, $Port

  # Give server time to start
  Start-Sleep -Seconds 2

  # Check if server started
  if ($serverJob.State -eq 'Failed') {
    throw "Failed to start image review server"
  }

  # Start cloudflared tunnel
  Write-Host "Starting Cloudflare Quick Tunnel..." -ForegroundColor Yellow
  Write-Host ""
  Write-Host "=" * 60 -ForegroundColor DarkGray
  Write-Host ""

  $cloudflaredPath = if ($cloudflared.Source) { $cloudflared.Source } else { "cloudflared" }
  $tunnelProcess = Start-Process -FilePath $cloudflaredPath `
    -ArgumentList "tunnel", "--url", "http://localhost:$Port" `
    -NoNewWindow -PassThru

  Write-Host "Tunnel is starting. Look for the URL above (*.trycloudflare.com)" -ForegroundColor Green
  Write-Host ""
  Write-Host "Copy the URL and open it on your phone to review images!" -ForegroundColor Cyan
  Write-Host ""
  Write-Host "Press Ctrl+C to stop the server and tunnel." -ForegroundColor DarkGray
  Write-Host ""

  # Wait for tunnel process or user interrupt
  while (-not $tunnelProcess.HasExited) {
    Start-Sleep -Seconds 1

    # Check if server job failed
    if ($serverJob.State -eq 'Failed' -or $serverJob.State -eq 'Stopped') {
      Write-Host "Server stopped unexpectedly" -ForegroundColor Red
      Receive-Job $serverJob -ErrorAction SilentlyContinue
      break
    }
  }

} catch {
  Write-Error $_
  $exitCode = 1
} finally {
  Write-Host ""
  Write-Host "Shutting down..." -ForegroundColor Yellow

  # Stop tunnel
  if ($tunnelProcess -and -not $tunnelProcess.HasExited) {
    Stop-Process -Id $tunnelProcess.Id -Force -ErrorAction SilentlyContinue
  }

  # Stop server job
  if ($serverJob) {
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -Force -ErrorAction SilentlyContinue
  }

  Write-Host "Cleanup complete." -ForegroundColor Green
}

if ($KeepOpen -or ($KeepOpenOnError -and $exitCode -ne 0)) {
  [void](Read-Host "Press Enter to close")
}

exit $exitCode
