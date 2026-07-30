[CmdletBinding()]
param(
  [string]$LogDir = "",
  [int]$Port = 6006,
  [switch]$KeepOpenOnError,
  [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"

$exitCode = 0
try {
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
  if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $scriptRoot "tensorboard"
  }

  . (Join-Path $scriptRoot "payton_env.ps1")

  if (-not (Test-Path $LogDir)) {
    throw "TensorBoard log directory not found: $LogDir"
  }

  if (-not $PaytonPython) {
    throw "PaytonPython not set. Ensure payton_env.ps1 completed successfully."
  }

  Write-Host "Starting TensorBoard on http://localhost:$Port"
  Write-Host "Log directory: $LogDir"
  Write-Host "Press Ctrl+C to stop"
  Write-Host ""

  & $PaytonPython -m tensorboard.main --logdir $LogDir --port $Port --bind_all
  if ($LASTEXITCODE -ne 0) {
    throw "TensorBoard exited with code $LASTEXITCODE"
  }
} catch {
  Write-Error $_
  $exitCode = 1
}

if ($KeepOpen -or ($KeepOpenOnError -and $exitCode -ne 0)) {
  [void](Read-Host "Press Enter to close")
}

if (-not ($KeepOpen -or $KeepOpenOnError)) {
  exit $exitCode
}

$global:LASTEXITCODE = $exitCode
return
