[CmdletBinding()]
param(
  [string]$TensorBoardDir = "",
  [int]$Port = 8080,
  [string]$BindAddress = "0.0.0.0",
  [switch]$KeepOpenOnError,
  [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"

$exitCode = 0
try {
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
  if ([string]::IsNullOrWhiteSpace($TensorBoardDir)) {
    $TensorBoardDir = Join-Path $scriptRoot "tensorboard"
  }

  . (Join-Path $scriptRoot "payton_env.ps1")

  if (-not (Test-Path $TensorBoardDir)) {
    throw "TensorBoard log directory not found: $TensorBoardDir"
  }

  if (-not $PaytonPython) {
    throw "PaytonPython not set. Ensure payton_env.ps1 completed successfully."
  }

  Write-Host "Starting Image Review on http://localhost:$Port"
  Write-Host "TensorBoard directory: $TensorBoardDir"
  Write-Host "Press Ctrl+C to stop"
  Write-Host ""

  & $PaytonPython -m image_review --tensorboard-dir $TensorBoardDir --port $Port --host $BindAddress
  if ($LASTEXITCODE -ne 0) {
    throw "Image Review exited with code $LASTEXITCODE"
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
