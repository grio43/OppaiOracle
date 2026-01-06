[CmdletBinding()]
param(
  [string]$ConfigPath = "",
  [switch]$KeepOpenOnError,
  [switch]$KeepOpen,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$TrainingArgs
)

$ErrorActionPreference = "Stop"

$exitCode = 0
try {
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
  if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $scriptRoot "configs\\unified_config.yaml"
  }

  . (Join-Path $scriptRoot "payton_env.ps1")

  if (-not (Test-Path $ConfigPath)) {
    throw "Config not found: $ConfigPath"
  }

  $trainScript = Join-Path $scriptRoot "train_direct.py"
  if (-not (Test-Path $trainScript)) {
    throw "Training script not found: $trainScript"
  }

  if (-not $PaytonPython) {
    throw "PaytonPython not set. Ensure payton_env.ps1 completed successfully."
  }

  & $PaytonPython $trainScript --config $ConfigPath @TrainingArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Training exited with code $LASTEXITCODE"
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
