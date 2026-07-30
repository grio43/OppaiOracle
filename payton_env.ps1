[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$VenvPath = "L:\\Dab\\payton_env",
  [string]$PythonVersion = "3.11",
  [string]$PythonExe = "",
  [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
  $ProjectRoot = (Resolve-Path $scriptRoot).Path
}

$pythonExe = Join-Path $VenvPath "Scripts\\python.exe"

if (-not (Test-Path $pythonExe)) {
  Write-Host "Creating Payton venv at $VenvPath"
  if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    & $PythonExe -m venv $VenvPath
  } else {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
      & $pyLauncher.Source "-$PythonVersion" -m venv $VenvPath
    } else {
      & python -m venv $VenvPath
    }
  }
  if (-not (Test-Path $pythonExe)) {
    throw "Venv creation failed. Expected $pythonExe"
  }
}

$venvVersion = (& $pythonExe -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null).Trim()
if ($venvVersion -ne $PythonVersion) {
  throw "Venv uses Python $venvVersion; expected $PythonVersion. Pass -PythonExe or adjust -PythonVersion."
}

$env:OPPAI_ORACLE_ROOT = $ProjectRoot

if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
  $env:PYTHONPATH = $ProjectRoot
} elseif (-not ($env:PYTHONPATH.Split(';') -contains $ProjectRoot)) {
  $env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
}

$env:VIRTUAL_ENV = $VenvPath
$venvScripts = Join-Path $VenvPath "Scripts"
if ([string]::IsNullOrWhiteSpace($env:Path)) {
  $env:Path = $venvScripts
} elseif (-not ($env:Path.Split(';') -contains $venvScripts)) {
  $env:Path = "$venvScripts;$env:Path"
}

Set-Location $ProjectRoot

$PaytonPython = $pythonExe

if ($InstallDeps) {
  & $PaytonPython -m pip install --upgrade pip
  & $PaytonPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}
