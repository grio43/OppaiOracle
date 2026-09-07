[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$VenvPath = "L:\\Dab\\payton_env",
  # Existing environments keep their interpreter; new environments default to 3.12.
  [string]$PythonVersion = "",
  [string]$PythonExe = "",
  [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
  $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }
  $ProjectRoot = (Resolve-Path $scriptRoot).Path
}

$venvPython = Join-Path $VenvPath "Scripts\\python.exe"

if (-not (Test-Path $venvPython)) {
  Write-Host "Creating Payton venv at $VenvPath"
  $creationVersion = if ($PythonVersion) { $PythonVersion } else { "3.12" }
  if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    & $PythonExe -m venv $VenvPath
  } else {
    # Enforce the default even when falling back to python without the launcher.
    $PythonVersion = $creationVersion
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
      & $pyLauncher.Source "-$creationVersion" -m venv $VenvPath
    } else {
      & python -m venv $VenvPath
    }
  }
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
    throw "Venv creation failed. Expected $venvPython"
  }
}

$venvVersion = (& $venvPython -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null).Trim()
if ($LASTEXITCODE -ne 0) {
  throw "Could not run $venvPython"
}
if ($PythonVersion -and $venvVersion -ne $PythonVersion) {
  throw "Venv uses Python $venvVersion; expected $PythonVersion. Select a different -VenvPath or adjust -PythonVersion."
}
if ([version]$venvVersion -lt [version]"3.11") {
  throw "The training dependencies require Python 3.11 or newer. Use Python 3.12 for a new environment."
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

$PaytonPython = $venvPython

if ($InstallDeps) {
  & $PaytonPython -m pip install --upgrade pip
  if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
  & $PaytonPython -m pip install -r (Join-Path $ProjectRoot "requirements-training-cu130.txt")
  if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
  & $PaytonPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
  if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed" }
  & $PaytonPython -m pip check
  if ($LASTEXITCODE -ne 0) { throw "Dependency conflicts remain; resolve the pip check errors before training" }
}
