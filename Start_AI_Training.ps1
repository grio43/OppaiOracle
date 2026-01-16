[CmdletBinding()]
param(
  [string]$ConfigPath = "",
  [switch]$KeepOpenOnError,
  [switch]$KeepOpen,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$TrainingArgs
)

$ErrorActionPreference = "Stop"

# Set up Visual Studio Build Tools environment for torch.compile
# Check multiple possible VS installation locations
$vsBuildToolsBase = $null
$vsSearchPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools",   # Non-standard VS 2022 location
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
)
foreach ($path in $vsSearchPaths) {
    if (Test-Path $path) {
        $vsBuildToolsBase = $path
        break
    }
}

if ($vsBuildToolsBase -and (Test-Path $vsBuildToolsBase)) {
    $msvcPath = Join-Path $vsBuildToolsBase "VC\Tools\MSVC"
    if (Test-Path $msvcPath) {
        $msvcVersion = (Get-ChildItem $msvcPath | Sort-Object Name -Descending | Select-Object -First 1).Name
        $msvcVersionPath = Join-Path $msvcPath $msvcVersion

        # Add cl.exe to PATH
        $clPath = Join-Path $msvcVersionPath "bin\Hostx64\x64"
        if (Test-Path $clPath) {
            $env:PATH = "$clPath;$env:PATH"
        }

        # Add include path for omp.h and other headers
        $includePath = Join-Path $msvcVersionPath "include"
        if (Test-Path $includePath) {
            $env:INCLUDE = "$includePath;$env:INCLUDE"
        }

        # Add lib path for linking
        $libPath = Join-Path $msvcVersionPath "lib\x64"
        if (Test-Path $libPath) {
            $env:LIB = "$libPath;$env:LIB"
        }
    }
}

# Add Windows SDK paths (required for crtdbg.h, windows.h, etc.)
$winSdkBase = "C:\Program Files (x86)\Windows Kits\10"
if (Test-Path $winSdkBase) {
    $sdkIncludePath = Join-Path $winSdkBase "Include"
    if (Test-Path $sdkIncludePath) {
        $sdkVersion = (Get-ChildItem $sdkIncludePath | Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } | Sort-Object Name -Descending | Select-Object -First 1).Name
        if ($sdkVersion) {
            $sdkVersionPath = Join-Path $sdkIncludePath $sdkVersion
            # Add UCRT, shared, and um include paths
            @("ucrt", "shared", "um") | ForEach-Object {
                $p = Join-Path $sdkVersionPath $_
                if (Test-Path $p) { $env:INCLUDE = "$p;$env:INCLUDE" }
            }
            # Add UCRT and um lib paths
            $sdkLibPath = Join-Path $winSdkBase "Lib\$sdkVersion"
            @("ucrt\x64", "um\x64") | ForEach-Object {
                $p = Join-Path $sdkLibPath $_
                if (Test-Path $p) { $env:LIB = "$p;$env:LIB" }
            }
        }
    }
}

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
