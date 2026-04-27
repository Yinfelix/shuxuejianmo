param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath
)

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$octaveSearchRoot = Join-Path $workspaceRoot 'tools\octave'

$octaveCli = Get-ChildItem -Path $octaveSearchRoot -Recurse -Filter 'octave-cli.exe' -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $octaveCli) {
    Write-Error 'Octave runtime not found under tools\octave. Install or extract Octave first.'
    exit 1
}

$resolvedScript = if ([System.IO.Path]::IsPathRooted($ScriptPath)) {
    $ScriptPath
} else {
    Join-Path (Get-Location) $ScriptPath
}

if (-not (Test-Path $resolvedScript)) {
    Write-Error "Script not found: $resolvedScript"
    exit 1
}

$scriptDirectory = Split-Path -Parent $resolvedScript
$scriptName = Split-Path -Leaf $resolvedScript

Push-Location $scriptDirectory
try {
    & $octaveCli '--quiet' '--eval' "run('$scriptName')"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}