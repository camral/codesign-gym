[CmdletBinding()]
param(
    [string]$Run = 'runs/mpm2_adam_to30_six_cycles',
    [string]$Gif,
    [string]$Python,
    [int]$Frames = 0,
    [int]$Width = 1280,
    [int]$Height = 800,
    [double]$Fps = 25,
    [switch]$Particles,
    [switch]$Elements,
    [switch]$NoSurface
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Python) {
    $documents = [Environment]::GetFolderPath('MyDocuments')
    $Python = Join-Path $documents 'ResearchCode/newton/.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Newton Python was not found at $Python. Pass -Python with the path to your Newton environment."
}
if (-not [IO.Path]::IsPathRooted($Run)) { $Run = Join-Path $projectRoot $Run }
$Run = (Resolve-Path -LiteralPath $Run).Path
$viewerArguments = @((Join-Path $PSScriptRoot 'view_newton.py'), '--run', $Run,
    '--width', "$Width", '--height', "$Height", '--fps', $Fps.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($Gif) {
    if (-not [IO.Path]::IsPathRooted($Gif)) { $Gif = Join-Path $projectRoot $Gif }
    $viewerArguments += @('--gif', $Gif)
}
if ($Frames -gt 0) { $viewerArguments += @('--max-frames', "$Frames") }
if ($Particles) { $viewerArguments += '--particles' }
if ($Elements) { $viewerArguments += '--elements' }
if ($NoSurface) { $viewerArguments += '--no-surface' }
if ($Frames -lt 0) { throw 'Frames must be zero (unlimited) or positive.' }
$cachePath = Join-Path $projectRoot '.cache/newton-warp'
New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
$previousCache = $env:WARP_CACHE_PATH
try {
    $env:WARP_CACHE_PATH = $cachePath
    & $Python @viewerArguments
    $viewerExitCode = $LASTEXITCODE
} finally {
    $env:WARP_CACHE_PATH = $previousCache
}
exit $viewerExitCode
