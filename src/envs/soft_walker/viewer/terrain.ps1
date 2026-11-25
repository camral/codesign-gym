[CmdletBinding()]
param(
    [string]$Preset = 'stairs',
    [ValidateSet(2,3)][int]$Dim = 2,
    [int]$Seed = 7,
    [string]$Out = 'runs/terrain_demo/custom',
    [string]$Gallery,
    [string]$Python,
    [switch]$Save,
    [switch]$Hidden,
    [int]$Frames = 0,
    [int]$Width = 1280,
    [int]$Height = 800
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
if ($Frames -lt 0) { throw 'Frames must be zero (unlimited) or positive.' }
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $projectRoot $Out }
$viewerArguments = @((Join-Path $PSScriptRoot 'terrain_editor.py'), '--preset', $Preset,
    '--dim', "$Dim", '--seed', "$Seed", '--out', $Out, '--width', "$Width", '--height', "$Height")
if ($Gallery) {
    if (-not [IO.Path]::IsPathRooted($Gallery)) { $Gallery = Join-Path $projectRoot $Gallery }
    $viewerArguments += @('--gallery', $Gallery)
}
if ($Frames -gt 0) { $viewerArguments += @('--max-frames', "$Frames") }
if ($Save) { $viewerArguments += '--save-current' }
if ($Hidden) { $viewerArguments += '--headless' }
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
