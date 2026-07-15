param([switch]$Build)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DockerBin = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
$DockerDesktop = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'

if (-not (Test-Path -LiteralPath $DockerBin)) {
    throw "Docker CLI not found: $DockerBin"
}
if (-not (& $DockerBin info 2>$null)) {
    Start-Process -FilePath $DockerDesktop -WindowStyle Hidden
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        if (& $DockerBin info 2>$null) { break }
    }
    if (-not (& $DockerBin info 2>$null)) { throw 'Docker Desktop did not become ready' }
}

function Test-Http([string]$Uri) {
    try {
        $null = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Http 'http://127.0.0.1:8001/docs')) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $PSScriptRoot 'start_rmbg_service.ps1')
    ) -WindowStyle Hidden
}
if (-not (Test-Http 'http://127.0.0.1:8002/health')) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $PSScriptRoot 'start_sam31_service.ps1')
    ) -WindowStyle Hidden
}

for ($i = 0; $i -lt 90; $i++) {
    if ((Test-Http 'http://127.0.0.1:8001/docs') -and (Test-Http 'http://127.0.0.1:8002/health')) { break }
    Start-Sleep -Seconds 2
}
if (-not (Test-Http 'http://127.0.0.1:8001/docs')) { throw 'RMBG service did not become ready' }
if (-not (Test-Http 'http://127.0.0.1:8002/health')) { throw 'SAM 3.1 service did not become ready' }

Push-Location $ProjectRoot
try {
    if ($Build -or -not (& $DockerBin image inspect 'image-asset-extractor:0.2.0' 2>$null)) {
        & $DockerBin compose build api
        if ($LASTEXITCODE -ne 0) { throw 'API image build failed' }
    }
    & $DockerBin compose up -d api
    if ($LASTEXITCODE -ne 0) { throw 'API container startup failed' }
} finally {
    Pop-Location
}

for ($i = 0; $i -lt 30; $i++) {
    if (Test-Http 'http://127.0.0.1:8000/health') { break }
    Start-Sleep -Seconds 1
}
if (-not (Test-Http 'http://127.0.0.1:8000/health')) { throw 'Project API did not become ready' }

$capabilities = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/capabilities' -TimeoutSec 5
[pscustomobject]@{
    API = 'http://127.0.0.1:8000'
    OpenCV = $capabilities.opencv.available
    RMBG = $capabilities.rmbg.available
    SAM31 = $capabilities.sam31.available
} | Format-List
