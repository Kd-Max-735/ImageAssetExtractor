param(
    [string]$Python = 'D:\sam3.1\miniconda3\envs\sam3\python.exe',
    [string]$Repository = 'D:\sam3.1\sam3-filtered',
    [string]$Checkpoint = 'D:\sam3.1\models\sam3.1_multiplex.pt',
    [int]$Port = 8002
)

$ErrorActionPreference = 'Stop'
$env:SAM31_CHECKPOINT = $Checkpoint
$env:SAM31_WORK_ROOT = 'D:\ImageAssetExtractor\output\sam31-work'
Set-Location -LiteralPath $Repository
& $Python -m uvicorn --app-dir 'D:\ImageAssetExtractor\scripts' sam31_service:app --host 0.0.0.0 --port $Port
