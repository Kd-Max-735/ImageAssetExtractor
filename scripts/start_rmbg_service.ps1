param(
    [string]$Python = 'D:\RMBG\RMBG\RMBG-2.0\.venv\Scripts\python.exe',
    [string]$Repository = 'D:\RMBG\RMBG\RMBG-2.0',
    [string]$ModelPath = 'D:\RMBG\RMBG\RMBG20',
    [int]$Port = 8001
)

$ErrorActionPreference = 'Stop'
$env:MODEL_PATH = $ModelPath
Set-Location -LiteralPath $Repository
& $Python -m uvicorn app:app --host 0.0.0.0 --port $Port
