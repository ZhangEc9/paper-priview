param(
    [string]$PythonExe = 'D:/Anaconda/envs/pythonproject/python.exe',
    [string]$IndexUrl = 'https://pypi.tuna.tsinghua.edu.cn/simple'
)
$ErrorActionPreference = 'Stop'
$TaskRoot = $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}

function Get-PinnedSource {
    param([string]$Directory, [string]$Url, [string]$Revision)
    if (-not (Test-Path -LiteralPath $Directory)) {
        New-Item -ItemType Directory -Path $Directory | Out-Null
        Invoke-Checked git @('-C', $Directory, 'init')
        Invoke-Checked git @('-C', $Directory, 'remote', 'add', 'origin', $Url)
        Invoke-Checked git @('-C', $Directory, 'fetch', '--depth', '1', 'origin', $Revision)
        Invoke-Checked git @('-C', $Directory, 'checkout', '--detach', 'FETCH_HEAD')
    }
    $actual = & git -C $Directory rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actual -ne $Revision) {
        throw "Existing checkout differs; left unchanged: $Directory"
    }
}

Get-PinnedSource (Join-Path $TaskRoot 'CKA-RL') 'https://github.com/Fhujinwu/CKA-RL.git' 'aa8817ace9f53120d5d205b9e4cafb3cdaa47803'
$VendorDir = Join-Path $TaskRoot 'vendor'
New-Item -ItemType Directory -Path $VendorDir -Force | Out-Null
Get-PinnedSource (Join-Path $VendorDir 'Metaworld') 'https://github.com/Farama-Foundation/Metaworld.git' 'c822f28f582ba1ad49eb5dcf61016566f28003ba'

$VenvDir = Join-Path $TaskRoot '.venv-cka'
$VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Invoke-Checked $PythonExe @('-m', 'venv', $VenvDir)
}
Invoke-Checked $VenvPython @('-m', 'pip', 'install', 'torch==2.1.2', '--index-url', 'https://download.pytorch.org/whl/cpu')
Invoke-Checked $VenvPython @('-m', 'pip', 'install', '-r', (Join-Path $TaskRoot 'requirements-cka-cpu-lock.txt'), '--index-url', $IndexUrl)
Invoke-Checked $VenvPython @('-m', 'pip', 'install', '--no-deps', '-e', (Join-Path $VendorDir 'Metaworld'), '--index-url', $IndexUrl)
Invoke-Checked $VenvPython @('-m', 'pip', 'check')
Invoke-Checked $VenvPython @((Join-Path $TaskRoot 'cka_cpu.py'), 'doctor')
