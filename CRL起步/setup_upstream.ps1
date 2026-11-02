$ErrorActionPreference = "Stop"
$upstreamUrl = "https://github.com/Fhujinwu/CKA-RL.git"
$upstreamCommit = "aa8817ace9f53120d5d205b9e4cafb3cdaa47803"
$destination = Join-Path $PSScriptRoot "CKA-RL"

if (Test-Path -LiteralPath $destination) {
    $head = git -C $destination rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head -ne $upstreamCommit) {
        throw "Existing CKA-RL directory differs from the pinned version. Inspect it before changing it."
    }
    Write-Host "CKA-RL is already at $upstreamCommit"
    exit 0
}

git clone --no-checkout $upstreamUrl $destination
if ($LASTEXITCODE -ne 0) { throw "Failed to clone CKA-RL." }
git -C $destination checkout --detach $upstreamCommit
if ($LASTEXITCODE -ne 0) { throw "Failed to check out the pinned CKA-RL commit." }
Write-Host "CKA-RL ready at $destination"
