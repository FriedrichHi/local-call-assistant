<#
.SYNOPSIS
  Check a downloaded recorder installer against the digest GitHub publishes for it.

.DESCRIPTION
  Computes the file's SHA-256, looks up the matching asset in the named GitHub
  release (public API, no sign-in) and compares the two. Also reports the
  Windows Authenticode signature status. Read-only: it never runs the installer.

.EXAMPLE
  .\scripts\Test-RecorderInstaller.ps1 -Path $HOME\Downloads\meetily_0.5.0_x64-setup.exe `
      -Repository Zackriya-Solutions/meeting-minutes -Tag v0.5.0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Path,
    [Parameter(Mandatory)] [string] $Repository,
    [Parameter(Mandatory)] [string] $Tag
)
$ErrorActionPreference = 'Stop'

$file = Get-Item -LiteralPath $Path
$actual = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/releases/tags/$Tag" `
    -Headers @{ 'User-Agent' = 'local-call-assistant-installer-check' }
$asset = $release.assets | Where-Object { $_.name -eq $file.Name } | Select-Object -First 1
if (-not $asset) { throw "Release $Tag of $Repository has no asset named $($file.Name)." }

$expected = ($asset.digest -replace '^sha256:', '').ToLowerInvariant()
$signature = Get-AuthenticodeSignature -LiteralPath $file.FullName

[pscustomobject]@{
    File            = $file.Name
    SizeMatches     = ($file.Length -eq $asset.size)
    Sha256          = $actual
    PublishedSha256 = $expected
    DigestMatches   = ($expected -and $actual -eq $expected)
    PublishedAt     = $release.published_at
    Authenticode    = $signature.Status
    Signer          = $signature.SignerCertificate.Subject
} | Format-List

if (-not $expected) { Write-Warning 'GitHub publishes no digest for this asset; compare against the project''s own checksum file.' }
elseif ($actual -ne $expected) { Write-Error 'Digest mismatch: do not run this installer.' }
