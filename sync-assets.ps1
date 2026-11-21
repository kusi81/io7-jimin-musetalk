# Copies Kim Ji-min Gwazang's latest 1080p Hub clips into lipsync-worker/assets
# before running `docker build`.
$ErrorActionPreference = "Stop"

$srcDir = Join-Path $PSScriptRoot "..\public\assets\presenter"
$dstDir = Join-Path $PSScriptRoot "assets"

New-Item -ItemType Directory -Force -Path $dstDir | Out-Null

$clips = @(
  "explain_hands_v2.mp4",
  "greet_wave_v2.mp4",
  "show_pouch_sample_v2.mp4",
  "idle_listen_v2.mp4"
)

foreach ($clip in $clips) {
  $src = Join-Path $srcDir $clip
  if (Test-Path $src) {
    Copy-Item -Force $src (Join-Path $dstDir $clip)
    Write-Host "[OK] Copied $clip"
  } else {
    Write-Warning "[MISS] $clip not found in $srcDir"
  }
}

Write-Host "`nDone! Ready to run: docker build -t <your-dockerhub>/io7-jimin-musetalk:v1 ."
