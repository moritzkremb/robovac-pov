# render.ps1 - render.sh'in Windows karsiligi.
# build -> timeline -> metering on-gecisi -> segmentli render (kaldigi yerden devam) -> ses -> birlestirme
# kullanim: .\render.ps1 [out.mp4] [spp] [-W genislik] [-H yukseklik] [-Drone]
#   -Drone: goz hizasindaki drone dongusu (path_drone.h). Ses src/audio_drone.py ile uretilir.
param([string]$Out = "renders/v1.mp4", [int]$Spp = 36, [int]$W = 1920, [int]$H = 1080, [switch]$Drone)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
[Environment]::CurrentDirectory = $PSScriptRoot  # cmd/ffmpeg/python gibi harici programlar da ayni klasorden calissin
$SEG = 100
$SCENE = @(); $TAG = ""
if ($Drone) { $SCENE = @("--drone"); $TAG = "drone_" }

& cmd /c "`"$PSScriptRoot\build.bat`""
if ($LASTEXITCODE -ne 0) { throw "build failed" }
# segment klasoru sahne, cozunurluk ve ornek sayisina gore ayri: farkli ayarla yarim kalan segmentler karismasin
$SEGDIR = "work/seg_${TAG}${W}x${H}_${Spp}"
$METER = "work/${TAG}meter.txt"
$TIMELINE = "work/${TAG}timeline.json"
New-Item -ItemType Directory -Force $SEGDIR, renders | Out-Null
if (-not (Test-Path .venv/Scripts/python.exe)) {
  python -m venv .venv
  .venv/Scripts/python.exe -m pip install -q numpy scipy
}
& ./robovac.exe @SCENE --timeline $TIMELINE
if ($LASTEXITCODE -ne 0) { throw "timeline failed" }
$FRAMES = (Get-Content $TIMELINE -Raw | ConvertFrom-Json).frames
# 1. kameranin olcum gecisi (kucuk, hizli) -> gecikmeli otomatik pozlamayi besler
if (-not (Test-Path $METER) -or (Get-Item $METER).Length -eq 0) {
  & ./robovac.exe @SCENE --w 480 --h 270 --spp 8 --step 2 --meter $METER
  if ($LASTEXITCODE -ne 0) { throw "meter pass failed" }
}
# 2. film, segmentler halinde; yarida kalirsa kaldigi yerden devam eder
$i = 0
for ($a = 0; $a -lt $FRAMES; $a += $SEG) {
  $b = [Math]::Min($a + $SEG - 1, $FRAMES - 1)
  $f = "$SEGDIR/s_{0:D3}.mp4" -f $i
  if (-not (Test-Path $f) -or (Get-Item $f).Length -eq 0) {
    & ./robovac.exe @SCENE --w $W --h $H --spp $Spp --from $a --to $b --expo $METER --seg "$SEGDIR/tmp.mp4"
    if ($LASTEXITCODE -ne 0) { throw "render failed in segment $i" }
    Move-Item -Force "$SEGDIR/tmp.mp4" $f
  }
  $i++
}
Get-ChildItem "$SEGDIR/s_*.mp4" | Sort-Object Name | ForEach-Object { "file '" + ($_.FullName -replace '\\', '/') + "'" } | Set-Content -Encoding ascii "$SEGDIR/list.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$SEGDIR/list.txt" -c copy "work/${TAG}video.mp4"
if ($LASTEXITCODE -ne 0) { throw "concat failed" }
if ($Drone) {
  # drone: FPV motor sesi (dongu suresiyle periyodik) + son encode
  .venv/Scripts/python.exe src/audio_drone.py $TIMELINE work/drone_audio.wav
  if ($LASTEXITCODE -ne 0) { throw "drone audio failed" }
  ffmpeg -y -loglevel error -i "work/${TAG}video.mp4" -i work/drone_audio.wav -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p `
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 256k -movflags +faststart -shortest $Out
  if ($LASTEXITCODE -ne 0) { throw "encode failed" }
  Write-Host "wrote $Out"
  exit 0
}
# 3. ses, olay listesinden sentezlenir
.venv/Scripts/python.exe src/audio.py $TIMELINE work/audio.wav
if ($LASTEXITCODE -ne 0) { throw "audio failed" }
# 4. son encode + mux
ffmpeg -y -loglevel error -i work/video.mp4 -i work/audio.wav -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p `
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 256k -movflags +faststart -shortest $Out
if ($LASTEXITCODE -ne 0) { throw "mux failed" }
Write-Host "wrote $Out"
