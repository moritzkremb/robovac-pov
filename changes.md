# Changes

## 2026-11-24 — Windows / CUDA port (branch `port/cuda-windows`)

Sebep: Orijinal kod sadece macOS + Metal uzerinde derleniyor. Video'yu Windows'ta RTX 3090 ile uretebilmek icin
render tarafi CUDA'ya port edildi. Mac dosyalari (`main.mm`, `render.metal`, `build.sh`, `render.sh`) degistirilmedi.

### Degisen dosya

**`src/geom.h`** (yedek: `src/20261124_024843_geom.h.bck`)
- Oncesi: `#include <dispatch/dispatch.h>` kosulsuz; `sdfMesh` icinde `dispatch_apply(..., ^(size_t k) {...})`
  (Apple GCD + blocks, MSVC derleyemez).
- Sonrasi: include `#ifndef _WIN32` altinda. `sdfMesh` dongusu `_WIN32` icin `#pragma omp parallel for` ile ayni
  isi yapiyor; Mac tarafinda `dispatch_apply` aynen duruyor.

### Yeni dosyalar

- `src/gpu_types.h` — host ve GPU'nun ortak kullandigi `Uni`, `Inst`, `TriA` yapilari (Metal'deki ile ayni duzen).
- `src/gpu_api.h` — `main_cuda.cpp`'nin `render.cu`'yu cagirdigi fonksiyonlar.
- `src/cuda_math.h` — Metal vektor fonksiyonlarinin (mix, smoothstep, saturate, fract, reflect, ...) CUDA karsiliklari.
- `src/bvh_build.h` — CPU'da binned SAH BVH kurulumu (Metal'deki donanim acceleration structure'in yerine).
- `src/render.cu` — `render.metal`'in CUDA cevirisi (render, bake, dnPrep, dnAtrous) + BVH traversal.
- `src/main_cuda.cpp` — `main.mm`'in Windows karsiligi (ayni komut satiri, ayni post-process).
- `build.bat` — nvcc + MSVC ile `robovac.exe` derler.
- `render.ps1` — `render.sh`'in Windows karsiligi.

## 2026-11-24 — Drone dongu sahnesi (ayri sahne, robot videosu degismedi)

Sebep: Ayni odada goz hizasinda (~1.5 m) dolasan, basi ve sonu dikissiz birlesen 20 sn'lik drone videosu istendi.

- **Yeni `src/path_drone.h`**: kapali Catmull-Rom rota (11 kontrol noktasi, 12.9 m, 1.45-1.55 m yukseklik). Hiz,
  yukseklik, donuslerde yatma ve titresim dongu suresiyle (20 sn) periyodik; kare 600 == kare 0.
- **`src/main_cuda.cpp`** (yedek: `src/20261124_034230_main_cuda.cpp.bck`)
  - Oncesi: sadece robot sahnesi.
  - Sonrasi: `--drone` secenegi. Drone modunda: 600 kare; kamera `droneRigAt`; robot dock'ta, dinozor ayakta (sabit);
    kadrajda govde (shell) yok; fade yok; odak sabit 1.6 m; gecikmeli pozlama iki tur calistirilip ikinci tur
    kullaniliyor (dongu dikisinde pozlama sicramasin); `--timeline` sadece kare sayisini yaziyor; `--collide` 3D kure
    kontrolu yapiyor. `--drone` verilmezse robot sahnesi eskisi gibi.
- **`render.ps1`** (yedek: `20261124_034230_render.ps1.bck`)
  - Oncesi: sadece robot sahnesi.
  - Sonrasi: `-Drone` anahtari: `--drone` gecer, ayri olcum/timeline/segment dosyalari (`work/drone_*`), ses adimi yok.

## 2026-11-24 — Drone sesi (kucuk FPV)

Sebep: Drone videosu sessizdi; kucuk FPV drone motor sesi istendi.

- **`src/main_cuda.cpp`** (yedek: `src/20261124_035811_main_cuda.cpp.bck`)
  - Oncesi: `--drone --timeline` sadece kare sayisini yaziyordu (`"state": []`).
  - Sonrasi: bir tur boyunca 120 Hz ucus durumu da yaziliyor: `[x, y, z, yatma]` ve `"loop"` (tur suresi).
- **Yeni `src/audio_drone.py`**: 4 rotor, 3 kanatli pervane (~800 Hz kanat gecisi + harmonikler), mil ve motor
  vizlamasi, pervane hava gurultusu, hiza bagli ruzgar, kucuk mikrofon EQ'su, kisa oda yankisi. Gaz; donus, ivme,
  tirmanis ve yatmadan geliyor. Her sey dairesel/periyodik uretildigi icin sesin sonu basina dikissiz baglaniyor.
  Sadece numpy kullaniyor.
- **`render.ps1`** (yedek: `20261124_035811_render.ps1.bck`)
  - Oncesi: `-Drone` modunda ses yoktu.
  - Sonrasi: `-Drone` modunda `audio_drone.py` calisip ses videoya ekleniyor (AAC 256k).

## 2026-11-24 — render.ps1 cozunurluk parametresi

Sebep: 960x540 / 16 ornek render istendi; betik sadece 1920x1080 render ediyordu.
Dosya: `render.ps1` (yedek: `20261124_031557_render.ps1.bck`)
- Oncesi: `param([string]$Out, [int]$Spp)`; segmentler `work/seg/` altinda; `robovac.exe`'ye cozunurluk verilmiyordu.
- Sonrasi: `-W` / `-H` parametreleri (varsayilan 1920 / 1080) `--w` / `--h` olarak geciliyor. Segment klasoru
  `work/seg_{W}x{H}_{spp}/` oldu, boylece farkli ayarlarla yarim kalan segmentler birbirine karismiyor.
