# audio_drone.py — kucuk FPV drone sesi, ucus durumundan (drone timeline) sentezlenir.
# Ses dongu suresiyle birebir periyodiktir: butun frekanslar tur basina tam sayida devir yapar, gurultu ve filtreler
# FFT ile (dairesel) uretilir, yanki da dairesel uygulanir. Boylece son ornek ilk ornege dikissiz baglanir.
# kullanim: python audio_drone.py work/drone_timeline.json work/drone_audio.wav
import sys, json, wave
import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def cfilter(x, shape):
    """dairesel (periyodik) frekans alani filtresi; shape(f) -> kazanc"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    return np.fft.irfft(X * shape(f), len(x))


def pnoise(n, shape):
    """periyodik renkli gurultu: rastgele fazli spektrum"""
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X = shape(f) * np.exp(2j * np.pi * rng.random(len(f)))
    X[0] = 0
    x = np.fft.irfft(X, n)
    return x / (np.std(x) + 1e-12)


def band(lo, hi, order=2):
    def s(f):
        g = np.ones_like(f)
        if lo: g = g / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
        if hi: g = g / np.sqrt(1 + (f / hi) ** (2 * order))
        return g
    return s


def csmooth(x, sigma):
    """dairesel gauss yumusatma (sigma: ornek)"""
    n = len(x); k = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-0.5 * (2 * np.pi * k * sigma) ** 2), n)


def periodic_phase(freq, multiple):
    """frekans egrisini, toplam devir 'multiple'in tam kati olacak sekilde olcekler; faz (radyan) dondurur"""
    cycles = freq.sum() / SR
    target = max(multiple, round(cycles / multiple) * multiple)
    freq = freq * (target / cycles)
    return 2 * np.pi * np.cumsum(freq) / SR, freq


def main(tl_path, out_path):
    tl = json.load(open(tl_path))
    T = float(tl["loop"]); N = int(round(T * SR))
    st = np.array(tl["state"], dtype=np.float64); n = len(st); dt = T / n
    pos, roll = st[:, :3], st[:, 3]
    vel = (np.roll(pos, -1, 0) - np.roll(pos, 1, 0)) / (2 * dt)
    acc = (np.roll(vel, -1, 0) - np.roll(vel, 1, 0)) / (2 * dt)
    sp = np.linalg.norm(vel, axis=1)
    fwd = vel / np.maximum(sp[:, None], 1e-6)
    along = (acc * fwd).sum(1)
    lat = np.linalg.norm(acc - along[:, None] * fwd, axis=1)
    vy = vel[:, 1]
    # gaz: yanal ivme (donus), ileri ivme, tirmanis, yatma
    thr = 1.0 + 0.55 * lat + 0.9 * along + 2.5 * vy + 0.8 * np.abs(roll)
    thr = csmooth(np.clip(thr, 0.85, 1.6), 6.0)

    # 120 Hz -> ses hizi (dairesel dogrusal ara deger)
    ta = np.arange(N) / SR / dt; i0 = np.floor(ta).astype(np.int64); fr = ta - i0
    def up(x): return x[i0 % n] * (1 - fr) + x[(i0 + 1) % n] * fr
    thr_a, roll_a, sp_a = up(thr), up(csmooth(roll, 6.0)), up(sp)

    # yavas, periyodik devir dalgalanmasi (kontrolcunun surekli duzeltmesi): tur basina tam sayida devir
    t = np.arange(N) / SR; w = 2 * np.pi / T
    def wob(seed):
        r = np.random.default_rng(seed); a = np.zeros(N)
        for h in (41, 67, 89, 131, 173, 227): a += r.uniform(0.3, 1.0) * np.sin(w * h * t + r.uniform(0, 2 * np.pi)) / np.sqrt(h / 41)
        return a / 3.0

    # 4 rotor: on-sol, on-sag, arka-sol, arka-sag. 3 kanatli pervane, havada ~16k RPM -> kanat gecis ~800 Hz
    BPF = 800.0
    rotors = [(-1, 1.000, 0.95), (1, 1.014, 0.80), (-1, 0.989, 0.75), (1, 1.023, 0.95)]  # (taraf, ayar, kazanc)
    L = np.zeros(N); R = np.zeros(N)
    for i, (side, detune, gain) in enumerate(rotors):
        rpm = np.sqrt(thr_a) * detune * (1.0 + 0.012 * wob(100 + i)) * (1.0 + side * 0.55 * roll_a)
        # faz: tur basina devir sayisi 3'un kati -> mil frekansi (BPF/3) ve motor sesi (BPF*7/3) de periyodik
        ph, f_inst = periodic_phase(BPF * rpm, 3)
        r = np.random.default_rng(200 + i)
        tone = np.zeros(N)
        for k in range(1, 15):  # kanat gecis harmonikleri: keskin, tiz vizilti
            tone += (1.0 / k ** 0.9) * r.uniform(0.6, 1.0) * np.sin(k * ph + r.uniform(0, 2 * np.pi))
        tone += 0.35 * np.sin(ph / 3 + r.uniform(0, 6.28))            # mil (hafif dengesizlik)
        tone += 0.06 * np.sin(ph * 7 / 3 + r.uniform(0, 6.28))        # motor/ESC elektrik vizlamasi
        tone += 0.04 * np.sin(ph * 14 / 3 + r.uniform(0, 6.28))
        # pervane hava gurultusu, her kanat gecisinde "doğranir"
        air = pnoise(N, band(400, 9000, 1)) * (1.0 + 0.6 * np.sin(ph + 0.4)) * 0.55
        s = (tone * 0.22 + air * 0.30) * gain * (0.85 + 0.15 * thr_a)
        pl = 0.72 if side < 0 else 0.42
        L += s * pl; R += s * (1.14 - pl)

    # hiza bagli hafif ruzgar ugultusu (mikrofon drone uzerinde)
    wind = pnoise(N, band(40, 320, 2)) * 0.10 * (sp_a / max(sp_a.max(), 1e-6))
    L += wind; R += cfilter(np.roll(wind, 911), lambda f: np.ones_like(f))

    # kucuk kamera mikrofonu: 120 Hz alti ve 11 kHz ustu zayif
    mic = band(120, 11000, 2)
    L = cfilter(L, mic); R = cfilter(R, mic)

    # oda yankisi: kisa, dairesel konvolusyon (dongu dikisinde kuyruk basa sarar)
    def ir(seed):
        r = np.random.default_rng(seed); m = int(0.6 * SR); tt = np.arange(m) / SR
        h = r.standard_normal(m) * np.exp(-tt / 0.085)
        h = cfilter(h, band(200, 5000, 1)); h[:int(0.004 * SR)] = 0
        for d, a in ((0.007, 0.5), (0.011, 0.35), (0.017, 0.3), (0.023, 0.2)): h[int(d * SR)] += a * (1 if r.random() < 0.5 else -1)
        out = np.zeros(N); out[:m] = h; return out / np.sqrt((h ** 2).sum())
    def conv(x, h): return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h), N)
    wet = 0.16
    L, R = L + wet * conv(L, ir(1)) * np.std(L) / (np.std(conv(L, ir(1))) + 1e-12), R + wet * conv(R, ir(2)) * np.std(R) / (np.std(conv(R, ir(2))) + 1e-12)

    # seviye: RMS ~ -20 dBFS, tepe en fazla -1 dBFS
    x = np.stack([L, R], 1)
    x *= 0.1 / (np.sqrt((x ** 2).mean()) + 1e-12)
    pk = np.abs(x).max()
    if pk > 0.89: x *= 0.89 / pk
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"wrote {out_path}: {N / SR:.3f} s, {N} samples/channel, rms {20 * np.log10(np.sqrt((x ** 2).mean())):.1f} dBFS, peak {20 * np.log10(np.abs(x).max()):.1f} dBFS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
