// path_drone.h — goz hizasinda (~1.5 m) odayi dolasan drone. Rota kapali bir egri, butun hareketler (hiz, yukseklik,
// yatma, titresim) dongu suresiyle periyodik; boylece son kareden ilk kareye gecis dikissiz.
#pragma once
#include "path.h"

static const float DRONE_T = 20.0f;   // bir tur (s)
static const float DRONE_R = 0.18f;   // carpisma kontrolu icin drone yaricapi (m)

struct DronePath {
  std::vector<V3> pts;      // yogun ornekler (kapali)
  std::vector<float> cum;   // kumulatif uzunluk
  float L = 0;

  DronePath() {
    // kontrol noktalari (x, y, z): pencerelerin onu -> dogu -> kitaplik onu -> koltuk onu -> bati -> basa don
    const V3 c[] = {
      V3(0.85f, 1.50f, 3.45f), V3(2.00f, 1.55f, 3.40f), V3(3.30f, 1.48f, 3.45f), V3(4.50f, 1.52f, 3.55f),
      V3(5.35f, 1.50f, 3.00f), V3(5.45f, 1.45f, 1.85f), V3(4.95f, 1.50f, 0.95f), V3(3.90f, 1.55f, 0.85f),
      V3(2.60f, 1.50f, 1.40f), V3(1.35f, 1.45f, 1.55f), V3(0.70f, 1.50f, 2.45f)};
    const int n = (int)(sizeof(c) / sizeof(c[0])); const int per = 200;
    for (int i = 0; i < n; i++) { // kapali Catmull-Rom
      V3 p0 = c[(i + n - 1) % n], p1 = c[i], p2 = c[(i + 1) % n], p3 = c[(i + 2) % n];
      for (int k = 0; k < per; k++) { float t = (float)k / per, t2 = t * t, t3 = t2 * t;
        pts.push_back(0.5f * ((2.0f * p1) + (p2 - p0) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (3.0f * p1 - p0 - 3.0f * p2 + p3) * t3)); }
    }
    cum.resize(pts.size() + 1); cum[0] = 0;
    for (size_t i = 0; i < pts.size(); i++) cum[i + 1] = cum[i] + len(pts[(i + 1) % pts.size()] - pts[i]);
    L = cum.back();
  }
  // yay uzunlugu s'deki nokta (s mod L)
  V3 at(float s) const {
    s = fmodf(s, L); if (s < 0) s += L;
    size_t lo = std::upper_bound(cum.begin(), cum.end(), s) - cum.begin(); if (lo > 0) lo--; if (lo >= pts.size()) lo = pts.size() - 1;
    float seg = cum[lo + 1] - cum[lo]; float u = seg > 1e-9f ? (s - cum[lo]) / seg : 0;
    return lerp(pts[lo], pts[(lo + 1) % pts.size()], u);
  }
};

// t'deki yay uzunlugu: sabit ortalama hiz + hafif periyodik hiz degisimi (tam tur basina 2 yavaslama)
static inline float droneS(const DronePath& P, float t) {
  float w = 2 * PI / DRONE_T;
  return P.L * (t / DRONE_T) + P.L * 0.018f * sinf(2 * w * t);
}
// periyodik kucuk titresim: frekanslar turun tam katlari
static inline float pnoise(float t, int ch) {
  float w = 2 * PI / DRONE_T; float a = 0;
  const int h[4] = {37, 53, 71, 97};
  for (int k = 0; k < 4; k++) a += sinf(w * h[k] * t + ch * 1.7f + k * 2.3f) / (k + 1);
  return a * 0.48f;
}

static inline Rig droneRigAt(const DronePath& P, float t) {
  t = fmodf(t, DRONE_T); if (t < 0) t += DRONE_T;
  float s = droneS(P, t), dt = 0.02f;
  V3 p = P.at(s);
  // bakis: 0.35 s ilerideki noktaya dogru (yumusak donusler)
  V3 ahead = P.at(droneS(P, t + 0.35f)); V3 f = ahead - p; f.y = 0; f = norm(f);
  // yatma: yanal ivme
  V3 v0 = (P.at(droneS(P, t)) - P.at(droneS(P, t - dt))) / dt, v1 = (P.at(droneS(P, t + dt)) - P.at(droneS(P, t))) / dt;
  V3 acc = (v1 - v0) / dt; V3 rgt = norm(cross(f, V3(0, 1, 0)));
  float roll = clampf(-dot(acc, rgt) * rad(4.0f), -rad(9.0f), rad(9.0f)) + rad(0.25f) * pnoise(t, 1);
  float pitch = rad(-7.0f) + rad(0.2f) * pnoise(t, 2);
  float yawJ = rad(0.15f) * pnoise(t, 3);
  // yon vektorleri (serbest kamerayla ayni el kurali)
  float cy = cosf(yawJ), sy = sinf(yawJ); V3 fy = norm(V3(f.x * cy - f.z * sy, 0, f.x * sy + f.z * cy));
  V3 F = norm(fy * cosf(pitch) + V3(0, 1, 0) * sinf(pitch));
  V3 R = norm(cross(F, V3(0, 1, 0))); V3 U = cross(R, F);
  V3 R2 = R * cosf(roll) + U * sinf(roll), U2 = U * cosf(roll) - R * sinf(roll);
  Rig g; g.eye = p + V3(pnoise(t, 4), pnoise(t, 5), pnoise(t, 6)) * 0.0025f;
  g.cF = F; g.cR = R2; g.cU = U2;
  g.base = g.eye; g.fwd = F; g.up = U2; g.right = R2;
  return g;
}
