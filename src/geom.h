// geom.h — procedural modelling toolkit: rounded boxes, lathes, tubes, quads and an SDF mesher (surface nets)
#pragma once
#include "base.h"
#ifndef _WIN32
#include <dispatch/dispatch.h>
#endif

// flat quad a,b,c,d (counter-clockwise seen from the side the normal points to)
static inline void quad(Mesh& m, V3 a, V3 b, V3 c, V3 d) {
  V3 n = norm(cross(b - a, c - a));
  u32 i0 = m.v(a, n, a), i1 = m.v(b, n, b), i2 = m.v(c, n, c), i3 = m.v(d, n, d);
  m.quad(i0, i1, i2, i3);
}

// rounded box with half extents h and edge radius r, placed by xf. tex = local position.
static inline void rbox(Mesh& dst, const M34& xf, V3 h, float r, int seg = 2) {
  Mesh m; m.mat = dst.mat; m.col = dst.col;
  r = std::min(r, 0.999f * std::min(h.x, std::min(h.y, h.z)));
  float hh[3] = {h.x, h.y, h.z};
  std::vector<float> co[3];
  for (int a = 0; a < 3; a++) {
    for (int i = seg; i >= 1; i--) co[a].push_back(-(hh[a] - r) - r * tanf(rad(45.0f) * i / seg));
    co[a].push_back(-(hh[a] - r)); co[a].push_back(hh[a] - r);
    for (int i = 1; i <= seg; i++) co[a].push_back((hh[a] - r) + r * tanf(rad(45.0f) * i / seg));
  }
  int N = 2 * seg + 2;
  for (int a = 0; a < 3; a++) for (int sgn = -1; sgn <= 1; sgn += 2) {
    int u = (a + 1) % 3, w = (a + 2) % 3; u32 base = (u32)m.p.size();
    for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
      float q[3]; q[a] = sgn * hh[a]; q[u] = co[u][i]; q[w] = co[w][j];
      float in[3]; for (int k = 0; k < 3; k++) in[k] = clampf(q[k], -(hh[k] - r), hh[k] - r);
      V3 d = norm(V3(q[0] - in[0], q[1] - in[1], q[2] - in[2]));
      V3 pos = V3(in[0], in[1], in[2]) + d * r;
      m.v(pos, d, pos);
    }
    for (int j = 0; j < N - 1; j++) for (int i = 0; i < N - 1; i++) {
      u32 a0 = base + j * N + i, a1 = a0 + 1, a2 = a0 + N + 1, a3 = a0 + N;
      if (sgn > 0) m.quad(a0, a1, a2, a3); else m.quad(a0, a3, a2, a1);
    }
  }
  add(dst, m, xf);
}
// axis-aligned convenience: min/max corners
static inline void rboxMM(Mesh& dst, V3 lo, V3 hi, float r, int seg = 2) { rbox(dst, T((lo + hi) * 0.5f), (hi - lo) * 0.5f, r, seg); }

// surface of revolution about local Y. Profile is (radius, y) ordered so the outside is on the left when walking it
// (bottom centre -> out -> up -> top centre for a closed solid). Segments meeting at more than creaseDeg are kept sharp.
typedef std::vector<std::pair<float, float>> Profile;
static inline void lathe(Mesh& dst, const M34& xf, const Profile& pr, int seg = 32, float creaseDeg = 38) {
  Mesh m; m.mat = dst.mat; m.col = dst.col;
  int n = (int)pr.size(); if (n < 2) return;
  std::vector<V3> sn(n - 1); // segment normals in (r, y): stored as (nr, ny, 0)
  for (int i = 0; i < n - 1; i++) { float dr = pr[i + 1].first - pr[i].first, dy = pr[i + 1].second - pr[i].second; float l = sqrtf(dr * dr + dy * dy); sn[i] = l > 1e-9f ? V3(dy / l, -dr / l, 0) : V3(1, 0, 0); }
  float cc = cosf(rad(creaseDeg));
  for (int s = 0; s < n - 1; s++) {
    V3 nA = sn[s], nB = sn[s];
    if (s > 0 && dot(sn[s - 1], sn[s]) > cc) nA = norm(sn[s - 1] + sn[s]);
    if (s < n - 2 && dot(sn[s + 1], sn[s]) > cc) nB = norm(sn[s + 1] + sn[s]);
    u32 base = (u32)m.p.size();
    for (int j = 0; j <= seg; j++) {
      float ph = 2 * PI * j / seg, c = cosf(ph), sv = sinf(ph);
      V3 pa(pr[s].first * c, pr[s].second, pr[s].first * sv), pb(pr[s + 1].first * c, pr[s + 1].second, pr[s + 1].first * sv);
      m.v(pa, V3(nA.x * c, nA.y, nA.x * sv), pa); m.v(pb, V3(nB.x * c, nB.y, nB.x * sv), pb);
    }
    for (int j = 0; j < seg; j++) {
      u32 A0 = base + j * 2, B0 = A0 + 1, A1 = A0 + 2, B1 = A0 + 3;
      if (pr[s].first > 1e-7f && pr[s + 1].first > 1e-7f) m.quad(A0, B0, B1, A1);
      else if (pr[s].first > 1e-7f) m.tri(A0, B0, A1);
      else if (pr[s + 1].first > 1e-7f) m.tri(A0, B0, B1);
    }
  }
  add(dst, m, xf);
}
// helpers built on the lathe
static inline Profile roundedCylProfile(float r0, float r1, float h, float bevel) { // tapered solid cylinder, y from 0 to h
  Profile p; p.push_back({0, 0}); p.push_back({r0 - bevel, 0});
  for (int i = 1; i <= 3; i++) { float a = rad(90.0f) * i / 3; p.push_back({r0 - bevel + bevel * sinf(a), bevel - bevel * cosf(a)}); }
  for (int i = 0; i <= 3; i++) { float a = rad(90.0f) * i / 3; p.push_back({r1 - bevel + bevel * cosf(a), h - bevel + bevel * sinf(a)}); }
  p.push_back({0, h}); return p;
}
static inline void cyl(Mesh& dst, V3 a, V3 b, float r0, float r1, int seg = 24, float bevel = 0.0015f) { // solid tapered cylinder from a to b
  float h = len(b - a); M34 xf = alignY(b - a); xf.t = a;
  lathe(dst, xf, roundedCylProfile(r0, r1, h, std::min(bevel, 0.45f * std::min(r0, r1))), seg, 50);
}
static inline void ellipsoid(Mesh& dst, const M34& xf, V3 rr, int seg = 28, int rings = 14) {
  Profile p; for (int i = 0; i <= rings; i++) { float a = -PI / 2 + PI * i / rings; p.push_back({std::max(0.0f, cosf(a)), sinf(a)}); }
  p.front().first = 0; p.back().first = 0;
  lathe(dst, xf * S(rr.x, rr.y, rr.z), p, seg, 80);
}
static inline void torus(Mesh& dst, const M34& xf, float R, float r, int seg = 28, int rs = 12) {
  Profile p; for (int i = 0; i <= rs; i++) { float a = -PI / 2 + 2 * PI * i / rs; p.push_back({R + r * cosf(a), r * sinf(a)}); }
  lathe(dst, xf, p, seg, 80);
}

// tube along a polyline with per-point radius
static inline void tube(Mesh& dst, const std::vector<V3>& pts, const std::vector<float>& rr, int seg = 8, bool caps = true) {
  int n = (int)pts.size(); if (n < 2) return;
  Mesh& m = dst; u32 base = (u32)m.p.size();
  V3 Tp = norm(pts[1] - pts[0]);
  V3 N = norm(cross(fabsf(Tp.y) < 0.9f ? V3(0, 1, 0) : V3(1, 0, 0), Tp));
  std::vector<V3> Ts(n);
  for (int i = 0; i < n; i++) {
    V3 Tn = norm(i == 0 ? pts[1] - pts[0] : (i == n - 1 ? pts[n - 1] - pts[n - 2] : pts[i + 1] - pts[i - 1]));
    N = norm(N - Tn * dot(N, Tn)); V3 B = cross(Tn, N); Ts[i] = Tn;
    for (int k = 0; k <= seg; k++) { float a = 2 * PI * k / seg; V3 d = N * cosf(a) + B * sinf(a); V3 pos = pts[i] + d * rr[i]; m.v(pos, d, pos); }
  }
  for (int i = 0; i < n - 1; i++) for (int k = 0; k < seg; k++) {
    u32 a = base + i * (seg + 1) + k, b = a + 1, c = a + seg + 2, d = a + seg + 1;
    m.quad(a, b, c, d);
  }
  if (caps) for (int e = 0; e < 2; e++) {
    int i = e ? n - 1 : 0; V3 nn = e ? Ts[i] : -Ts[i]; u32 c = m.v(pts[i], nn, pts[i]); u32 rb = (u32)m.p.size();
    for (int k = 0; k <= seg; k++) m.v(m.p[base + i * (seg + 1) + k], nn, m.p[base + i * (seg + 1) + k]);
    for (int k = 0; k < seg; k++) { if (e) m.tri(c, rb + k, rb + k + 1); else m.tri(c, rb + k + 1, rb + k); }
  }
}
static inline void tubeC(Mesh& dst, const std::vector<V3>& pts, float r, int seg = 8, bool caps = true) { tube(dst, pts, std::vector<float>(pts.size(), r), seg, caps); }
// Catmull-Rom resample of a control polyline
static inline std::vector<V3> spline(const std::vector<V3>& c, int per) {
  std::vector<V3> o; int n = (int)c.size();
  for (int i = 0; i < n - 1; i++) {
    V3 p0 = c[std::max(i - 1, 0)], p1 = c[i], p2 = c[i + 1], p3 = c[std::min(i + 2, n - 1)];
    for (int k = 0; k < per; k++) { float t = (float)k / per, t2 = t * t, t3 = t2 * t;
      o.push_back(0.5f * ((2.0f * p1) + (p2 - p0) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (3.0f * p1 - p0 - 3.0f * p2 + p3) * t3)); }
  }
  o.push_back(c.back()); return o;
}

// ---- signed distance helpers ----
static inline float sdSphere(V3 p, V3 c, float r) { return len(p - c) - r; }
static inline float sdEllipsoid(V3 p, V3 c, V3 r) { V3 q = p - c; float k0 = len(V3(q.x / r.x, q.y / r.y, q.z / r.z)); float k1 = len(V3(q.x / (r.x * r.x), q.y / (r.y * r.y), q.z / (r.z * r.z))); return k1 > 1e-12f ? k0 * (k0 - 1.0f) / k1 : -std::min(r.x, std::min(r.y, r.z)); }
static inline float sdCapsule(V3 p, V3 a, V3 b, float ra, float rb) { // tapered capsule (approximate: linear radius along the segment)
  V3 pa = p - a, ba = b - a; float h = clampf(dot(pa, ba) / dot(ba, ba), 0, 1); return len(pa - ba * h) - lerpf(ra, rb, h);
}
static inline float sdBox(V3 p, V3 c, V3 h, float r) { V3 q(fabsf(p.x - c.x) - h.x + r, fabsf(p.y - c.y) - h.y + r, fabsf(p.z - c.z) - h.z + r);
  return len(V3(std::max(q.x, 0.f), std::max(q.y, 0.f), std::max(q.z, 0.f))) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.f) - r; }
static inline float smin(float a, float b, float k) { float h = std::max(k - fabsf(a - b), 0.0f) / k; return std::min(a, b) - h * h * k * 0.25f; }
static inline float smax(float a, float b, float k) { return -smin(-a, -b, k); }

// surface nets: mesh the zero set of f inside [lo,hi] with the given cell size. Inside is f < 0. tex = local position.
static inline void sdfMesh(Mesh& dst, const M34& xf, const std::function<float(V3)>& f, V3 lo, V3 hi, float cell) {
  Mesh m; m.mat = dst.mat; m.col = dst.col;
  int nx = (int)ceilf((hi.x - lo.x) / cell), ny = (int)ceilf((hi.y - lo.y) / cell), nz = (int)ceilf((hi.z - lo.z) / cell);
  size_t sx = nx + 1, sy = ny + 1, sz = nz + 1;
  std::vector<float> val(sx * sy * sz); float* vp = val.data(); const std::function<float(V3)>* fp = &f;
#ifdef _WIN32
  // Windows: GCD/blocks yok, ayni dongu OpenMP ile paralel
  #pragma omp parallel for schedule(dynamic)
  for (int kk = 0; kk < (int)sz; kk++) { size_t k = (size_t)kk;
    for (size_t j = 0; j < sy; j++) for (size_t i = 0; i < sx; i++) vp[(k * sy + j) * sx + i] = (*fp)(V3(lo.x + i * cell, lo.y + j * cell, lo.z + k * cell));
  }
#else
  dispatch_apply(sz, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
    for (size_t j = 0; j < sy; j++) for (size_t i = 0; i < sx; i++) vp[(k * sy + j) * sx + i] = (*fp)(V3(lo.x + i * cell, lo.y + j * cell, lo.z + k * cell));
  });
#endif
  auto V = [&](int i, int j, int k) { return val[((size_t)k * sy + j) * sx + i]; };
  std::vector<int> vid((size_t)nx * ny * nz, -1);
  auto C = [&](int i, int j, int k) -> int& { return vid[((size_t)k * ny + j) * nx + i]; };
  static const int E[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (int k = 0; k < nz; k++) for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    float c[8]; int neg = 0;
    for (int q = 0; q < 8; q++) { c[q] = V(i + (q & 1), j + ((q >> 1) & 1), k + ((q >> 2) & 1)); neg += c[q] < 0; }
    if (neg == 0 || neg == 8) continue;
    V3 acc; int cnt = 0;
    for (int e = 0; e < 12; e++) { int a = E[e][0], b = E[e][1]; if ((c[a] < 0) == (c[b] < 0)) continue; float t = c[a] / (c[a] - c[b]);
      V3 pa((float)(a & 1), (float)((a >> 1) & 1), (float)((a >> 2) & 1)), pb((float)(b & 1), (float)((b >> 1) & 1), (float)((b >> 2) & 1)); acc += lerp(pa, pb, t); cnt++; }
    V3 pos = lo + (V3((float)i, (float)j, (float)k) + acc / (float)cnt) * cell;
    float e = cell * 0.6f;
    V3 g(f(pos + V3(e, 0, 0)) - f(pos - V3(e, 0, 0)), f(pos + V3(0, e, 0)) - f(pos - V3(0, e, 0)), f(pos + V3(0, 0, e)) - f(pos - V3(0, 0, e)));
    C(i, j, k) = (int)m.v(pos, norm(g), pos);
  }
  for (int k = 0; k < nz; k++) for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    float v0 = V(i, j, k);
    for (int a = 0; a < 3; a++) {
      int d[3] = {0, 0, 0}; d[a] = 1; int u = (a + 1) % 3, w = (a + 2) % 3; int ijk[3] = {i, j, k};
      if (ijk[u] < 1 || ijk[w] < 1) continue;
      float v1 = V(i + d[0], j + d[1], k + d[2]);
      if ((v0 < 0) == (v1 < 0)) continue;
      int q[4][3]; for (int s = 0; s < 4; s++) { q[s][0] = i; q[s][1] = j; q[s][2] = k; }
      q[0][u] -= 1; q[0][w] -= 1; q[1][w] -= 1; q[3][u] -= 1; // (-,-) (0,-) (0,0) (-,0): CCW seen from +a
      int id[4]; bool ok = true;
      for (int s = 0; s < 4; s++) { if (q[s][0] >= nx || q[s][1] >= ny || q[s][2] >= nz) { ok = false; break; } id[s] = C(q[s][0], q[s][1], q[s][2]); if (id[s] < 0) ok = false; }
      if (!ok) continue;
      if (v0 < 0) m.quad(id[0], id[1], id[2], id[3]); else m.quad(id[0], id[3], id[2], id[1]);
    }
  }
  if (m.ntri()) fixWinding(m, 0);
  add(dst, m, xf);
}
