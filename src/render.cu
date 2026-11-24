// render.cu — render.metal'in CUDA cevirisi. Path tracer (render), isik grid bake'i (bake) ve yumusak isik filtresi
// (dnPrep, dnAtrous). Metal'in donanim acceleration structure'i yerine bvh_build.h ile kurulan BVH'ler dolasiliyor.
// Malzemeler ve isik hesabi render.metal ile birebir ayni tutuldu.
#include "cuda_math.h"
#include "gpu_types.h"
#include "gpu_api.h"
#include "bvh_build.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#define PI 3.14159265358979f
#define MASK_WORLD 1u
#define MASK_ROBOT 2u
#define EPS 2.5e-4f
typedef unsigned int uint;

enum { M_FLOOR = 0, M_WOOD, M_PAINT, M_TRIM, M_FABRIC, M_RUG, M_PLASTIC, M_METAL, M_CERAMIC, M_LEAF, M_ROBOT, M_EMIT, M_MATTE,
       M_FACADE, M_FOLIAGE, M_RUBBER, M_CONCRETE, M_CURTAIN, M_BOOK, M_ART, M_SKIN, M_SOCKET };

struct BVH { const float4* nodes; const float4* tris; };
struct Ctx {
  BVH blas[4];               // 0 world, 1 dino, 2 robot, 3 shell (kamera uzayi)
  const uint* idx;
  const float3* pos; const float3* nrm; const float3* tex;
  const TriA* tri;
  const Inst* inst;          // object->world (normal donusumu icin)
  const float4* inv;         // instance i icin world->object satirlari: inv[i*3 + 0..2]
  cudaTextureObject_t g[6];
};
__constant__ Uni cU;
__constant__ Ctx cC;
#define DF __device__ __forceinline__

// ---------------------------------------------------------------- random and noise
DF uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
DF float rnd(uint& s) { s = pcg(s); return float(s >> 8) * (1.0f / 16777216.0f); }
DF float hashi(int x, int y, int z) { return float(pcg((uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u) >> 8) * (1.0f / 16777216.0f); }
DF float vnoise(float3 p) {
  float3 fl = floor3(p); int ix = (int)fl.x, iy = (int)fl.y, iz = (int)fl.z; float3 f = p - fl; f = f * f * (3.0f - 2.0f * f);
  float a = mix(hashi(ix, iy, iz), hashi(ix + 1, iy, iz), f.x), b = mix(hashi(ix, iy + 1, iz), hashi(ix + 1, iy + 1, iz), f.x);
  float c = mix(hashi(ix, iy, iz + 1), hashi(ix + 1, iy, iz + 1), f.x), d = mix(hashi(ix, iy + 1, iz + 1), hashi(ix + 1, iy + 1, iz + 1), f.x);
  return mix(mix(a, b, f.y), mix(c, d, f.y), f.z);
}
DF float fbm(float3 p, int oct) { float a = 0.5f, s = 0; for (int i = 0; i < oct; i++) { s += a * vnoise(p); p = p * 2.03f + f3(17.1f, 3.3f, 9.7f); a *= 0.5f; } return s; }
DF float lum(float3 c) { return dot(c, f3(0.2126f, 0.7152f, 0.0722f)); }
DF float3 srgb(uint c) { float3 v = f3(float(c & 255u), float((c >> 8) & 255u), float((c >> 16) & 255u)) * (1.0f / 255.0f); return pow3(v, 2.2f); }
DF void basis(float3 n, float3& t, float3& b) {
  float s = n.z >= 0 ? 1.0f : -1.0f; float a = -1.0f / (s + n.z); float c = n.x * n.y * a;
  t = f3(1.0f + s * n.x * n.x * a, s * c, -s * n.x); b = f3(c, s + n.y * n.y * a, -n.y);
}

// ---------------------------------------------------------------- sky
DF float3 sky(float3 d, const Uni& U) {
  float3 s = xyz(U.sunDir); float cs = dot(d, s); float h = fmaxf(d.y, 0.0f);
  float3 c = mix(f3(0.62f, 0.72f, 0.86f), f3(0.13f, 0.27f, 0.66f), powf(h, 0.42f));
  c += f3(1.0f, 0.72f, 0.45f) * (0.55f * powf(fmaxf(cs, 0.0f), 28.0f) + 0.16f * powf(fmaxf(cs, 0.0f), 5.0f));
  if (d.y > 0.01f) {
    float2 uv = f2(d.x, d.z) / (d.y + 0.12f) * 0.9f;
    float cl = fbm(f3(uv.x, uv.y, 3.7f), 5);
    cl = smoothstep(0.50f, 0.74f, cl) * smoothstep(0.01f, 0.12f, d.y);
    float3 cc = f3(1.0f, 0.96f, 0.90f) * (0.85f + 0.9f * powf(fmaxf(cs, 0.0f), 3.0f));
    c = mix(c, cc, cl * 0.85f);
  } else {
    c = mix(c, f3(0.20f, 0.21f, 0.20f), smoothstep(0.0f, -0.08f, d.y));
  }
  return c * U.misc2.y;
}
DF float3 skyAmbientE(float3 n, const Uni& U) {
  return f3(0.36f, 0.50f, 0.78f) * U.misc2.y * PI * (0.62f + 0.38f * n.y);
}

// ---------------------------------------------------------------- BVH traversal (Metal intersector'unun yerine)
// tmin < t < tmax olan en yakin ucgen; any = true ise ilk bulunan yeter (golge)
DF bool bvhHit(const BVH& B, float3 o, float3 d, float tmin, float& tmax, uint& prim, float2& bc, bool any) {
  float3 id = f3(fabsf(d.x) > 1e-12f ? 1.0f / d.x : copysignf(1e12f, d.x), fabsf(d.y) > 1e-12f ? 1.0f / d.y : copysignf(1e12f, d.y), fabsf(d.z) > 1e-12f ? 1.0f / d.z : copysignf(1e12f, d.z));
  float3 oid = o * id;
  int stack[64]; int sp = 0; int ni = 0; bool hit = false;
  while (true) {
    float4 a = __ldg(&B.nodes[ni * 2]), b = __ldg(&B.nodes[ni * 2 + 1]);
    int cnt = __float_as_int(b.w);
    if (cnt > 0) {
      int first = __float_as_int(a.w);
      for (int i = 0; i < cnt; i++) {
        const float4* T = &B.tris[(size_t)(first + i) * 3];
        float4 v0 = __ldg(&T[0]), e1 = __ldg(&T[1]), e2 = __ldg(&T[2]);
        float3 E1 = xyz(e1), E2 = xyz(e2);
        float3 pv = cross(d, E2); float det = dot(E1, pv);
        if (fabsf(det) < 1e-14f) continue;
        float inv = 1.0f / det; float3 tv = o - xyz(v0);
        float u = dot(tv, pv) * inv; if (u < 0.0f || u > 1.0f) continue;
        float3 q = cross(tv, E1); float v = dot(d, q) * inv; if (v < 0.0f || u + v > 1.0f) continue;
        float t = dot(E2, q) * inv;
        if (t > tmin && t < tmax) { tmax = t; prim = __float_as_uint(v0.w); bc = f2(u, v); hit = true; if (any) return true; }
      }
      if (sp == 0) break; ni = stack[--sp]; continue;
    }
    int l = __float_as_int(a.w);
    float4 la = __ldg(&B.nodes[l * 2]), lb = __ldg(&B.nodes[l * 2 + 1]), ra = __ldg(&B.nodes[l * 2 + 2]), rb = __ldg(&B.nodes[l * 2 + 3]);
    float3 l0 = xyz(la) * id - oid, l1 = xyz(lb) * id - oid, r0 = xyz(ra) * id - oid, r1 = xyz(rb) * id - oid;
    float ln = fmaxf(fmaxf(fminf(l0.x, l1.x), fminf(l0.y, l1.y)), fmaxf(fminf(l0.z, l1.z), tmin));
    float lf = fminf(fminf(fmaxf(l0.x, l1.x), fmaxf(l0.y, l1.y)), fminf(fmaxf(l0.z, l1.z), tmax));
    float rn = fmaxf(fmaxf(fminf(r0.x, r1.x), fminf(r0.y, r1.y)), fmaxf(fminf(r0.z, r1.z), tmin));
    float rf = fminf(fminf(fmaxf(r0.x, r1.x), fmaxf(r0.y, r1.y)), fminf(fmaxf(r0.z, r1.z), tmax));
    bool hl = ln <= lf, hr = rn <= rf;
    if (hl && hr) {
      int nearN = ln <= rn ? l : l + 1, farN = ln <= rn ? l + 1 : l;
      if (sp < 64) stack[sp++] = farN;
      ni = nearN;
    } else if (hl) ni = l;
    else if (hr) ni = l + 1;
    else { if (sp == 0) break; ni = stack[--sp]; }
  }
  return hit;
}

struct Hit { bool ok; bool shell; float t; uint prim; uint inst; float2 bc; };
__constant__ uint cInstMask[3] = {MASK_WORLD, MASK_WORLD, MASK_ROBOT};
DF void toObject(const Ctx& C, int i, float3 o, float3 d, float3& oo, float3& dd) {
  if (i == 0) { oo = o; dd = d; return; }
  float4 a = C.inv[i * 3], b = C.inv[i * 3 + 1], c = C.inv[i * 3 + 2];
  oo = f3(dot(xyz(a), o) + a.w, dot(xyz(b), o) + b.w, dot(xyz(c), o) + c.w);
  dd = f3(dot(xyz(a), d), dot(xyz(b), d), dot(xyz(c), d));
}
DF Hit traceWorld(const Ctx& C, float3 o, float3 d, float tmin, float tmax, uint mask) {
  Hit h; h.ok = false; h.shell = false; h.t = tmax; h.prim = 0; h.inst = 0; h.bc = f2(0.0f);
  for (int i = 0; i < 3; i++) {
    if (!(cInstMask[i] & mask)) continue;
    float3 oo, dd; toObject(C, i, o, d, oo, dd);
    uint p; float2 bc;
    if (bvhHit(C.blas[i], oo, dd, tmin, h.t, p, bc, false)) { h.ok = true; h.prim = p; h.inst = (uint)i; h.bc = bc; }
  }
  return h;
}
DF bool occluded(const Ctx& C, float3 o, float3 d, float tmax, uint mask) {
  for (int i = 0; i < 3; i++) {
    if (!(cInstMask[i] & mask)) continue;
    float3 oo, dd; toObject(C, i, o, d, oo, dd);
    uint p; float2 bc; float tm = tmax;
    if (bvhHit(C.blas[i], oo, dd, 0.0f, tm, p, bc, true)) return true;
  }
  return false;
}

// ---------------------------------------------------------------- surfaces
struct Surf {
  float3 p, n, ng; float3 alb, F0, trans, emis; float rough; uint mat; bool back; bool metal;
};
DF bool insideRoom(float3 p, const Uni& U) {
  return p.x > U.roomLo.x && p.y > U.roomLo.y && p.z > U.roomLo.z && p.x < U.roomHi.x && p.y < U.roomHi.y && p.z < U.roomHi.z;
}

__device__ Surf getSurf(const Ctx& C, Hit h, float3 o, float3 d, bool lite, const Uni& U, float3 cR, float3 cU, float3 cF) {
  Surf s; s.p = o + d * h.t;
  uint gp = (h.shell ? U.counts.w : C.inst[h.inst].meta.x) + h.prim;
  uint i0 = C.idx[gp * 3], i1 = C.idx[gp * 3 + 1], i2 = C.idx[gp * 3 + 2];
  float w1 = h.bc.x, w2 = h.bc.y, w0 = 1.0f - w1 - w2;
  float3 P0 = C.pos[i0], P1 = C.pos[i1], P2 = C.pos[i2];
  float3 ngo = cross(P1 - P0, P2 - P0);
  float3 no = C.nrm[i0] * w0 + C.nrm[i1] * w1 + C.nrm[i2] * w2;
  float3 tx = C.tex[i0] * w0 + C.tex[i1] * w1 + C.tex[i2] * w2;
  float3 ng, n;
  if (h.shell) { ng = ngo.x * cR + ngo.y * cU + ngo.z * cF; n = no.x * cR + no.y * cU + no.z * cF; }
  else if (h.inst == 0) { ng = ngo; n = no; }
  else { Inst I = C.inst[h.inst]; ng = f3(dot(xyz(I.r0), ngo), dot(xyz(I.r1), ngo), dot(xyz(I.r2), ngo)); n = f3(dot(xyz(I.r0), no), dot(xyz(I.r1), no), dot(xyz(I.r2), no)); }
  ng = normalize(ng); n = normalize(n);
  s.back = dot(ng, d) > 0.0f;
  if (s.back) ng = -ng;
  if (dot(n, ng) < 0.0f) n = -n;
  s.ng = ng;
  TriA ta = C.tri[gp]; uint mat = ta.mat; s.mat = mat;
  float3 col = srgb(ta.col); float pa = float(ta.col >> 24) * (1.0f / 255.0f);
  s.alb = col; s.F0 = f3(0.04f); s.trans = f3(0.0f); s.emis = f3(0.0f); s.rough = 0.6f; s.metal = false;
  float dist = h.t;
  switch (mat) {
    case M_FLOOR: {
      float w = tx.z + 0.012f * sinf(tx.x * 1.7f + pa * 40.0f) + 0.010f * (vnoise(f3(tx.x * 2.5f, pa * 50.0f, 1.0f)) - 0.5f);
      float rings = fbm(f3(tx.x * 0.9f, w * 95.0f, pa * 13.0f), 4);
      rings = smoothstep(0.25f, 0.75f, rings);
      float fine = lite ? 0.5f : vnoise(f3(tx.x * 45.0f, w * 1300.0f, pa * 7.0f));
      float drift = vnoise(f3(tx.x * 1.1f, pa * 31.0f, 4.0f));
      float3 base = mix(f3(0.335f, 0.215f, 0.118f), f3(0.455f, 0.315f, 0.180f), pa * 0.7f + drift * 0.3f);
      s.alb = base * (0.86f + 0.20f * rings) * (0.93f + 0.14f * fine);
      float wear = lite ? 0.5f : fbm(f3(s.p.x * 2.1f, 0.3f, s.p.z * 2.1f), 3);
      s.rough = 0.20f + 0.12f * wear + 0.05f * fine;
      if (!lite) { float g2 = vnoise(f3(tx.x * 55.0f, (w + 0.0006f) * 1100.0f, pa * 7.0f)); n = normalize(n + f3(0, 0, 1) * (g2 - fine) * 0.05f); }
    } break;
    case M_WOOD: {
      float along = pa < 0.5f ? tx.x : tx.y; float3 q = pa < 0.5f ? f3(tx.y, tx.z, 0) : f3(tx.x, tx.z, 0);
      float w = q.x * 1.0f + q.y * 0.6f + 0.006f * sinf(along * 7.0f);
      float rings = 0.5f + 0.5f * sinf(w * 320.0f + 4.0f * vnoise(f3(along * 3.0f, w * 30.0f, 2.0f)));
      float fine = lite ? 0.5f : vnoise(f3(along * 60.0f, q.x * 900.0f, q.y * 900.0f));
      s.alb = col * (0.78f + 0.30f * rings) * (0.9f + 0.2f * fine);
      s.rough = 0.30f + 0.1f * fine;
    } break;
    case M_PAINT: {
      s.rough = 0.85f; s.F0 = f3(0.03f);
      if (!lite && dist < 3.0f) { float e = 0.0015f; float3 q = s.p * 420.0f; float n0 = vnoise(q); n = normalize(n + 0.035f * f3(vnoise(q + f3(e * 420.0f, 0, 0)) - n0, vnoise(q + f3(0, e * 420.0f, 0)) - n0, vnoise(q + f3(0, 0, e * 420.0f)) - n0) * 6.0f); }
    } break;
    case M_TRIM: { s.rough = 0.32f; } break;
    case M_FABRIC: {
      float3 an = fabs3(n); float2 uv = an.y > an.x && an.y > an.z ? f2(tx.x, tx.z) : (an.x > an.z ? f2(tx.y, tx.z) : f2(tx.x, tx.y));
      float fade = lite ? 0.0f : saturate(1.6f - dist * 0.55f);
      float wv = sinf(uv.x * 1500.0f) * sinf(uv.y * 1500.0f);
      float mot = fbm(f3(uv.x * 14.0f, uv.y * 14.0f, pa * 9.0f), 3);
      s.alb = col * (0.84f + 0.2f * mot) * (1.0f + 0.16f * wv * fade);
      s.rough = 0.95f; s.F0 = f3(0.02f);
      if (!lite) { float fz = powf(1.0f - saturate(dot(n, -d)), 3.0f); s.alb *= 1.0f + 0.45f * fz; }
    } break;
    case M_RUG: {
      float2 q = f2(tx.x, tx.z) / 0.46f; q += 0.06f * f2(vnoise(f3(tx.x * 3.0f, tx.z * 3.0f, 1.0f)), vnoise(f3(tx.x * 3.0f, tx.z * 3.0f, 5.0f)));
      float d1 = fabsf(fract(q.x + q.y) - 0.5f), d2 = fabsf(fract(q.x - q.y) - 0.5f);
      float line = smoothstep(0.035f, 0.022f, fminf(d1, d2));
      float3 cream = f3(0.60f, 0.54f, 0.44f), dark = f3(0.105f, 0.092f, 0.082f);
      float v = 0.88f + 0.22f * pa;
      s.alb = mix(cream, dark, line) * v * mix(0.55f, 1.0f, saturate(tx.y));
      s.rough = 1.0f; s.F0 = f3(0.015f);
      if (tx.y > 0.05f) n = normalize(ng * 0.45f + f3(0, 1, 0));
    } break;
    case M_PLASTIC: { s.rough = 0.06f + 0.6f * pa; s.F0 = f3(0.045f); } break;
    case M_METAL: { s.metal = true; s.F0 = col; s.alb = f3(0.0f); s.rough = 0.08f + 0.6f * pa; } break;
    case M_CERAMIC: {
      if (pa > 0.5f) { float nz = lite ? 0.5f : fbm(tx * 260.0f, 3); s.alb = col * (0.8f + 0.4f * nz); s.rough = 0.9f; s.F0 = f3(0.02f); }
      else { s.rough = 0.09f; s.F0 = f3(0.05f); }
    } break;
    case M_LEAF: {
      float u = tx.x, v = tx.y;
      float mid = smoothstep(0.045f, 0.0f, fabsf(v));
      float side = smoothstep(0.75f, 1.0f, fabsf(sinf((u * 9.0f - fabsf(v) * 2.2f) * PI))) * 0.5f;
      float nz = fbm(f3(u * 9.0f, v * 5.0f, tx.z * 31.0f), 3);
      s.alb = col * (0.75f + 0.5f * nz); s.alb = mix(s.alb, f3(0.42f, 0.50f, 0.20f), fmaxf(mid, side * 0.35f));
      s.trans = s.alb * f3(0.75f, 1.0f, 0.3f) * 0.55f;
      s.rough = 0.22f + 0.1f * nz; s.F0 = f3(0.045f);
    } break;
    case M_ROBOT: {
      float r = length(f2(tx.x, tx.z));
      s.alb = col; s.rough = 0.07f; s.F0 = f3(0.045f);
      if (!lite) {
        float sm = fbm(f3(tx.x * 14.0f, 2.0f, tx.z * 14.0f), 4);
        s.rough += 0.16f * smoothstep(0.42f, 0.75f, sm);
        float du = vnoise(f3(tx.x * 3600.0f, 7.0f, tx.z * 3600.0f)); float du2 = vnoise(f3(tx.x * 900.0f, 3.0f, tx.z * 900.0f));
        float dust = smoothstep(0.925f, 0.975f, du) * smoothstep(0.35f, 0.7f, du2 + 0.25f * sm);
        s.alb = mix(s.alb, f3(0.30f, 0.285f, 0.26f), dust * 0.5f); s.rough = mix(s.rough, 0.9f, dust);
        float scr = vnoise(f3(r * 5200.0f, atan2f(tx.z, tx.x) * 3.0f, 1.0f)); s.rough += 0.05f * scr;
      }
      if (pa > 0.75f) { s.rough = 0.42f; }
    } break;
    case M_EMIT: { s.emis = col * pa * 40.0f; s.alb = f3(0.02f); } break;
    case M_MATTE: { s.rough = 1.0f; s.F0 = f3(0.02f); if (!lite) s.alb *= 0.85f + 0.3f * vnoise(tx * 900.0f); } break;
    case M_FACADE: {
      float2 q = f2(tx.x / 3.1f, tx.y / 2.9f); float2 f = fract(q); float2 idd = floor2(q);
      float win = step(0.22f, f.x) * step(f.x, 0.78f) * step(0.25f, f.y) * step(f.y, 0.80f);
      float hsh = hashi((int)idd.x, (int)idd.y, 7);
      s.alb = mix(col * (0.9f + 0.2f * vnoise(f3(tx.x * 0.7f, tx.y * 0.7f, 1.0f))), f3(0.05f, 0.07f, 0.10f) + 0.25f * step(0.8f, hsh), win);
      s.rough = mix(0.9f, 0.15f, win);
    } break;
    case M_FOLIAGE: { float nz = fbm(tx * 5.0f, 4), n2 = fbm(tx * 23.0f, 3); s.alb = col * (0.25f + 1.1f * nz) * (0.45f + 1.1f * n2); s.rough = 0.8f; s.trans = s.alb * 0.25f;
      if (!lite) n = normalize(n + 0.9f * (f3(fbm(tx * 23.0f + 7.1f, 2), fbm(tx * 23.0f + 13.3f, 2), fbm(tx * 23.0f + 29.7f, 2)) - 0.5f)); } break;
    case M_RUBBER: { s.rough = 0.55f; s.F0 = f3(0.035f); } break;
    case M_CONCRETE: { float nz = lite ? 0.5f : fbm(tx * 30.0f, 4); s.alb = col * (0.75f + 0.5f * nz); s.rough = 0.9f; } break;
    case M_CURTAIN: {
      float wv = lite ? 0.0f : sinf(tx.x * 2600.0f) * sinf(tx.y * 2600.0f) * saturate(1.5f - dist * 0.6f);
      s.alb = col * (0.92f + 0.08f * wv); s.trans = col * 0.42f; s.rough = 0.95f; s.F0 = f3(0.02f);
    } break;
    case M_BOOK: {
      float band = step(0.035f, tx.y) * step(tx.y, 0.075f) + step(-0.085f, tx.y) * step(tx.y, -0.075f);
      float3 lab = pa > 0.5f ? f3(0.75f, 0.70f, 0.60f) : f3(0.03f);
      s.alb = mix(col, lab, band * 0.9f);
      if (n.y > 0.7f) s.alb = f3(0.72f, 0.68f, 0.58f) * (0.85f + 0.15f * sinf(tx.x * 4000.0f));
      s.rough = 0.55f;
    } break;
    case M_ART: {
      float2 q = f2(tx.x, tx.y); float sd = tx.z;
      float3 bg = f3(0.80f, 0.74f, 0.64f); float3 c = bg;
      for (int k = 0; k < 4; k++) {
        float fk = float(k) + sd * 10.0f; float2 cc = f2(0.25f + 0.5f * fract(sinf(fk * 12.9f) * 43758.5f), 0.25f + 0.5f * fract(sinf(fk * 78.2f) * 12543.1f));
        float rr = 0.12f + 0.16f * fract(sinf(fk * 3.7f) * 9631.7f);
        float3 pc = k == 0 ? f3(0.62f, 0.25f, 0.13f) : (k == 1 ? f3(0.10f, 0.20f, 0.26f) : (k == 2 ? f3(0.75f, 0.52f, 0.20f) : f3(0.30f, 0.38f, 0.30f)));
        float dd = k % 2 == 0 ? length(q - cc) - rr : fmaxf(fabsf(q.x - cc.x) - rr * 0.7f, fabsf(q.y - cc.y) - rr * 1.3f);
        c = mix(c, pc, smoothstep(0.004f, 0.0f, dd));
      }
      s.alb = c; s.rough = 0.7f;
    } break;
    case M_SKIN: {
      s.rough = 0.36f; s.F0 = f3(0.045f);
      if (!lite) {
        float3 q = tx * 520.0f; float e = 0.35f; float n0 = fbm(q, 2);
        float3 g = f3(fbm(q + f3(e, 0, 0), 2) - n0, fbm(q + f3(0, e, 0), 2) - n0, fbm(q + f3(0, 0, e), 2) - n0);
        n = normalize(n + g * 0.45f * saturate(1.4f - dist * 1.2f));
        s.alb *= 0.90f + 0.2f * n0;
      }
    } break;
    default: break;
  }
  if (dot(n, ng) < 0.05f) n = normalize(n + ng * 0.3f);
  if (dot(n, -d) < 0.02f) n = normalize(n - d * (0.04f - dot(n, -d)));
  s.n = n;
  return s;
}

// ---------------------------------------------------------------- BSDF helpers
DF float3 fresnel(float3 F0, float c) { float m = 1.0f - saturate(c); float m2 = m * m; return F0 + (1.0f - F0) * (m2 * m2 * m); }
DF float G1(float nx, float a) { float a2 = a * a; return 2.0f * nx / (nx + sqrtf(a2 + (1.0f - a2) * nx * nx)); }
DF float3 specEval(const Surf& s, float3 v, float3 l) {
  float a = fmaxf(s.rough * s.rough, 0.003f); float3 hv = normalize(v + l);
  float nh = saturate(dot(s.n, hv)), nv = fmaxf(dot(s.n, v), 1e-4f), nl = fmaxf(dot(s.n, l), 1e-4f);
  float a2 = a * a; float dd = nh * nh * (a2 - 1.0f) + 1.0f; float D = a2 / (PI * dd * dd);
  return fresnel(s.F0, dot(v, hv)) * (D * G1(nv, a) * G1(nl, a) / (4.0f * nv * nl));
}
DF float3 sampleVNDF(float3 Ve, float a, float2 u) {
  float3 Vh = normalize(f3(a * Ve.x, a * Ve.y, Ve.z));
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = lensq > 1e-12f ? f3(-Vh.y, Vh.x, 0) / sqrtf(lensq) : f3(1, 0, 0);
  float3 T2 = cross(Vh, T1);
  float r = sqrtf(u.x), phi = 2.0f * PI * u.y; float t1 = r * cosf(phi), t2 = r * sinf(phi);
  float sv = 0.5f * (1.0f + Vh.z); t2 = (1.0f - sv) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + sv * t2;
  float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  return normalize(f3(a * Nh.x, a * Nh.y, fmaxf(1e-6f, Nh.z)));
}
DF float3 coneDir(float3 s, float cosMax, float2 u) {
  float ct = 1.0f - u.x * (1.0f - cosMax), st = sqrtf(fmaxf(0.0f, 1.0f - ct * ct)), ph = 2.0f * PI * u.y;
  float3 t, b; basis(s, t, b); return normalize(t * (st * cosf(ph)) + b * (st * sinf(ph)) + s * ct);
}

// bounce-light grid: six directional irradiance volumes
DF float3 sampleG(cudaTextureObject_t t, float3 uvw) { float4 v = tex3D<float4>(t, uvw.x, uvw.y, uvw.z); return f3(v.x, v.y, v.z); }
DF float3 gridE(const Ctx& C, float3 p, float3 n, const Uni& U) {
  p = clamp3(p, xyz(U.roomLo) + 0.06f, xyz(U.roomHi) - 0.06f);
  float3 uvw = (p - xyz(U.gridLo)) * xyz(U.gridInv); float3 n2 = n * n;
  float3 ex = n.x >= 0 ? sampleG(C.g[0], uvw) : sampleG(C.g[1], uvw);
  float3 ey = n.y >= 0 ? sampleG(C.g[2], uvw) : sampleG(C.g[3], uvw);
  float3 ez = n.z >= 0 ? sampleG(C.g[4], uvw) : sampleG(C.g[5], uvw);
  return n2.x * ex + n2.y * ey + n2.z * ez;
}
DF float3 portalPoint(const Uni& U, uint& seed) {
  float u = rnd(seed) * U.misc2.x; uint k = 0; uint np = U.counts.y;
  for (uint i = 0; i < np; i++) { float4 q = U.portal[i]; float a = (q.y - q.x) * (q.w - q.z); k = i; if (u < a) break; u -= a; }
  float4 q = U.portal[k];
  float rx = rnd(seed), ry = rnd(seed);
  return f3(mix(q.x, q.y, rx), mix(q.z, q.w, ry), U.roomHi.z);
}
DF bool throughPortal(float3 p, const Uni& U) {
  float3 s = xyz(U.sunDir); float t = (U.roomHi.z - p.z) / s.z; if (t < 0) return false;
  float3 q = p + s * t; uint np = U.counts.y;
  for (uint i = 0; i < np; i++) { float4 r = U.portal[i]; if (q.x > r.x && q.x < r.y && q.y > r.z && q.y < r.w) return true; }
  return false;
}
DF float3 portalE(const Ctx& C, float3 p, float3 pOff, float3 n, const Uni& U, uint& seed, uint mask, bool mis) {
  float3 q = portalPoint(U, seed); float3 dv = q - p; float r2 = dot(dv, dv); float3 w = dv / sqrtf(r2);
  float cx = dot(n, w); if (cx <= 0.0f || w.z <= 1e-4f) return f3(0.0f);
  if (occluded(C, pOff, w, 1e4f, mask)) return f3(0.0f);
  float G = cx * w.z * U.misc2.x / r2;
  if (mis) { float pp = r2 / (U.misc2.x * w.z), pb = cx / PI; G *= pp / (pp + pb); } else G = fminf(G, PI);
  return sky(w, U) * G;
}
DF float skyRayWeight(float3 p, float3 wi, float nl, const Uni& U) {
  if (wi.z <= 1e-4f) return 1.0f;
  float t = (U.roomHi.z - p.z) / wi.z; float pp = t * t / (U.misc2.x * wi.z), pb = nl / PI;
  return pb / (pp + pb);
}

__device__ float3 shadeLite(const Ctx& C, const Surf& y, float3 wIn, const Uni& U, uint& seed, uint mask, bool misSun, float cosx, float rdist) {
  float3 s = xyz(U.sunDir); float3 L = y.emis; bool in = insideRoom(y.p, U);
  float3 kd = y.metal ? y.F0 * 0.6f : y.alb;
  float ra = rnd(seed), rb = rnd(seed);
  float3 sd = coneDir(s, U.sunDir.w, f2(ra, rb));
  float ns = dot(y.n, sd);
  bool tr = anyPos(y.trans);
  if (ns > 0.0f || tr) {
    float3 f = ns > 0.0f ? kd : y.trans;
    float3 po = y.p + y.ng * (ns > 0.0f ? EPS : -EPS);
    if (!occluded(C, po, sd, 1e4f, mask)) {
      float w = 1.0f;
      if (misSun && in && throughPortal(y.p, U)) {
        float cy = fmaxf(dot(y.n, -wIn), 1e-4f);
        float pv = (fabsf(ns) / U.misc.w) * rdist * rdist / cy; float pb = cosx / PI;
        w = pb / (pb + pv);
      }
      L += f * (1.0f / PI) * xyz(U.sunCol) * fabsf(ns) * w;
    }
  }
  float3 E = f3(0.0f);
  if (in) {
    E += portalE(C, y.p, y.p + y.ng * EPS, y.n, U, seed, mask, false);
    E += gridE(C, y.p + y.n * 0.05f, y.n, U);
    if (tr) L += y.trans * (1.0f / PI) * (portalE(C, y.p, y.p - y.ng * EPS, -y.n, U, seed, mask, false) + gridE(C, y.p - y.n * 0.05f, -y.n, U));
  } else {
    E += skyAmbientE(y.n, U);
  }
  L += kd * (1.0f / PI) * E;
  return L;
}

__device__ float3 sunBounce(const Ctx& C, const Surf& x, float3 fd, const Uni& U, uint& seed, uint mask) {
  float3 s = xyz(U.sunDir);
  float3 q = portalPoint(U, seed);
  Hit h = traceWorld(C, q, -s, 0.0f, 60.0f, mask);
  if (!h.ok) return f3(0.0f);
  Surf y = getSurf(C, h, q, -s, true, U, f3(0.0f), f3(0.0f), f3(0.0f));
  float3 dv = y.p - x.p; float r2 = dot(dv, dv); if (r2 < 1e-8f) return f3(0.0f);
  float r = sqrtf(r2); float3 w = dv / r;
  float cx = dot(x.n, w); if (cx <= 0.0f || dot(x.ng, w) <= 0.0f) return f3(0.0f);
  float cy = dot(y.n, -w);
  float3 fy = cy > 0.0f ? (y.metal ? y.F0 * 0.6f : y.alb) : y.trans;
  float acy = fabsf(cy); if (acy < 1e-4f || !anyPos(fy)) return f3(0.0f);
  if (occluded(C, x.p + x.ng * EPS, w, r - 2.0f * EPS, mask)) return f3(0.0f);
  float nys = fmaxf(dot(y.n, s), 1e-4f);
  float pv = (nys / U.misc.w) * r2 / acy, pb = cx / PI;
  float wgt = pv / (pv + pb);
  return fd * cx * xyz(U.sunCol) * fy * (1.0f / PI) * (U.misc.w * acy / r2) * wgt;
}

// ---------------------------------------------------------------- render kernel
__global__ void __launch_bounds__(128) kRender(float4* acc, float4* aov0, float4* aov1, float4* accE) {
  const Uni& U = cU; const Ctx& C = cC;
  uint tx_ = blockIdx.x * blockDim.x + threadIdx.x, ty_ = blockIdx.y * blockDim.y + threadIdx.y;
  uint W = U.dims.x, H = U.dims.y;
  if (tx_ >= W || ty_ >= H) return;
  uint pix = tx_ + ty_ * W;
  uint seed = pcg(pix * 9781u + U.counts.x * 6271u + 17u);
  float3 s = xyz(U.sunDir);
  float3 sum = f3(0.0f), sAlb = f3(0.0f), sNrm = f3(0.0f), sumE = f3(0.0f); float sDepth = 0, sLum2 = 0, nE = 0;
  uint spp = U.counts.z; uint mode = U.dims.z;
  for (uint si = 0; si < spp; si++) {
    // ---- camera sample: time, pixel (tent filter), lens
    float tt = rnd(seed);
    float3 eye = mix(xyz(U.camP0), xyz(U.camP1), tt);
    float3 cR = normalize(mix(xyz(U.camR0), xyz(U.camR1), tt)), cUp = normalize(mix(xyz(U.camU0), xyz(U.camU1), tt)), cF = normalize(mix(xyz(U.camF0), xyz(U.camF1), tt));
    float j1 = rnd(seed), j2 = rnd(seed);
    float2 jit = f2(j1 < 0.5f ? sqrtf(2.0f * j1) - 1.0f : 1.0f - sqrtf(2.0f - 2.0f * j1), j2 < 0.5f ? sqrtf(2.0f * j2) - 1.0f : 1.0f - sqrtf(2.0f - 2.0f * j2)) * 0.85f;
    float2 pn = (f2((float)tx_, (float)ty_) + 0.5f + jit) / f2((float)W, (float)H) * 2.0f - 1.0f; pn.y = -pn.y * U.lens.w;
    float rr = length(pn); float th = 2.0f * atanf(rr * U.lens.x);
    float3 dc = rr > 1e-6f ? f3(sinf(th) * pn.x / rr, sinf(th) * pn.y / rr, cosf(th)) : f3(0, 0, 1);
    float la = sqrtf(rnd(seed)) * U.lens.y, lp = 2.0f * PI * rnd(seed); float2 lo = f2(cosf(lp), sinf(lp) * 0.85f) * la;
    float3 pf = dc * (U.lens.z / fmaxf(dc.z, 0.05f));
    float3 oc = f3(lo.x, lo.y, 0); float3 dcl = normalize(pf - oc);
    float3 o = eye + cR * oc.x + cUp * oc.y; float3 d = normalize(cR * dcl.x + cUp * dcl.y + cF * dcl.z);

    // ---- primary hit: the robot's own shell (camera space) first, then the world
    Hit h; h.ok = false; h.shell = false; h.t = 0; h.prim = 0; h.inst = 0; h.bc = f2(0.0f); float tmax = 1e4f;
    if (U.misc2.z > 0.5f) {
      float ts = 2.0f; uint p; float2 bc;
      if (bvhHit(C.blas[3], oc, dcl, 1e-4f, ts, p, bc, false)) { h.ok = true; h.shell = true; h.t = ts; h.prim = p; h.inst = 0; h.bc = bc; tmax = ts; }
    }
    Hit hw = traceWorld(C, o, d, 1e-4f, tmax, MASK_WORLD);
    if (hw.ok) h = hw;

    float3 L = f3(0.0f); float3 fAlb = f3(0.0f), fNrm = f3(0.0f); float fDepth = 60.0f;
    float3 Lind = f3(0.0f), E = f3(0.0f); bool dv = false;
    if (!h.ok) {
      L = sky(d, U);
      float cs = dot(d, s); if (cs > 0.99990f) L += xyz(U.sunCol) * 60.0f;
    } else {
      Surf x = getSurf(C, h, o, d, false, U, cR, cUp, cF);
      uint mask = h.shell ? MASK_WORLD : (MASK_WORLD | MASK_ROBOT);
      float3 v = -d; bool in = insideRoom(x.p, U);
      float nv = fmaxf(dot(x.n, v), 1e-4f);
      float3 Fv = fresnel(x.F0, nv);
      float3 fd = x.metal ? f3(0.0f) : x.alb * (1.0f - Fv) * (1.0f / PI);
      float3 kd = x.metal ? f3(0.0f) : x.alb * (1.0f - Fv);
      fAlb = kd; fNrm = x.n; fDepth = h.t; dv = !x.metal;
      L = x.emis;
      bool tr = anyPos(x.trans);
      // sun, both lobes
      float ra = rnd(seed), rb = rnd(seed);
      float3 sd = coneDir(s, U.sunDir.w, f2(ra, rb));
      float ns = dot(x.n, sd);
      if (ns > 0.0f) {
        if (dot(x.ng, sd) > 0.0f && !occluded(C, x.p + x.ng * EPS, sd, 1e4f, mask)) L += xyz(U.sunCol) * (fd + specEval(x, v, sd)) * ns;
      } else if (tr) {
        if (!occluded(C, x.p - x.ng * EPS, sd, 1e4f, mask)) L += xyz(U.sunCol) * x.trans * (1.0f / PI) * (-ns);
      }
      if (in) {
        if (!x.metal) E += portalE(C, x.p, x.p + x.ng * EPS, x.n, U, seed, mask, true) * (1.0f / PI);
        if (tr) L += x.trans * (1.0f / PI) * portalE(C, x.p, x.p - x.ng * EPS, -x.n, U, seed, mask, false);
        if (!x.metal) E += sunBounce(C, x, f3(1.0f / PI), U, seed, mask);
      }
      // one bounce
      float pS = x.metal ? 1.0f : clampf_(lum(Fv) / (lum(Fv) + lum(x.alb) * (1.0f - lum(Fv)) + 1e-4f), 0.06f, 0.94f);
      if (!x.metal && x.rough > 0.55f) pS = 0.0f;
      float3 t1, b1; basis(x.n, t1, b1);
      if (rnd(seed) < pS) {
        float a = fmaxf(x.rough * x.rough, 0.003f);
        float3 Ve = f3(dot(v, t1), dot(v, b1), dot(v, x.n));
        float u1 = rnd(seed), u2 = rnd(seed);
        float3 hh = sampleVNDF(Ve, a, f2(u1, u2));
        float3 hw2 = normalize(t1 * hh.x + b1 * hh.y + x.n * hh.z);
        float3 wi = reflect(-v, hw2);
        float nl = dot(x.n, wi);
        if (nl > 0.0f && dot(x.ng, wi) > 0.0f) {
          float3 wgt = fresnel(x.F0, dot(v, hw2)) * G1(nl, a) / pS;
          Hit hb = traceWorld(C, x.p + x.ng * EPS, wi, 0.0f, 1e4f, mask);
          if (!hb.ok) Lind += wgt * sky(wi, U);
          else { Surf y = getSurf(C, hb, x.p + x.ng * EPS, wi, true, U, f3(0.0f), f3(0.0f), f3(0.0f)); Lind += wgt * shadeLite(C, y, wi, U, seed, mask, false, 1.0f, hb.t); }
        }
      } else {
        float u1 = rnd(seed), u2 = rnd(seed); float r = sqrtf(u1), ph = 2.0f * PI * u2;
        float3 wi = normalize(t1 * (r * cosf(ph)) + b1 * (r * sinf(ph)) + x.n * sqrtf(fmaxf(0.0f, 1.0f - u1)));
        float nl = dot(x.n, wi);
        if (nl > 0.0f && dot(x.ng, wi) > 0.0f) {
          float wgt = 1.0f / (1.0f - pS);
          Hit hb = traceWorld(C, x.p + x.ng * EPS, wi, 0.0f, 1e4f, mask);
          if (!hb.ok) { E += wgt * sky(wi, U) * (in ? skyRayWeight(x.p, wi, nl, U) : 1.0f); }
          else { Surf y = getSurf(C, hb, x.p + x.ng * EPS, wi, true, U, f3(0.0f), f3(0.0f), f3(0.0f)); E += wgt * shadeLite(C, y, wi, U, seed, mask, in, nl, hb.t); }
        }
      }
      if (mode == 1) { L = x.alb + x.F0 * 0.5f; Lind = f3(0.0f); E = f3(0.0f); }
      if (mode == 2) { L = x.n * 0.5f + 0.5f; Lind = f3(0.0f); E = f3(0.0f); }
      if (mode == 3) { L = in ? gridE(C, x.p + x.n * 0.05f, x.n, U) * x.alb / PI : f3(0.0f); Lind = f3(0.0f); E = f3(0.0f); }
      if (mode == 4) { Lind = f3(0.0f); E = f3(0.0f); }
      if (mode == 5) { L = f3(0.0f); }
    }
    // sunlight scattered by dusty air along the camera ray
    if (U.misc.z > 0.0f && mode == 0) {
      float tf = fminf(h.ok ? h.t : 14.0f, 14.0f); float3 pfog = o + d * (tf * rnd(seed));
      if (insideRoom(pfog, U)) {
        float ra = rnd(seed), rb = rnd(seed);
        float3 sd2 = coneDir(s, U.sunDir.w, f2(ra, rb));
        if (!occluded(C, pfog, sd2, 1e4f, MASK_WORLD | MASK_ROBOT)) {
          float g = 0.72f; float cs = dot(d, s); float ph = (1.0f - g * g) / (4.0f * PI * powf(1.0f + g * g - 2.0f * g * cs, 1.5f));
          L += xyz(U.sunCol) * (U.misc.z * tf * (ph + 0.012f));
        }
      }
    }
    float li = lum(Lind); if (li > U.misc.y) Lind *= U.misc.y / li;
    L += Lind;
    float ld = lum(L); if (ld > U.misc.y * 12.0f) L *= U.misc.y * 12.0f / ld;
    if (anyBad(L)) L = f3(0.0f);
    float le = lum(E); if (le > U.misc.y * 2.0f) E *= U.misc.y * 2.0f / le;
    if (anyBad(E)) E = f3(0.0f);
    sum += L; sAlb += fAlb; sNrm += fNrm; sDepth += fDepth;
    if (dv) { sumE += E; nE += 1.0f; sLum2 += lum(E) * lum(E); }
  }
  acc[pix] += f4(sum, float(spp));
  aov0[pix] += f4(sAlb, sDepth);
  aov1[pix] += f4(sNrm, sLum2);
  accE[pix] += f4(sumE, nE);
}

// ---------------------------------------------------------------- filter for the soft diffuse light
__global__ void kDnPrep(const float4* accE, const float4* aov0, const float4* aov1, const float4* acc, float4* oE, float4* oG, uint W, uint H) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return; uint i = x + y * W;
  float4 e = accE[i]; float spp = fmaxf(acc[i].w, 1.0f); float n = e.w;
  if (n < 0.5f) { oE[i] = f4(0, 0, 0, 0); oG[i] = f4(0, 0, 0, -1); return; }
  float3 E = xyz(e) / n; float l = lum(E);
  float var = fmaxf(aov1[i].w / n - l * l, 0.0f) / n;
  float3 nn = xyz(aov1[i]); float ln = length(nn); nn = ln > 1e-6f ? nn / ln : f3(0, 1, 0);
  oE[i] = f4(E, var); oG[i] = f4(nn, aov0[i].w / spp);
}
__global__ void kDnAtrous(const float4* iE, const float4* G, float4* oE, uint Wu, uint Hu, uint stepU) {
  int W = (int)Wu, H = (int)Hu; int x = (int)(blockIdx.x * blockDim.x + threadIdx.x), y = (int)(blockIdx.y * blockDim.y + threadIdx.y); if (x >= W || y >= H) return;
  int i = x + y * W; float4 gp = G[i]; float4 ep = iE[i];
  if (gp.w < 0.0f) { oE[i] = ep; return; }
  float vs = 0, vw = 0;
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
    int q = clampi(x + dx, 0, W - 1) + clampi(y + dy, 0, H - 1) * W; if (G[q].w < 0.0f) continue;
    float w = (dx == 0 ? 2.0f : 1.0f) * (dy == 0 ? 2.0f : 1.0f); vs += w * iE[q].w; vw += w; }
  float sd = sqrtf(fmaxf(vs / fmaxf(vw, 1.0f), 0.0f)); float lp = lum(xyz(ep));
  float za = G[min(x + 1, W - 1) + y * W].w, zb = G[max(x - 1, 0) + y * W].w, zc = G[x + min(y + 1, H - 1) * W].w, zd = G[x + max(y - 1, 0) * W].w;
  float gzx = fminf(za < 0.0f ? 1e9f : fabsf(za - gp.w), zb < 0.0f ? 1e9f : fabsf(zb - gp.w)); if (gzx > 1e8f) gzx = 0.0f;
  float gzy = fminf(zc < 0.0f ? 1e9f : fabsf(zc - gp.w), zd < 0.0f ? 1e9f : fabsf(zd - gp.w)); if (gzy > 1e8f) gzy = 0.0f;
  const float kw[3] = {0.375f, 0.25f, 0.0625f};
  float w0 = kw[0] * kw[0]; float3 sum = xyz(ep) * w0; float wsum = w0, vsum = ep.w * w0 * w0;
  int st = (int)stepU; float sl = 1.0f / (4.0f * sd + 0.02f * lp + 1e-5f);
  for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
    if (dx == 0 && dy == 0) continue;
    int qx = x + dx * st, qy = y + dy * st; if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
    int q = qx + qy * W; float4 gq = G[q]; if (gq.w < 0.0f) continue; float4 eq = iE[q];
    float cn = fmaxf(dot(xyz(gp), xyz(gq)), 0.0f); float c2 = cn * cn, c4 = c2 * c2, c8 = c4 * c4, c16 = c8 * c8; float wn = c16 * c16 * c16;
    float wz = expf(-fabsf(gp.w - gq.w) / ((gzx * float(abs(dx)) + gzy * float(abs(dy))) * float(st) * 1.5f + 0.004f * gp.w + 1e-4f));
    float wl = expf(-fabsf(lp - lum(xyz(eq))) * sl);
    float w = kw[abs(dx)] * kw[abs(dy)] * wn * wz * wl;
    sum += xyz(eq) * w; wsum += w; vsum += eq.w * w * w;
  }
  oE[i] = f4(sum / wsum, vsum / (wsum * wsum));
}

// ---------------------------------------------------------------- bounce-light bake
__global__ void __launch_bounds__(128) kBake(float4* gacc) {
  const Uni& U = cU; const Ctx& C = cC;
  uint nx = U.grid.x, ny = U.grid.y, nz = U.grid.z;
  uint tx_ = blockIdx.x * blockDim.x + threadIdx.x, ty_ = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx_ >= nx * nz || ty_ >= ny) return;
  uint ix = tx_ % nx, iz = tx_ / nx, iy = ty_;
  uint vox = ix + nx * (iy + ny * iz);
  uint seed = pcg(vox * 7919u + U.counts.x * 104729u + 3u);
  float3 cell = 1.0f / (xyz(U.gridInv) * f3((float)nx, (float)ny, (float)nz));
  float3 e[6]; for (int k = 0; k < 6; k++) e[k] = f3(0.0f);
  float inside = 0; uint K = U.grid.w;
  for (uint k = 0; k < K; k++) {
    float r1 = rnd(seed), r2 = rnd(seed), r3 = rnd(seed);
    float3 o = xyz(U.gridLo) + (f3((float)ix, (float)iy, (float)iz) + 0.5f + (f3(r1, r2, r3) - 0.5f) * 0.6f) * cell;
    float z = 1.0f - 2.0f * rnd(seed), ph = 2.0f * PI * rnd(seed), r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float3 d = f3(r * cosf(ph), z, r * sinf(ph));
    Hit h = traceWorld(C, o, d, 0.0f, 1e4f, MASK_WORLD);
    if (!h.ok) continue;
    Surf y = getSurf(C, h, o, d, true, U, f3(0.0f), f3(0.0f), f3(0.0f));
    bool twoSided = y.mat == M_RUG || y.mat == M_LEAF || y.mat == M_CURTAIN || y.mat == M_FOLIAGE;
    if (y.back && !twoSided) { inside += 1.0f; continue; }
    float3 L = shadeLite(C, y, d, U, seed, MASK_WORLD, false, 1.0f, h.t);
    float l = lum(L); if (l > 40.0f) L *= 40.0f / l;
    e[0] += L * fmaxf(d.x, 0.0f); e[1] += L * fmaxf(-d.x, 0.0f); e[2] += L * fmaxf(d.y, 0.0f);
    e[3] += L * fmaxf(-d.y, 0.0f); e[4] += L * fmaxf(d.z, 0.0f); e[5] += L * fmaxf(-d.z, 0.0f);
  }
  for (int k = 0; k < 6; k++) gacc[vox * 7 + k] += f4(e[k], 0.0f);
  gacc[vox * 7 + 6] += f4(inside, float(K), 0.0f, 0.0f);
}

// ================================================================ host tarafi
static Ctx hC; static std::string gErr;
static Inst* dInst = nullptr; static float4* dInv = nullptr;
static cudaArray_t gArr[6] = {}; static int gGx = 0, gGy = 0, gGz = 0;
static float4* dGacc = nullptr; static size_t gNvox = 0;
static int fW = 0, fH = 0; static float4 *dAcc = nullptr, *dA0 = nullptr, *dA1 = nullptr, *dAccE = nullptr, *dE0 = nullptr, *dE1 = nullptr, *dGd = nullptr;
static int gLastDn = 0;

static bool ck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  gErr = std::string(what) + ": " + cudaGetErrorString(e); fprintf(stderr, "CUDA error %s\n", gErr.c_str()); return false;
}
const char* gpuLastError() { return gErr.c_str(); }
static bool pushCtx() { return ck(cudaMemcpyToSymbol(cC, &hC, sizeof(Ctx)), "ctx upload"); }
template <class T> static bool upload(T*& dst, const void* src, size_t bytes, const char* what) {
  if (!ck(cudaMalloc((void**)&dst, bytes ? bytes : 16), what)) return false;
  return bytes == 0 || ck(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), what);
}

bool gpuInit(const GpuMesh parts[4], size_t tBase[4]) {
  int dev = 0; cudaDeviceProp pr; if (!ck(cudaGetDeviceProperties(&pr, dev), "device")) return false;
  fprintf(stderr, "GPU: %s, %.1f GB\n", pr.name, pr.totalGlobalMem / 1073741824.0);
  size_t nv = 0, nt = 0, vBase[4];
  for (int k = 0; k < 4; k++) { vBase[k] = nv; tBase[k] = nt; nv += parts[k].nv; nt += parts[k].nt; }
  std::vector<float> P(nv * 3), N(nv * 3), X(nv * 3); std::vector<uint32_t> I(nt * 3), TA(nt * 2);
  for (int k = 0; k < 4; k++) {
    const GpuMesh& m = parts[k];
    memcpy(&P[vBase[k] * 3], m.pos, m.nv * 12); memcpy(&N[vBase[k] * 3], m.nrm, m.nv * 12); memcpy(&X[vBase[k] * 3], m.tex, m.nv * 12);
    for (size_t i = 0; i < m.nt * 3; i++) I[tBase[k] * 3 + i] = m.idx[i] + (uint32_t)vBase[k];
    for (size_t i = 0; i < m.nt; i++) { TA[(tBase[k] + i) * 2] = m.tmat[i]; TA[(tBase[k] + i) * 2 + 1] = m.tcol[i]; }
  }
  float3 *dP, *dN, *dX; uint32_t* dI; TriA* dT;
  if (!upload(dP, P.data(), P.size() * 4, "pos") || !upload(dN, N.data(), N.size() * 4, "nrm") || !upload(dX, X.data(), X.size() * 4, "tex")) return false;
  if (!upload(dI, I.data(), I.size() * 4, "idx") || !upload(dT, TA.data(), TA.size() * 4, "tri")) return false;
  hC.pos = dP; hC.nrm = dN; hC.tex = dX; hC.idx = dI; hC.tri = dT;
  for (int k = 0; k < 4; k++) {
    BvhBuilder b; std::vector<BvhNodeH> nodes; std::vector<BvhTriH> tris;
    b.build(parts[k].pos, parts[k].idx, parts[k].nt, nodes, tris);
    float4 *dn, *dt;
    if (!upload(dn, nodes.data(), nodes.size() * sizeof(BvhNodeH), "bvh nodes") || !upload(dt, tris.data(), tris.size() * sizeof(BvhTriH), "bvh tris")) return false;
    hC.blas[k].nodes = dn; hC.blas[k].tris = dt;
    fprintf(stderr, "  BVH %d: %zu triangles, %zu nodes\n", k, parts[k].nt, nodes.size());
  }
  if (!ck(cudaMalloc((void**)&dInst, sizeof(Inst) * 3), "inst") || !ck(cudaMalloc((void**)&dInv, sizeof(float4) * 9), "inv")) return false;
  hC.inst = dInst; hC.inv = dInv;
  for (int k = 0; k < 6; k++) hC.g[k] = 0;
  return pushCtx();
}
void gpuSetInstances(const Inst inst[3], const float4 inv[3][3]) {
  ck(cudaMemcpy(dInst, inst, sizeof(Inst) * 3, cudaMemcpyHostToDevice), "inst upload");
  ck(cudaMemcpy(dInv, inv, sizeof(float4) * 9, cudaMemcpyHostToDevice), "inv upload");
}
void gpuSetUniforms(const Uni& U) { ck(cudaMemcpyToSymbol(cU, &U, sizeof(Uni)), "uniform upload"); }

bool gpuGridAlloc(int gx, int gy, int gz) {
  gGx = gx; gGy = gy; gGz = gz;
  cudaChannelFormatDesc cd = cudaCreateChannelDescHalf4();
  for (int k = 0; k < 6; k++) {
    if (!ck(cudaMalloc3DArray(&gArr[k], &cd, make_cudaExtent(gx, gy, gz)), "grid array")) return false;
    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd)); rd.resType = cudaResourceTypeArray; rd.res.array.array = gArr[k];
    cudaTextureDesc td; memset(&td, 0, sizeof(td));
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeElementType; td.normalizedCoords = 1;
    if (!ck(cudaCreateTextureObject(&hC.g[k], &rd, &td, nullptr), "grid texture")) return false;
  }
  return pushCtx();
}
void gpuGridUpload(const uint16_t* data) {
  size_t nvox = (size_t)gGx * gGy * gGz;
  for (int k = 0; k < 6; k++) {
    cudaMemcpy3DParms p; memset(&p, 0, sizeof(p));
    p.srcPtr = make_cudaPitchedPtr((void*)(data + (size_t)k * nvox * 4), (size_t)gGx * 8, gGx, gGy);
    p.dstArray = gArr[k]; p.extent = make_cudaExtent(gGx, gGy, gGz); p.kind = cudaMemcpyHostToDevice;
    ck(cudaMemcpy3D(&p), "grid upload");
  }
}
bool gpuBakeAlloc(size_t nvox) { gNvox = nvox; return ck(cudaMalloc((void**)&dGacc, nvox * 7 * sizeof(float4)), "bake buffer"); }
void gpuBakeClear() { ck(cudaMemset(dGacc, 0, gNvox * 7 * sizeof(float4)), "bake clear"); }
bool gpuBakePass(int gx, int gy, int gz) {
  dim3 bs(16, 8), gs((gx * gz + 15) / 16, (gy + 7) / 8);
  kBake<<<gs, bs>>>(dGacc);
  return ck(cudaGetLastError(), "bake launch") && ck(cudaDeviceSynchronize(), "bake");
}
void gpuBakeRead(float* dst) { ck(cudaMemcpy(dst, dGacc, gNvox * 7 * sizeof(float4), cudaMemcpyDeviceToHost), "bake read"); }

bool gpuFrameAlloc(int W, int H) {
  fW = W; fH = H; size_t b = (size_t)W * H * sizeof(float4);
  float4** bufs[7] = {&dAcc, &dA0, &dA1, &dAccE, &dE0, &dE1, &dGd};
  for (auto p : bufs) if (!ck(cudaMalloc((void**)p, b), "frame buffer")) return false;
  return true;
}
void gpuFrameClear() {
  size_t b = (size_t)fW * fH * sizeof(float4);
  cudaMemset(dAcc, 0, b); cudaMemset(dA0, 0, b); cudaMemset(dA1, 0, b); cudaMemset(dAccE, 0, b);
}
bool gpuRenderPass() {
  dim3 bs(16, 8), gs((fW + 15) / 16, (fH + 7) / 8);
  kRender<<<gs, bs>>>(dAcc, dA0, dA1, dAccE);
  return ck(cudaGetLastError(), "render launch") && ck(cudaDeviceSynchronize(), "render");
}
bool gpuDenoise(int dnIter) {
  dim3 bs(16, 8), gs((fW + 15) / 16, (fH + 7) / 8);
  kDnPrep<<<gs, bs>>>(dAccE, dA0, dA1, dAcc, dE0, dGd, fW, fH);
  for (int k = 0; k < dnIter; k++) kDnAtrous<<<gs, bs>>>((k & 1) ? dE1 : dE0, dGd, (k & 1) ? dE0 : dE1, fW, fH, 1u << k);
  gLastDn = dnIter;
  return ck(cudaGetLastError(), "filter launch") && ck(cudaDeviceSynchronize(), "filter");
}
void gpuFrameRead(float* acc, float* aov0, float* E) {
  size_t b = (size_t)fW * fH * sizeof(float4);
  ck(cudaMemcpy(acc, dAcc, b, cudaMemcpyDeviceToHost), "read acc");
  ck(cudaMemcpy(aov0, dA0, b, cudaMemcpyDeviceToHost), "read aov0");
  ck(cudaMemcpy(E, (gLastDn & 1) ? dE1 : dE0, b, cudaMemcpyDeviceToHost), "read E");
}
