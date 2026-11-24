// bvh_build.h — CPU'da binned SAH BVH (Metal'in donanim acceleration structure'i yerine)
#pragma once
#include <vector>
#include <cstdint>
#include <cfloat>
#include <algorithm>
#include <cstring>

// GPU'ya giden dugum: a = (lo.xyz, sol cocuk ya da ilk ucgen), b = (hi.xyz, ucgen sayisi; 0 = ic dugum)
// Ic dugumde cocuklar yan yana: sol = idx, sag = idx + 1.
struct BvhNodeH { float lo[3]; int32_t first; float hi[3]; int32_t count; };
// BVH sirasinda ucgen: v0 (w = mesh icindeki ucgen no, bit olarak), e1 = v1 - v0, e2 = v2 - v0
struct BvhTriH { float v0[4]; float e1[4]; float e2[4]; };

struct BvhBuilder {
  struct Ref { float lo[3], hi[3], c[3]; uint32_t id; };
  std::vector<Ref> refs; std::vector<BvhNodeH> nodes;
  static const int BINS = 16, LEAF = 4;

  static void grow(float* lo, float* hi, const float* plo, const float* phi) {
    for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], plo[k]); hi[k] = std::max(hi[k], phi[k]); }
  }
  static float area(const float* lo, const float* hi) {
    float dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
    if (dx < 0 || dy < 0 || dz < 0) return 0; return 2.0f * (dx * dy + dy * dz + dz * dx);
  }
  void build(const float* pos, const uint32_t* idx, size_t nt, std::vector<BvhNodeH>& outNodes, std::vector<BvhTriH>& outTris) {
    refs.resize(nt);
    for (size_t i = 0; i < nt; i++) {
      Ref& r = refs[i]; r.id = (uint32_t)i;
      for (int k = 0; k < 3; k++) { r.lo[k] = FLT_MAX; r.hi[k] = -FLT_MAX; }
      for (int v = 0; v < 3; v++) { const float* p = &pos[(size_t)idx[i * 3 + v] * 3]; grow(r.lo, r.hi, p, p); }
      for (int k = 0; k < 3; k++) r.c[k] = 0.5f * (r.lo[k] + r.hi[k]);
    }
    nodes.clear(); nodes.reserve(nt * 2 / LEAF + 16);
    nodes.push_back(BvhNodeH());
    if (nt) split(0, 0, (int)nt, 0); else { BvhNodeH& n = nodes[0]; for (int k = 0; k < 3; k++) { n.lo[k] = 0; n.hi[k] = 0; } n.first = 0; n.count = 0; }
    outNodes = nodes;
    outTris.resize(nt);
    for (size_t i = 0; i < nt; i++) {
      uint32_t t = refs[i].id; const float* a = &pos[(size_t)idx[t * 3] * 3]; const float* b = &pos[(size_t)idx[t * 3 + 1] * 3]; const float* c = &pos[(size_t)idx[t * 3 + 2] * 3];
      BvhTriH& o = outTris[i];
      for (int k = 0; k < 3; k++) { o.v0[k] = a[k]; o.e1[k] = b[k] - a[k]; o.e2[k] = c[k] - a[k]; }
      uint32_t idbits = t; memcpy(&o.v0[3], &idbits, 4); o.e1[3] = 0; o.e2[3] = 0;
    }
  }
  // [b, e) araligi icin dugum ni'yi doldur, gerekirse bol
  // derinlik 40'i gecerse SAH yerine ortadan bolunur (GPU yigini 64 eleman)
  void split(int ni, int b, int e, int depth) {
    float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    float clo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, chi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = b; i < e; i++) { grow(lo, hi, refs[i].lo, refs[i].hi); grow(clo, chi, refs[i].c, refs[i].c); }
    { BvhNodeH& n = nodes[ni]; for (int k = 0; k < 3; k++) { n.lo[k] = lo[k]; n.hi[k] = hi[k]; } }
    int cnt = e - b;
    if (cnt <= LEAF) { nodes[ni].first = b; nodes[ni].count = cnt; return; }
    // en iyi bolme: 3 eksen x BINS
    float best = FLT_MAX; int bestAxis = -1, bestBin = 0;
    for (int ax = 0; ax < 3; ax++) {
      float ext = chi[ax] - clo[ax]; if (ext < 1e-12f) continue;
      float blo[BINS][3], bhi[BINS][3]; int bc[BINS] = {0};
      for (int k = 0; k < BINS; k++) for (int j = 0; j < 3; j++) { blo[k][j] = FLT_MAX; bhi[k][j] = -FLT_MAX; }
      float sc = BINS / ext;
      for (int i = b; i < e; i++) { int k = std::min(BINS - 1, (int)((refs[i].c[ax] - clo[ax]) * sc)); bc[k]++; grow(blo[k], bhi[k], refs[i].lo, refs[i].hi); }
      float rA[BINS]; int rC[BINS];
      { float l[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, h[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX}; int c = 0;
        for (int k = BINS - 1; k > 0; k--) { c += bc[k]; if (bc[k]) grow(l, h, blo[k], bhi[k]); rA[k] = area(l, h); rC[k] = c; } }
      { float l[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, h[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX}; int c = 0;
        for (int k = 0; k < BINS - 1; k++) { c += bc[k]; if (bc[k]) grow(l, h, blo[k], bhi[k]);
          if (c == 0 || rC[k + 1] == 0) continue;
          float cost = area(l, h) * c + rA[k + 1] * rC[k + 1];
          if (cost < best) { best = cost; bestAxis = ax; bestBin = k; } } }
    }
    int mid;
    if (bestAxis < 0 || depth > 40) { // butun merkezler ayni nokta ya da cok derin: en uzun eksende ortadan bol
      int ax = 0; for (int k = 1; k < 3; k++) if (chi[k] - clo[k] > chi[ax] - clo[ax]) ax = k;
      mid = b + cnt / 2;
      std::nth_element(refs.data() + b, refs.data() + mid, refs.data() + e, [&](const Ref& x, const Ref& y) { return x.c[ax] < y.c[ax]; });
    }
    else {
      float leafCost = area(lo, hi) * cnt;
      if (best >= leafCost && cnt <= 16) { nodes[ni].first = b; nodes[ni].count = cnt; return; }
      float ext = chi[bestAxis] - clo[bestAxis], sc = BINS / ext;
      Ref* p = std::partition(refs.data() + b, refs.data() + e, [&](const Ref& r) { return std::min(BINS - 1, (int)((r.c[bestAxis] - clo[bestAxis]) * sc)) <= bestBin; });
      mid = (int)(p - refs.data());
      if (mid == b || mid == e) mid = b + cnt / 2;
    }
    int l = (int)nodes.size(); nodes.push_back(BvhNodeH()); nodes.push_back(BvhNodeH());
    nodes[ni].first = l; nodes[ni].count = 0;
    split(l, b, mid, depth + 1); split(l + 1, mid, e, depth + 1);
  }
};
