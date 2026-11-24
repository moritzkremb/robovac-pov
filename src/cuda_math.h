// cuda_math.h — Metal shader dilindeki vektor tipleri/fonksiyonlarinin CUDA karsiliklari
#pragma once
#include <cuda_runtime.h>
#include <math.h>

#define HD __host__ __device__ __forceinline__

// ---- kurucular
HD float2 f2(float x, float y) { return make_float2(x, y); }
HD float2 f2(float s) { return make_float2(s, s); }
HD float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
HD float3 f3(float s) { return make_float3(s, s, s); }
HD float4 f4(float3 v, float w) { return make_float4(v.x, v.y, v.z, w); }
HD float4 f4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }
HD float3 xyz(float4 v) { return make_float3(v.x, v.y, v.z); }

// ---- float2
HD float2 operator+(float2 a, float2 b) { return f2(a.x + b.x, a.y + b.y); }
HD float2 operator-(float2 a, float2 b) { return f2(a.x - b.x, a.y - b.y); }
HD float2 operator*(float2 a, float2 b) { return f2(a.x * b.x, a.y * b.y); }
HD float2 operator*(float2 a, float s) { return f2(a.x * s, a.y * s); }
HD float2 operator*(float s, float2 a) { return f2(a.x * s, a.y * s); }
HD float2 operator/(float2 a, float2 b) { return f2(a.x / b.x, a.y / b.y); }
HD float2 operator/(float2 a, float s) { return f2(a.x / s, a.y / s); }
HD float2 operator+(float2 a, float s) { return f2(a.x + s, a.y + s); }
HD float2 operator-(float2 a, float s) { return f2(a.x - s, a.y - s); }
HD float2& operator+=(float2& a, float2 b) { a.x += b.x; a.y += b.y; return a; }
HD float length(float2 a) { return sqrtf(a.x * a.x + a.y * a.y); }
HD float2 fract(float2 a) { return f2(a.x - floorf(a.x), a.y - floorf(a.y)); }
HD float2 floor2(float2 a) { return f2(floorf(a.x), floorf(a.y)); }

// ---- float3
HD float3 operator+(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
HD float3 operator-(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
HD float3 operator-(float3 a) { return f3(-a.x, -a.y, -a.z); }
HD float3 operator*(float3 a, float3 b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
HD float3 operator*(float3 a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
HD float3 operator*(float s, float3 a) { return f3(a.x * s, a.y * s, a.z * s); }
HD float3 operator/(float3 a, float3 b) { return f3(a.x / b.x, a.y / b.y, a.z / b.z); }
HD float3 operator/(float3 a, float s) { float r = 1.0f / s; return f3(a.x * r, a.y * r, a.z * r); }
HD float3 operator/(float s, float3 a) { return f3(s / a.x, s / a.y, s / a.z); }
HD float3 operator+(float3 a, float s) { return f3(a.x + s, a.y + s, a.z + s); }
HD float3 operator-(float3 a, float s) { return f3(a.x - s, a.y - s, a.z - s); }
HD float3 operator-(float s, float3 a) { return f3(s - a.x, s - a.y, s - a.z); }
HD float3& operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HD float3& operator-=(float3& a, float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
HD float3& operator*=(float3& a, float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
HD float3& operator*=(float3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD float3 cross(float3 a, float3 b) { return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
HD float length(float3 a) { return sqrtf(dot(a, a)); }
HD float3 normalize(float3 a) { return a * rsqrtf(dot(a, a)); }
HD float3 fabs3(float3 a) { return f3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
HD float3 floor3(float3 a) { return f3(floorf(a.x), floorf(a.y), floorf(a.z)); }
HD float3 fmax3(float3 a, float3 b) { return f3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
HD float3 fmin3(float3 a, float3 b) { return f3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
HD float3 pow3(float3 a, float e) { return f3(powf(a.x, e), powf(a.y, e), powf(a.z, e)); }
HD float3 reflect(float3 i, float3 n) { return i - n * (2.0f * dot(n, i)); }
HD bool anyPos(float3 a) { return a.x > 0.0f || a.y > 0.0f || a.z > 0.0f; }
HD bool anyBad(float3 a) { return isnan(a.x) || isnan(a.y) || isnan(a.z) || isinf(a.x) || isinf(a.y) || isinf(a.z); }

// ---- float4
HD float4 operator+(float4 a, float4 b) { return f4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
HD float4& operator+=(float4& a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }

// ---- skaler (Metal adlariyla)
HD float saturate(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }
HD float clampf_(float x, float a, float b) { return fminf(fmaxf(x, a), b); }
HD float3 clamp3(float3 x, float3 a, float3 b) { return fmin3(fmax3(x, a), b); }
HD float mix(float a, float b, float t) { return a + (b - a) * t; }
HD float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
HD float smoothstep(float e0, float e1, float x) { float t = saturate((x - e0) / (e1 - e0)); return t * t * (3.0f - 2.0f * t); }
HD float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
HD float fract(float x) { return x - floorf(x); }
HD int clampi(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }
