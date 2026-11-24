// gpu_api.h — main_cuda.cpp'nin render.cu'yu kullandigi arayuz
#pragma once
#include <cstddef>
#include <cstdint>
#include "gpu_types.h"

struct GpuMesh {
  const float* pos; const float* nrm; const float* tex; // 3 float / vertex
  const uint32_t* idx;                                  // 3 / ucgen (mesh icinde yerel)
  const uint32_t* tmat; const uint32_t* tcol;           // 1 / ucgen
  size_t nv, nt;
};

// 0: world, 1: dino, 2: robot, 3: shell (kamera uzayi). BVH'ler burada kurulur. tBase[k] = parcanin ilk global ucgeni.
bool gpuInit(const GpuMesh parts[4], size_t tBase[4]);
// instance 0..2 icin object->world (Inst) ve world->object (satirlar) — her karede
void gpuSetInstances(const Inst inst[3], const float4 inv[3][3]);
void gpuSetUniforms(const Uni& U);
// isik grid'i: 6 adet RGBA16F 3D texture, data = 6 * nvox * 4 half
bool gpuGridAlloc(int gx, int gy, int gz);
void gpuGridUpload(const uint16_t* data);
// bake: gacc = nvox * 7 float4
bool gpuBakeAlloc(size_t nvox);
void gpuBakeClear();
bool gpuBakePass(int gx, int gy, int gz);
void gpuBakeRead(float* dst);
// kare
bool gpuFrameAlloc(int W, int H);
void gpuFrameClear();
bool gpuRenderPass();
bool gpuDenoise(int dnIter);
// acc, aov0 ve filtrelenmis E (hepsi W*H float4) host'a
void gpuFrameRead(float* acc, float* aov0, float* E);
const char* gpuLastError();
