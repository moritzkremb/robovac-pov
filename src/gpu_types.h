// gpu_types.h — host ve GPU'nun ortak veri yapilari (render.metal'deki duzenin aynisi)
#pragma once
#include <vector_types.h>

struct Uni {
  float4 camP0, camP1;
  float4 camR0, camU0, camF0, camR1, camU1, camF1;
  float4 lens;      // x: stereographic k = tan(hfov/4), y: aperture radius (m), z: focus distance (m), w: height/width
  float4 sunDir;    // xyz: towards the sun, w: cos(angular radius)
  float4 sunCol;    // irradiance on a surface facing the sun
  float4 roomLo, roomHi;
  float4 gridLo, gridInv;
  float4 portal[4]; // x0, x1, y0, y1
  float4 misc;      // x: time (s), y: indirect clamp, z: fog density, w: sum of portal area * |n.s|
  float4 misc2;     // x: total portal area, y: sky scale, z: shell present (1/0), w: unused
  uint4 counts;     // x: seed, y: number of portals, z: samples per pixel this pass, w: first triangle of the shell
  uint4 dims;       // x: width, y: height, z: debug mode, w: unused
  uint4 grid;       // x,y,z: voxel counts, w: rays per voxel per pass (bake)
};
// object->world satirlari (r0..r2), meta.x = ilk ucgen
struct Inst { float4 r0, r1, r2; uint4 meta; };
struct TriA { unsigned int mat; unsigned int col; };
