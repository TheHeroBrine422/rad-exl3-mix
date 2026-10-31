// OP=4 (default): iu4 16x16x32; OP=8: iu8 16x16x16. WMMA with per-K-group scales applied to the i32 partial sums every G/32 instructions:
// mode 0 none, 1 integer group scale (acc += part * s, i32), 2 float (acc_f += float(part) * s)
#include <hip/hip_runtime.h>
#include <cstdio>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
#define N 8
#ifndef OP
#define OP 4
#endif
#define KS (OP == 8 ? 16 : 32)
template <int MODE, int G> __global__ void k(v8i* o, int seed, int it) {
  v2i a = {seed + (int)threadIdx.x, seed * 3}, b = {seed ^ 5, (int)threadIdx.x};
  v8i acc[N] = {}; v8f facc[N] = {};
  for (int i = 0; i < it; i++) {
    v8i p[N] = {}; a.x ^= i; b.y += i; v2i bj[N];
#pragma unroll
    for (int j = 0; j < N; j++) bj[j] = b + j * 0x1111;
#pragma unroll
    for (int g = 0; g < G / KS; g++)
#pragma unroll
      for (int j = 0; j < N; j++) p[j] = OP == 8 ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, bj[j], p[j], false)
                      : __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, bj[j], p[j], false);
    int s = (i * 7 + seed) & 255;
#pragma unroll
    for (int j = 0; j < N; j++) {
      if constexpr (MODE == 0) acc[j] += p[j];
      if constexpr (MODE == 1) acc[j] += p[j] * s;
      if constexpr (MODE == 2) facc[j] += __builtin_convertvector(p[j], v8f) * (float)s;
    }
  }
  v8i r = {};
  for (int j = 0; j < N; j++) r += acc[j] + __builtin_bit_cast(v8i, facc[j]);
  o[blockIdx.x * blockDim.x + threadIdx.x] = r;
}
template <int MODE, int G> void run() {
  int blocks = 64 * 32, threads = 128, it = 4000 * 4 / (G / KS); v8i* o; hipMalloc(&o, blocks * threads * sizeof(v8i));
  k<MODE, G><<<blocks, threads>>>(o, 1, 4); hipDeviceSynchronize();
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0); k<MODE, G><<<blocks, threads>>>(o, 1, it); hipEventRecord(e1); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  double ops = 2.0 * 16 * 16 * G * N * it * (blocks * threads / 32.0);
  printf("iu%d mode %d (%s) group %4d: %6.1f TOPS\n", OP, MODE, MODE == 0 ? "none " : MODE == 1 ? "i32  " : "float", G, ops / ms / 1e9); hipFree(o);
}
int main() { run<0,512>(); run<2,64>(); run<2,128>(); run<2,256>(); run<2,512>(); run<2,1024>(); }
