// peak WMMA throughput on gfx1201: 8 independent accumulators per wave, many waves
#include <hip/hip_runtime.h>
#include <cstdio>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
#define N 8
template <int T> __global__ void k(v8i* o, int seed, int it) {
  v2i a = {seed + (int)threadIdx.x, seed * 3}, b = {seed ^ 5, (int)threadIdx.x};
  v8i c[N] = {}; v8f f[N] = {};
  v2i bj[N];
#pragma unroll
  for (int j = 0; j < N; j++) bj[j] = b + j * 0x1111;
  for (int i = 0; i < it; i++) { a.x ^= i;
#pragma unroll
    for (int j = 0; j < N; j++) { v2i b = bj[j];
      if constexpr (T == 0) c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c[j], false);
      if constexpr (T == 1) c[j] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, false, b, c[j], false);
      if constexpr (T == 2) c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(false, a.x, false, b.x, c[j], false);
      if constexpr (T == 3) f[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, f[j]);
    } }
  v8i s = {};
  for (int j = 0; j < N; j++) s += c[j] + __builtin_bit_cast(v8i, f[j]);
  o[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int T> void run(const char* name, double kdim) {
  int blocks = 64 * 32, threads = 128, it = 4000; v8i* o; hipMalloc(&o, blocks * threads * sizeof(v8i));
  k<T><<<blocks, threads>>>(o, 1, 10); hipDeviceSynchronize();
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0); k<T><<<blocks, threads>>>(o, 1, it); hipEventRecord(e1); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  double ops = 2.0 * 16 * 16 * kdim * N * it * (blocks * threads / 32.0);
  printf("%-22s %7.1f TOPS\n", name, ops / ms / 1e9); hipFree(o);
}
int main() { run<0>("iu8 16x16x16", 16); run<1>("iu4 16x16x32", 32); run<2>("iu4 16x16x16", 16); run<3>("fp8 16x16x16", 16); }
