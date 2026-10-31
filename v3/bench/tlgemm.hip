// tlgemm.hip -- standalone f16 WMMA GEMM microbenchmark for the EXL3 prefill rewrite (gfx1201).
// C[M][N] (f32) = A[M][K] (f16, row-major, the rotated activations) x W^T, W given in WMMA fragment order
// Wf[ks][nt][32 lanes][8] (what exl3r4d's decode_frag produces: lane (h, n) holds W[16 nt + n][16 ks + 8 (e >> 2) +
// 4 h + (e & 3)]). Config by -D: MT NPW WM WN KC WLDS GROUP. Prints ms, TFLOPS, max rel err vs a naive kernel.
//   hipcc -O3 --offload-arch=gfx1201 -DMT=4 -DNPW=4 -DWM=4 -DWN=2 -DKC=2 -DWLDS=1 -DGROUP=8 tlgemm.hip -o tlg
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
#ifndef MT
#define MT 4
#endif
#ifndef NPW
#define NPW 4
#endif
#ifndef WM
#define WM 4
#endif
#ifndef WN
#define WN 2
#endif
#ifndef KC
#define KC 2
#endif
#ifndef WLDS
#define WLDS 1
#endif
#ifndef GROUP
#define GROUP 8
#endif
#ifndef PURE
#define PURE 0   // 1: no memory at all in the K loop (fragments from registers): the WMMA ceiling of the shape
#endif
#ifndef WRING
#define WRING 1  // weight fragments two chunks ahead (register ring, loop unrolled by 2)
#endif
#ifndef LB
#define LB 1     // workgroups a CU the launch bound asks for (occupancy hint)
#endif
#define CK(x) do { if ((x) != hipSuccess) { printf("fail %s line %d\n", #x, __LINE__); exit(1); } } while (0)

static __device__ __forceinline__ void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
constexpr int BM = 16 * MT * WM, BN = 16 * NPW * WN, NTH = 32 * WM * WN, RT = BM / 16, CT = BN / 16;
constexpr int ASLAB = RT * KC * 32 * 8;            // halfs, fragment order [RT][KC][32]
constexpr int NSUB = KC * CT, WSLAB = NSUB * 32 * 8;
constexpr int NLA = (BM * KC * 2 + NTH - 1) / NTH; // 16-B activation chunks a thread
static_assert(NSUB * 32 % NTH == 0, "every thread loads the same number of weight chunks");
static_assert(BM * KC * 2 % NTH == 0, "every thread loads the same number of activation chunks");
constexpr int NLW = (NSUB * 32 + NTH - 1) / NTH;   // 16-B weight chunks a thread (WLDS)

__global__ void __launch_bounds__(NTH, LB) gemm(const _Float16* __restrict__ A, const v8h* __restrict__ Wf,
                                                float* __restrict__ C, int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) _Float16 abuf[2 * ASLAB];
#if WLDS
  __shared__ __attribute__((aligned(16))) v8h wbuf[2 * NSUB * 32];
#endif
  const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int wm = wave / WN, wn = wave % WN;
  const int nmb = (M + BM - 1) / BM, nnb = N / BN, lin = blockIdx.x;
  const int per = GROUP * nnb, first = lin / per * GROUP, r0 = lin % per;
  const int gm = min(nmb - first, GROUP);
  const int mb = first + r0 % gm, nb = r0 / gm;
  const int m0 = mb * BM, ntile0 = nb * CT, ksteps = K / 16, ntiles = N / 16;
  v4u sa[NLA];
#if WLDS
  v4u sw[NLW];
#endif
  auto load = [&](int c) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, r = idx / (2 * KC), q = idx % (2 * KC);
      const int gr = m0 + r < M ? m0 + r : M - 1;
      sa[i] = PURE == 2 ? (v4u)((unsigned)(c + i)) : *(const v4u*)(A + (size_t)gr * K + c * KC * 16 + q * 8);
    }
#if WLDS
#pragma unroll
    for (int i = 0; i < NLW; i++) {
      const int idx = tid + NTH * i, sub = idx / 32, l = idx % 32, u = sub / CT, j = sub % CT;
      sw[i] = *(const v4u*)(Wf + ((size_t)(c * KC + u) * ntiles + ntile0 + j) * 32 + l);
    }
#endif
  };
  auto store = [&](int b) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, r = idx / (2 * KC), q = idx % (2 * KC);
      {
        const int rt = r / 16, n = r % 16, u = q / 2, qh = q % 2;
        _Float16* const t0 = abuf + b * ASLAB + (rt * KC + u) * 32 * 8 + qh * 4;
        *(v2u*)(t0 + n * 8) = (v2u){sa[i][0], sa[i][1]};
        *(v2u*)(t0 + (n + 16) * 8) = (v2u){sa[i][2], sa[i][3]};
      }
    }
#if WLDS
#pragma unroll
    for (int i = 0; i < NLW; i++) {
      const int idx = tid + NTH * i;
      *(v4u*)(wbuf + b * NSUB * 32 + idx) = sw[i];
    }
#endif
  };
  v8f acc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[t][j] = (v8f)(0.f);
#if !WLDS
  // weight fragments a whole chunk (KC steps) ahead: wq[u] holds step c KC + u, refilled with step (c + 1) KC + u
  v8h wq[KC][NPW];
  auto wfetch = [&](int ks, v8h (&o)[NPW]) {
#pragma unroll
    for (int j = 0; j < NPW; j++) o[j] = PURE == 2 ? (v8h)((_Float16)(ks + j)) : Wf[((size_t)ks * ntiles + ntile0 + wn * NPW + j) * 32 + lane];
  };
#pragma unroll
  for (int u = 0; u < KC; u++) wfetch(u, wq[u]);
#endif
  const int nch = ksteps / KC;
  // A: chunk c + 2 is loaded at the end of iteration c (after its WMMAs) and stored at the end of c + 1, so every
  // wait in the loop targets loads a whole iteration old (gfx12's load counter retires in order)
  load(0);
  store(0);
  load(min(1, nch - 1));
  lds_barrier();
#if !WLDS && WRING
  // weight fragments two chunks ahead: wr[s][u] holds chunk (c with c % 2 == s) step u; refilled with chunk c + 2
  v8h wr[2][KC][NPW];
#pragma unroll
  for (int u = 0; u < KC; u++) { wfetch(u, wr[0][u]); wfetch(min(KC + u, ksteps - 1), wr[1][u]); }
  auto body = [&](int c, v8h (&cur)[KC][NPW]) {
    const v8h* const ab = (const v8h*)(abuf + (c & 1) * ASLAB);
#pragma unroll
    for (int u = 0; u < KC; u++) {
      v8h af[MT];
#pragma unroll
      for (int t = 0; t < MT; t++) af[t] = ab[((wm * MT + t) * KC + u) * 32 + lane];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int t = 0; t < MT; t++) acc[t][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(cur[u][j], af[t], acc[t][j]);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int u = 0; u < KC; u++) wfetch(min((c + 2) * KC + u, ksteps - 1), cur[u]);
    store((c + 1) & 1);
    load(min(c + 2, nch - 1));
    lds_barrier();
  };
  for (int c = 0; c < nch; c += 2) {
    body(c, wr[0]);
    if (c + 1 < nch) body(c + 1, wr[1]);
  }
#else
  for (int c = 0; c < nch; c++) {
    const v8h* const ab = (const v8h*)(abuf + (c & 1) * ASLAB);
#if WLDS
    const v8h* const wb = wbuf + (c & 1) * NSUB * 32;
#endif
#pragma unroll
    for (int u = 0; u < KC; u++) {
      v8h af[MT];
#pragma unroll
      for (int t = 0; t < MT; t++) af[t] = (PURE == 1 || PURE == 3) ? (v8h)((_Float16)(c + t + u)) : ab[((wm * MT + t) * KC + u) * 32 + lane];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
#if WLDS
        const v8h wf = PURE ? (v8h)((_Float16)(c * 3 + j + lane)) : wb[(u * CT + wn * NPW + j) * 32 + lane];
#else
        const v8h wf = wq[u][j];
#endif
#pragma unroll
        for (int t = 0; t < MT; t++) acc[t][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(wf, af[t], acc[t][j]);
      }
#if !WLDS
      wfetch(min((c + 1) * KC + u, ksteps - 1), wq[u]);
#endif
    }
#ifndef SCHB
#define SCHB 1
#endif
    if (SCHB) __builtin_amdgcn_sched_barrier(0);   // keep the store after this chunk's WMMAs (no hoisting)
    if (PURE != 1) { store((c + 1) & 1); load(min(c + 2, nch - 1)); }
    if (PURE != 1) lds_barrier();
  }
#endif
  const int h = lane >> 4, n = lane & 15;
#pragma unroll
  for (int t = 0; t < MT; t++) {
    const int row = m0 + 16 * (wm * MT + t) + n;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      float* dst = C + (size_t)row * N + (ntile0 + wn * NPW + j) * 16 + 8 * h;
      *(v4u*)dst = (v4u){__builtin_bit_cast(unsigned, acc[t][j][0]), __builtin_bit_cast(unsigned, acc[t][j][1]),
                         __builtin_bit_cast(unsigned, acc[t][j][2]), __builtin_bit_cast(unsigned, acc[t][j][3])};
      *(v4u*)(dst + 4) = (v4u){__builtin_bit_cast(unsigned, acc[t][j][4]), __builtin_bit_cast(unsigned, acc[t][j][5]),
                               __builtin_bit_cast(unsigned, acc[t][j][6]), __builtin_bit_cast(unsigned, acc[t][j][7])};
    }
  }
}

__global__ void ref(const _Float16* A, const _Float16* W, float* C, int M, int N, int K) {
  const int m = blockIdx.y, n = blockIdx.x * 256 + threadIdx.x;
  if (n >= N) return;
  float s = 0.f;
  for (int k = 0; k < K; k++) s += (float)A[(size_t)m * K + k] * (float)W[(size_t)n * K + k];
  C[(size_t)m * N + n] = s;
}

int main(int argc, char** argv) {
  const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), it = argc > 4 ? atoi(argv[4]) : 20;
  if (N % BN || K % (16 * KC)) { printf("shape not divisible (BN %d, KC %d)\n", BN, KC); return 1; }
  std::vector<_Float16> a((size_t)M * K), w((size_t)N * K), wf((size_t)N * K);
  unsigned s = 1;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 9) & 0xffff) / 65536.f - 0.5f; };
  for (auto& x : a) x = (_Float16)rnd();
  for (auto& x : w) x = (_Float16)rnd();
  const int ntiles = N / 16;
  for (int ks = 0; ks < K / 16; ks++)
    for (int nt = 0; nt < ntiles; nt++)
      for (int l = 0; l < 32; l++)
        for (int e = 0; e < 8; e++) {
          const int h = l >> 4, n = l & 15, k = 16 * ks + 8 * (e >> 2) + 4 * h + (e & 3);
          wf[(((size_t)ks * ntiles + nt) * 32 + l) * 8 + e] = w[(size_t)(16 * nt + n) * K + k];
        }
  _Float16 *dA, *dW, *dWf; float *dC, *dR;
  CK(hipMalloc(&dA, a.size() * 2)); CK(hipMalloc(&dW, w.size() * 2)); CK(hipMalloc(&dWf, wf.size() * 2));
  CK(hipMalloc(&dC, (size_t)M * N * 4)); CK(hipMalloc(&dR, (size_t)M * N * 4));
  CK(hipMemcpy(dA, a.data(), a.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(dW, w.data(), w.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(dWf, wf.data(), wf.size() * 2, hipMemcpyHostToDevice));
  const int grid = ((M + BM - 1) / BM) * (N / BN);
  gemm<<<grid, NTH>>>(dA, (const v8h*)dWf, dC, M, N, K);
  CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  for (int i = 0; i < 3; i++) gemm<<<grid, NTH>>>(dA, (const v8h*)dWf, dC, M, N, K);
  CK(hipEventRecord(e0));
  for (int i = 0; i < it; i++) gemm<<<grid, NTH>>>(dA, (const v8h*)dWf, dC, M, N, K);
  CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
  float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= it;
  double err = 0;
  if (!getenv("NOREF")) {
    ref<<<dim3((N + 255) / 256, M), 256>>>(dA, dW, dR, M, N, K);
    CK(hipDeviceSynchronize());
    std::vector<float> c((size_t)M * N), r((size_t)M * N);
    CK(hipMemcpy(c.data(), dC, c.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(r.data(), dR, r.size() * 4, hipMemcpyDeviceToHost));
    double mx = 0; for (size_t i = 0; i < r.size(); i++) mx = fmax(mx, fabs(r[i]));
    for (size_t i = 0; i < r.size(); i++) err = fmax(err, fabs(c[i] - r[i]) / mx);
  }
  printf("MT%d NPW%d WM%d WN%d KC%d WLDS%d G%d LB%d (BM %d BN %d, %d thr): M=%d N=%d K=%d  %.3f ms  %.1f TFLOPS  err %.2e\n",
         MT, NPW, WM, WN, KC, WLDS, GROUP, LB, BM, BN, NTH, M, N, K, ms, 2.0 * M * N * K / ms / 1e9, err);
  return 0;
}
