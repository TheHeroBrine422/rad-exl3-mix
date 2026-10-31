// igemm.hip -- E2: int8 (iu8 16x16x16) and int4 (iu4 16x16x32) WMMA GEMM microbenchmark, gfx1201, T2's structure.
// C[M][N] (f32) = (A_int[M][K] x W_int[N][K]^T) * sa[m] * sw[n]: per-token activation scale x per-output-channel weight
// scale, applied once in the epilogue (the scheme that keeps an integer GEMM at full rate; v3/bench/wmmagrp.hip).
// Both operands arrive in WMMA fragment order (8 bytes a lane a WMMA: 8 int8 or 16 int4 values), as the activation
// quantise/prep pass and the weight decode pass would write them:
//   A: [M/16][K/KS][32 lanes] v2u,   W: [K/KS][N/16][32 lanes] v2u,   KS = 16 (iu8) or 32 (iu4).
// A workgroup is BM x BN = 16 MT WM x 16 NPW WN; activation fragments staged through LDS (double-buffered, one barrier a
// chunk of KC steps), weight fragments straight from global into registers one chunk ahead (T2's arrangement).
// The fragment K-permutation is the same for both operands, so the dot products are exact whatever the hardware's lane
// mapping; a naive reference checks every output (integer sums: must match to f32 rounding of the scale product).
//   hipcc -O3 --offload-arch=gfx1201 -DOP=8 igemm.hip -o ig8;  ./ig8 M N K [iters]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
#ifndef OP
#define OP 8
#endif
#ifndef MT
#define MT 4
#endif
#ifndef NPW
#define NPW 4
#endif
#ifndef WM
#define WM 2
#endif
#ifndef WN
#define WN 4
#endif
#ifndef KC
#define KC (OP == 8 ? 4 : 2)
#endif
#ifndef GROUP
#define GROUP 8
#endif
#ifndef GS
#define GS 0      // activation scale group along K (0: per token, applied in the epilogue; else every GS inputs, in the K loop)
#endif
#ifndef PURE
#define PURE 0   // 1: no memory in the K loop (the WMMA ceiling of the tile)
#endif
#define CK(x) do { if ((x) != hipSuccess) { printf("fail %s line %d\n", #x, __LINE__); exit(1); } } while (0)
constexpr int KS = OP == 8 ? 16 : 32;
constexpr int BM = 16 * MT * WM, BN = 16 * NPW * WN, NTH = 32 * WM * WN, RT = BM / 16;
constexpr int ASLAB = RT * KC * 32;                 // v2u, [RT][KC][32]
constexpr int NLA = ASLAB / 2 / NTH;                // 16-B (two-fragment) loads a thread
static_assert(GS == 0 || (GS % (KS * KC) == 0), "a scale group is whole K chunks");
static_assert(ASLAB % (2 * NTH) == 0, "every thread loads the same number of activation chunks");

static __device__ __forceinline__ void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
static __device__ __forceinline__ v8i mma(v2u w, v2u a, v8i c) {
#if OP == 8
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, __builtin_bit_cast(v2i, w), true, __builtin_bit_cast(v2i, a), c, false);
#else
  return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, __builtin_bit_cast(v2i, w), true, __builtin_bit_cast(v2i, a), c, false);
#endif
}

__global__ void __launch_bounds__(NTH) gemm(const v2u* __restrict__ Af, const v2u* __restrict__ Wf, const float* __restrict__ sa,
                                            const float* __restrict__ sw, float* __restrict__ C, int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) v2u abuf[2 * ASLAB];
  const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int wm = wave / WN, wn = wave % WN;
  const int nmb = (M + BM - 1) / BM, nnb = N / BN, lin = blockIdx.x;
  const int per = GROUP * nnb, first = lin / per * GROUP, r0 = lin % per;
  const int gm = min(nmb - first, GROUP);
  const int mb = first + r0 % gm, nb = r0 / gm;
  const int mt0 = mb * RT, ntile0 = nb * (BN / 16), ksteps = K / KS, ntiles = N / 16, mtiles = (M + 15) / 16;
  v4u sa_[NLA];
  auto load = [&](int c) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, rt = idx / (KC * 16), rem = idx % (KC * 16);   // 16 two-fragment pairs a (rt, step)
      const int mt = min(mt0 + rt, mtiles - 1);
      sa_[i] = PURE ? (v4u)((unsigned)(c + i)) : *(const v4u*)(Af + ((size_t)mt * ksteps + (size_t)c * KC) * 32 + 2 * rem);
    }
  };
  auto store = [&](int b) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i;
      *(v4u*)(abuf + b * ASLAB + 2 * idx) = sa_[i];
    }
  };
  v2u wq[KC][NPW];
  auto wfetch = [&](int ks, v2u (&o)[NPW]) {
#pragma unroll
    for (int j = 0; j < NPW; j++) o[j] = PURE ? (v2u)((unsigned)(ks + j)) : Wf[((size_t)ks * ntiles + ntile0 + wn * NPW + j) * 32 + lane];
  };
  v8i acc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[t][j] = (v8i)(0);
#if GS
  // per-group activation scales: the integer partial sums of a group are scaled into float accumulators
  typedef float v8f __attribute__((ext_vector_type(8)));
  v8f facc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) facc[t][j] = (v8f)(0.f);
  const int ngr = K / GS, CPG = GS / (KS * KC);   // groups along K, chunks a group
  const int lrow = lane & 15;
#endif
#pragma unroll
  for (int u = 0; u < KC; u++) wfetch(u, wq[u]);
  const int nch = ksteps / KC;
  load(0);
  store(0);
  load(min(1, nch - 1));
  lds_barrier();
  for (int c = 0; c < nch; c++) {
    const v2u* const ab = abuf + (c & 1) * ASLAB;
#pragma unroll
    for (int u = 0; u < KC; u++) {
      v2u af[MT];
#pragma unroll
      for (int t = 0; t < MT; t++) af[t] = ab[((wm * MT + t) * KC + u) * 32 + lane];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int t = 0; t < MT; t++) acc[t][j] = mma(wq[u][j], af[t], acc[t][j]);
      wfetch(min((c + 1) * KC + u, ksteps - 1), wq[u]);
    }
#if GS
    if ((c + 1) % CPG == 0) {   // a group ends with this chunk: fold its partial sums
      const int gi = c / CPG;
#pragma unroll
      for (int t = 0; t < MT; t++) {
        const int row = min(16 * (mt0 + wm * MT + t) + lrow, M - 1);
        const float sg = sa[(size_t)row * ngr + gi];
#pragma unroll
        for (int j = 0; j < NPW; j++) { facc[t][j] += __builtin_convertvector(acc[t][j], v8f) * sg; acc[t][j] = (v8i)(0); }
      }
    }
#endif
    __builtin_amdgcn_sched_barrier(0);
    store((c + 1) & 1);
    load(min(c + 2, nch - 1));
    lds_barrier();
  }
  // epilogue: lane (h, n) holds rows 16 tile + n, columns 16 ntile + 8 h + e (the W operand is WMMA's A: C^T layout)
  const int h = lane >> 4, n = lane & 15;
#pragma unroll
  for (int t = 0; t < MT; t++) {
    const int row = 16 * (mt0 + wm * MT + t) + n;
    if (row >= M) continue;
#if GS
    const float ra = 1.f;
#else
    const float ra = sa[row];
#endif
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int col = (ntile0 + wn * NPW + j) * 16 + 8 * h;
      float o[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
#if GS
        o[e] = facc[t][j][e] * sw[col + e];
#else
        o[e] = (float)acc[t][j][e] * ra * sw[col + e];
#endif
      }
      *(v4u*)(C + (size_t)row * N + col) = (v4u){__builtin_bit_cast(unsigned, o[0]), __builtin_bit_cast(unsigned, o[1]),
                                                   __builtin_bit_cast(unsigned, o[2]), __builtin_bit_cast(unsigned, o[3])};
      *(v4u*)(C + (size_t)row * N + col + 4) = (v4u){__builtin_bit_cast(unsigned, o[4]), __builtin_bit_cast(unsigned, o[5]),
                                                       __builtin_bit_cast(unsigned, o[6]), __builtin_bit_cast(unsigned, o[7])};
    }
  }
}

// fragment order, the same K mapping for both operands: lane (h, n), element e of 8 (iu8) or 16 (iu4) along the
// fragment's K: k = KS/2 * (e / (EPL/2)) + EPL/4 * h + e % (EPL/2) ... any bijection works; this is the f16 one scaled.
static inline int fk(int h, int e) {
  constexpr int EPL = OP == 8 ? 8 : 16, Q = EPL / 2;   // values a lane; half of them per K half
  return (KS / 2) * (e / Q) + Q * h + e % Q;
}
static void pack(const std::vector<signed char>& x, int R, int K, std::vector<unsigned>& out) {   // [R/16][K/KS][32][2]
  const int ks = K / KS, EPL = OP == 8 ? 8 : 16;
  out.assign((size_t)(R / 16) * ks * 32 * 2, 0u);
  for (int rt = 0; rt < R / 16; rt++)
    for (int s = 0; s < ks; s++)
      for (int l = 0; l < 32; l++) {
        const int h = l >> 4, n = l & 15;
        unsigned w[2] = {0, 0};
        for (int e = 0; e < EPL; e++) {
          const int v = x[(size_t)(16 * rt + n) * K + KS * s + fk(h, e)];
          if (OP == 8) w[e / 4] |= (unsigned)(v & 0xff) << (8 * (e % 4));
          else w[e / 8] |= (unsigned)(v & 0xf) << (4 * (e % 8));
        }
        const size_t o = (((size_t)rt * ks + s) * 32 + l) * 2;
        out[o] = w[0]; out[o + 1] = w[1];
      }
}
// W packed [K/KS][N/16][32][2]: transpose the tile order of pack()'s [N/16][K/KS]
static void packW(const std::vector<signed char>& x, int N, int K, std::vector<unsigned>& out) {
  std::vector<unsigned> t; pack(x, N, K, t);
  const int ks = K / KS, nt = N / 16;
  out.assign(t.size(), 0u);
  for (int r = 0; r < nt; r++)
    for (int s = 0; s < ks; s++)
      for (int q = 0; q < 64; q++) out[(((size_t)s * nt + r) * 64) + q] = t[(((size_t)r * ks + s) * 64) + q];
}

__global__ void ref(const signed char* A, const signed char* W, const float* sa, const float* sw, float* C, int M, int N, int K) {
  const int m = blockIdx.y, n = blockIdx.x * 256 + threadIdx.x;
  if (n >= N) return;
  const int G = GS ? GS : K;
  float f = 0.f;
  for (int g = 0; g < K / G; g++) {
    int s = 0;
    for (int k = g * G; k < (g + 1) * G; k++) s += (int)A[(size_t)m * K + k] * (int)W[(size_t)n * K + k];
    f += (float)s * (GS ? sa[(size_t)m * (K / G) + g] : 1.f);
  }
  C[(size_t)m * N + n] = f * (GS ? 1.f : sa[m]) * sw[n];
}

int main(int argc, char** argv) {
  const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), it = argc > 4 ? atoi(argv[4]) : 20;
  if (N % BN || K % (KS * KC) || M % 16) { printf("shape not divisible (BN %d, K step %d, M %% 16)\n", BN, KS * KC); return 1; }
  const int lo = OP == 8 ? -128 : -8, span = OP == 8 ? 256 : 16;
  std::vector<signed char> a((size_t)M * K), w((size_t)N * K);
  std::vector<float> fa((size_t)M * (GS ? K / GS : 1)), fw(N);
  unsigned s = 1;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
  for (auto& x : a) x = (signed char)(lo + (int)(rnd() % span));
  for (auto& x : w) x = (signed char)(lo + (int)(rnd() % span));
  for (auto& x : fa) x = 1e-3f * (1 + rnd() % 1000);
  for (auto& x : fw) x = 1e-3f * (1 + rnd() % 1000);
  std::vector<unsigned> ap, wp;
  pack(a, M, K, ap); packW(w, N, K, wp);
  signed char *dA, *dW; unsigned *dAp, *dWp; float *dsa, *dsw, *dC, *dR;
  CK(hipMalloc(&dA, a.size())); CK(hipMalloc(&dW, w.size()));
  CK(hipMalloc(&dAp, ap.size() * 4)); CK(hipMalloc(&dWp, wp.size() * 4));
  CK(hipMalloc(&dsa, fa.size() * 4)); CK(hipMalloc(&dsw, N * 4));
  CK(hipMalloc(&dC, (size_t)M * N * 4)); CK(hipMalloc(&dR, (size_t)M * N * 4));
  CK(hipMemcpy(dA, a.data(), a.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dW, w.data(), w.size(), hipMemcpyHostToDevice));
  CK(hipMemcpy(dAp, ap.data(), ap.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dWp, wp.data(), wp.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(dsa, fa.data(), fa.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dsw, fw.data(), N * 4, hipMemcpyHostToDevice));
  const int grid = ((M + BM - 1) / BM) * (N / BN);
  auto go = [&]() { gemm<<<grid, NTH>>>((const v2u*)dAp, (const v2u*)dWp, dsa, dsw, dC, M, N, K); };
  go(); CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  for (int i = 0; i < 3; i++) go();
  CK(hipEventRecord(e0));
  for (int i = 0; i < it; i++) go();
  CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
  float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= it;
  double err = 0;
  if (!getenv("NOREF") && !PURE) {
    ref<<<dim3((N + 255) / 256, M), 256>>>(dA, dW, dsa, dsw, dR, M, N, K);
    CK(hipDeviceSynchronize());
    std::vector<float> c((size_t)M * N), r((size_t)M * N);
    CK(hipMemcpy(c.data(), dC, c.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(r.data(), dR, r.size() * 4, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < r.size(); i++) err = fmax(err, fabs(c[i] - r[i]) / fmax(fabs(r[i]), 1e-3));
  }
  printf("GS%d iu%d MT%d NPW%d WM%d WN%d KC%d G%d PURE%d (BM %d BN %d): M=%d N=%d K=%d  %.3f ms  %.1f TOPS  max rel err %.1e\n",
         GS, OP, MT, NPW, WM, WN, KC, GROUP, PURE, BM, BN, M, N, K, ms, 2.0 * M * N * K / ms / 1e9, err);
  return 0;
}
