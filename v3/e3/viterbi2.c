// viterbi2.c -- format-agnostic trellis search core (EXL3/QTIP/CPTC bitshift trellis, 16x16 tiles).
//
// The search is the SAME for every codebook (QTIP 3.1: Mao & Gray bitshift trellis; identical to
// exllamav3's and CPTC's): state s (16 bits) shifts K bits in per step, s' = ((s << K) | j) & 0xffff,
// and a step emits V values, value v of step-state s is cb[s*V + v]. "cb" is the CODEBOOK = a
// 2^L x V state->value map. One LUT covers the whole family (see PLAN.md / V2.md):
//   - E3 mul1 / mul1_i8 / gauss_i8 / int4grid: V=1, cb[s] = fitted table value
//   - CPTC:        V=1, cb[s] = decode((a0,a1,s) -> 64-value table)   [precomputed LUT]
//   - mad (ours):  V=1, cb[s] = byte-sum hash value                    [precomputed LUT]
//   - QTIP 1MAD / 3INST: V=1, computable -- precompute the LUT, OR use the VAL-macro custom
//     template (V2.md) to keep the zero-LUT decode
//   - QTIP HYB:    V=2, cb[s*2+v] = 2D k-means LUT pair with sign flip [precomputed LUT]
//   - QTIP RPTC:   V=1 random-permutation LUT -- DROPPED from the sweep (quality ~= 1MAD at 2
//     bits, 128KB random gather decode = worst in the family on cost)
//   - QuIP# E8 / lattice VQ: not a trellis -- Python-level, outside this interface
//
// Free initial state; squared error minimised over the tile (exllamav3's objective). Tail-biting
// (start = end state) is viterbi2_tiles_tb: ntry roll candidates, each a free rolled run followed
// by a constrained unrolled run (same scheme as viterbi.c's viterbi_tiles_tb / QTIP Algorithm 4).
//
// Backends (same interface; the AVX2 backend uses the identical per-element sub/mul/add ops, so
// its outputs are bit-identical to the scalar backend):
//   scalar: gcc -O3 -march=native -fopenmp -shared -fPIC viterbi2.c -o libviterbi2.so
//   avx2:   gcc -O3 -march=native -fopenmp -shared -fPIC -DV2_AVX2 viterbi2.c -o libviterbi2_avx2.so
//
//   void viterbi2_tiles(const float* x, float* out, int ntiles, const float* cb, int K, int V);
//   void viterbi2_tiles_tb(const float* x, float* out, int ntiles, const float* cb, int K, int V, int ntry);
//     x/out: [ntiles][256] (256/V steps of V values); cb: 65536*V values; K: bits per step (1..8);
//     V: values per step (1 or 2); ntry: tail-biting roll candidates (1..16).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef V2_AVX2
#include <immintrin.h>
#endif

// ---------------- per-step cores (backend-conditional; bit-identical numerics) ----------------
static void gmin_pass(const float* cost, float* gmin, void* b, int NG, int SH, int NJ) {
#ifdef V2_AVX2
  int* bj = (int*)b;
  for (int g8 = 0; g8 < NG; g8 += 8) {
    __m256 m = _mm256_loadu_ps(cost + g8);
    __m256i bi = _mm256_setzero_si256();
    for (int j = 1; j < NJ; j++) {
      __m256 cj = _mm256_loadu_ps(cost + ((size_t)j << SH) + g8);
      __m256 lt = _mm256_cmp_ps(cj, m, _CMP_LT_OQ);
      __m256i mi = _mm256_castps_si256(lt);   // sign bit -> full byte mask
      mi = _mm256_and_si256(mi, _mm256_set1_epi32(0x80808080));
      mi = _mm256_or_si256(mi, _mm256_srli_epi32(mi, 1));
      mi = _mm256_or_si256(mi, _mm256_srli_epi32(mi, 2));
      mi = _mm256_or_si256(mi, _mm256_srli_epi32(mi, 4));
      mi = _mm256_or_si256(mi, _mm256_srli_epi32(mi, 8));
      mi = _mm256_or_si256(mi, _mm256_srli_epi32(mi, 16));
      bi = _mm256_blendv_epi8(bi, _mm256_set1_epi32(j), mi);
      m = _mm256_blendv_ps(m, cj, lt);
    }
    _mm256_storeu_ps(gmin + g8, m);
    _mm256_storeu_si256((__m256i*)(bj + g8), bi);
  }
#else
  int* bi = (int*)b;
  for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; bi[g] = 0; }
  for (int j = 1; j < NJ; j++) {
    const float* cj = cost + (j << SH);
    for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; bi[g] = lt ? j : bi[g]; }
  }
#endif
}

static void ncost_pass(const float* xi, const float* cb, const float* gmin, float* ncost, int NS, int K, int V) {
#ifdef V2_AVX2
  if (V == 1) {
    const float x0 = xi[0];
    if (K >= 3) {
      const int NG = NS >> K;
      const int vp = 1 << (K - 3);
      for (int sg = 0; sg < NG; sg++) {
        const __m256 g = _mm256_set1_ps(gmin[sg]);
        const size_t s0 = (size_t)sg << K;
        for (int v = 0; v < vp; v++) {
          const size_t s = s0 + (size_t)v * 8;
          const __m256 c = _mm256_loadu_ps(cb + s);
          const __m256 d = _mm256_sub_ps(_mm256_set1_ps(x0), c);
          _mm256_storeu_ps(ncost + s, _mm256_add_ps(g, _mm256_mul_ps(d, d)));
        }
      }
    } else {
      for (int s = 0; s < NS; s += 8) {
        int idx[8];
        for (int l = 0; l < 8; l++) idx[l] = (s + l) >> K;
        const __m256 g = _mm256_i32gather_ps(gmin, _mm256_loadu_si256((__m256i*)idx), 4);
        const __m256 c = _mm256_loadu_ps(cb + s);
        const __m256 d = _mm256_sub_ps(_mm256_set1_ps(x0), c);
        _mm256_storeu_ps(ncost + s, _mm256_add_ps(g, _mm256_mul_ps(d, d)));
      }
    }
  } else {
    // V=2: state-major interleaved cb (cb[2q+v]) -- scalar d^2 in both backends (bit-identical;
    // the gmin pass is V-independent and vectorized). HYB is a secondary format.
    const float x0 = xi[0], x1 = xi[1];
    for (int s = 0; s < NS; s++) {
      const float d0 = x0 - cb[2 * s], d1 = x1 - cb[2 * s + 1];
      ncost[s] = gmin[s >> K] + d0 * d0 + d1 * d1;
    }
  }
#else
  if (V == 1) {
    const float x0 = xi[0];
    for (int s = 0; s < NS; s++) { const float d = x0 - cb[s]; ncost[s] = gmin[s >> K] + d * d; }
  } else {
    const float x0 = xi[0], x1 = xi[1];
    for (int s = 0; s < NS; s++) {
      const float d0 = x0 - cb[2 * s], d1 = x1 - cb[2 * s + 1];
      ncost[s] = gmin[s >> K] + d0 * d0 + d1 * d1;
    }
  }
#endif
}

// One Viterbi run over a (rotated) tile: free (c < 0) or group-constrained (c) start, tail group c,
// records the per-position state in st[256/V ... nsteps] (indexed by step), returns the final cost.
static float v2_run(const float* xt, const float* cb, int K, int V, int roll, int c,
                    float* cost, float* ncost, float* gmin, void* bp, int* st) {
  const int NS = 65536, SH = 16 - K, NG = NS >> K, NJ = 1 << K, nsteps = 256 / V;
  if (V == 1) {
    const float x0 = xt[roll & 255];
    for (int s = 0; s < NS; s++) { const float d = x0 - cb[s]; cost[s] = (c < 0 || (s >> K) == c) ? d * d : FLT_MAX; }
  } else {
    const float x0 = xt[(2 * roll) & 255], x1 = xt[(2 * roll + 1) & 255];
    for (int s = 0; s < NS; s++) {
      const float d0 = x0 - cb[2 * s], d1 = x1 - cb[2 * s + 1];
      cost[s] = (c < 0 || (s >> K) == c) ? d0 * d0 + d1 * d1 : FLT_MAX;
    }
  }
  for (int i = 1; i < nsteps; i++) {
    const float* xiv = xt + ((V * i + roll) & 255);   // rolled value index (rolls are multiples of V in practice)
    gmin_pass(cost, gmin, (char*)bp + (size_t)i * NG * sizeof(int), NG, SH, NJ);
    ncost_pass(xiv, cb, gmin, ncost, NS, K, V);
    float* t = cost; cost = ncost; ncost = t;
  }
  int sb = -1; float m = FLT_MAX;
  for (int s = 0; s < NS; s++) if ((c < 0 || (s & (NG - 1)) == c) && cost[s] < m) { m = cost[s]; sb = s; }
  if (sb < 0) return m;
  st[nsteps - 1] = sb;
  for (int i = nsteps - 1; i > 0; i--) {
    const int g = st[i] >> K;
    st[i - 1] = (int)(((((int*)bp)[i * NG + g]) & 0xff) << SH) | g;   // scalar stores bytes; avx2 stores ints
  }
  return m;
}

void viterbi2_tiles(const float* x, float* out, int ntiles, const float* cb, int K, int V) {
#pragma omp parallel
  {
    const int NS = 65536, NG = NS >> K, nsteps = 256 / V;
    float* cost = malloc(sizeof(float) * NS);
    float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * NG);
    void* bp = malloc(sizeof(int) * nsteps * NG);
    int* st = malloc(sizeof(int) * nsteps);
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256;
      v2_run(xt, cb, K, V, 0, -1, cost, ncost, gmin, bp, st);
      float* ot = out + (size_t)t * 256;
      if (V == 1) for (int i = 0; i < nsteps; i++) ot[i] = cb[st[i]];
      else for (int i = 0; i < nsteps; i++) { ot[2 * i] = cb[2 * st[i]]; ot[2 * i + 1] = cb[2 * st[i] + 1]; }
    }
    free(cost); free(ncost); free(gmin); free(bp); free(st);
  }
}

void viterbi2_tiles_tb(const float* x, float* out, int ntiles, const float* cb, int K, int V, int ntry) {
  static const int rolls[16] = {128, 64, 192, 32, 96, 160, 224, 16, 48, 80, 112, 144, 176, 208, 240, 0};
#pragma omp parallel
  {
    const int NS = 65536, NG = NS >> K, nsteps = 256 / V;
    float* cost = malloc(sizeof(float) * NS);
    float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * NG);
    void* bp = malloc(sizeof(int) * nsteps * NG);
    int* st = malloc(sizeof(int) * nsteps);
    int* best = malloc(sizeof(int) * nsteps);
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256;
      float bm = FLT_MAX;
      for (int r = 0; r < ntry && r < 16; r++) {
        v2_run(xt, cb, K, V, rolls[r], -1, cost, ncost, gmin, bp, st);
        const int c = st[(nsteps - rolls[r] / V) & (nsteps - 1)] >> K;   // wrap group at position 0
        const float m = v2_run(xt, cb, K, V, 0, c, cost, ncost, gmin, bp, st);
        if (m < bm) { bm = m; memcpy(best, st, sizeof(int) * nsteps); }
      }
      float* ot = out + (size_t)t * 256;
      if (V == 1) for (int i = 0; i < nsteps; i++) ot[i] = cb[best[i]];
      else for (int i = 0; i < nsteps; i++) { ot[2 * i] = cb[2 * best[i]]; ot[2 * i + 1] = cb[2 * best[i] + 1]; }
    }
    free(cost); free(ncost); free(gmin); free(bp); free(st); free(best);
  }
}
