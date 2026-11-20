// viterbi3.c -- the tail-biting bitshift-trellis search of viterbi.c (viterbi_tiles_tb / viterbi_tiles_tb_st), same
// interface and BIT-IDENTICAL outputs, with a fused step that never stores the 2^16-entry cost array.
//
// viterbi.c per step: gmin[g] = min_j cost[j << SH | g] (one pass), then ncost[s] = gmin[s >> K] + (x - cb[s])^2 (a
// second pass), then swap: three 2^16 arrays (cost, ncost, cb = 768 KB) streamed per step per thread -> L2 misses,
// and with SMT siblings or many processes it is L3 / memory bound (11-20: colo / thor / box2 ran 3-10x below the idle
// single-job rate).
// Here: a step's cost[s] is only ever consumed by the NEXT step's group minimum, whose group is g' = s & (NG - 1) and
// member j' = s >> SH. Iterating s = j << SH | g with j outer and g inner visits each group's members in increasing j,
// exactly the order (and strict-< tie rule) of viterbi.c's min pass, so
//     c = gmin_cur[s >> K] + (x - cb[s])^2;  gmin_next[g] = c < gmin_next[g] ? c : gmin_next[g]   (j = 0: init)
// gives the same gmin / argmin bits with a working set of cb (256 KB, read-only, shared by all threads) + two NG
// arrays (K4: 16 KB each). The last step's costs are formed the same way for the final argmin.
//   gcc -O3 -march=native -fopenmp -shared -fPIC viterbi3.c -o libviterbi3.so
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define V3_SIMD 1
#endif

// One run: tile rolled by `roll`, start group c (-1 free), end state must wrap onto c. Writes the state path st[256].
// G: [256][NG] history, row i = the group minima feeding step i (rows 1..255). The hot loop is a pure min (no argmin):
// the backtrack re-forms the 2^K candidate costs of the one group it follows, in viterbi.c's order and strict-< rule.
static inline float cost_at(const float* xt, const float* cb, const float* G, int K, int roll, int c, int i, int p) {
  const float xi = xt[(i + roll) & 255]; const float d = xi - cb[p];
  if (i == 0) return (c < 0 || (p >> K) == c) ? d * d : FLT_MAX;
  return G[(size_t)i * (65536 >> K) + (p >> K)] + d * d;
}
static float tb_run3(const float* xt, const float* cb, int K, int roll, int c, float* G, uint16_t* st) {
  const int NS = 65536, SH = 16 - K, NG = NS >> K, NJ = 1 << K;
  { // step 0 -> G row 1
    const float x0 = xt[roll & 255]; float* gn = G + (size_t)1 * NG;
    for (int j = 0; j < NJ; j++) {
      const float* cbj = cb + (j << SH);
      for (int g = 0; g < NG; g++) { const int s = (j << SH) | g; const float d = x0 - cbj[g];
        const float v = (c < 0 || (s >> K) == c) ? d * d : FLT_MAX;
        gn[g] = (j == 0 || v < gn[g]) ? v : gn[g]; }
    }
  }
  for (int i = 1; i < 255; i++) {
    const float xi = xt[(i + roll) & 255];
    const float* gcur = G + (size_t)i * NG; float* gn = G + (size_t)(i + 1) * NG;
    if (K >= 3) {          // gc[g >> K] is constant over runs of 2^K >= 8 lanes: broadcast instead of a gather
      const int R = 1 << K, NH = NG >> K;
      for (int j = 0; j < NJ; j++) {
        const float* cbj = cb + (j << SH); const float* gc = gcur + ((j << SH) >> K);
        for (int h = 0; h < NH; h++) {
          const float gv = gc[h]; const float* cbh = cbj + h * R; float* gnh = gn + h * R;
          if (j == 0) { for (int l = 0; l < R; l++) { const float d = xi - cbh[l]; gnh[l] = gv + d * d; } }
          else { for (int l = 0; l < R; l++) { const float d = xi - cbh[l]; const float v = gv + d * d; gnh[l] = v < gnh[l] ? v : gnh[l]; } }
        }
      }
#ifdef V3_SIMD
    } else if (K == 2 || K == 1) {   // 8-lane blocks: lane l takes gc[(g >> K) + (l >> K)] -> one permute of a load.
      // fmadd(d, d, g) = the contracted g + d * d of the scalar loops (gcc -ffp-contract=fast, FMA targets); min_ps(v, gn)
      // = v < gn ? v : gn. (Bit-identity with viterbi.c is checked by v3test; build both with the same flags.)
      const __m256i pidx = K == 2 ? _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1) : _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
      const __m256 xv = _mm256_set1_ps(xi);
      for (int j = 0; j < NJ; j++) {
        const float* cbj = cb + (j << SH); const float* gc = gcur + ((j << SH) >> K);
        for (int g = 0; g < NG; g += 8) {
          const __m256 gx = _mm256_permutevar8x32_ps(_mm256_loadu_ps(gc + (g >> K)), pidx);
          const __m256 d = _mm256_sub_ps(xv, _mm256_loadu_ps(cbj + g));
          const __m256 v = _mm256_fmadd_ps(d, d, gx);
          _mm256_storeu_ps(gn + g, j == 0 ? v : _mm256_min_ps(v, _mm256_loadu_ps(gn + g)));
        }
      }
#endif
    } else {
      { const float* gc = gcur; const float* cbj = cb;
        for (int g = 0; g < NG; g++) { const float d = xi - cbj[g]; gn[g] = gc[g >> K] + d * d; } }
      for (int j = 1; j < NJ; j++) {
        const float* cbj = cb + (j << SH); const float* gc = gcur + ((j << SH) >> K);
        for (int g = 0; g < NG; g++) { const float d = xi - cbj[g]; const float v = gc[g >> K] + d * d;
          gn[g] = v < gn[g] ? v : gn[g]; }
      }
    }
  }
  const float* gl = G + (size_t)255 * NG; const float xl = xt[(255 + roll) & 255];
  int sb = -1; float m = FLT_MAX;
  for (int s = 0; s < NS; s++) {
    if (c >= 0 && (s & (NG - 1)) != c) continue;
    const float d = xl - cb[s]; const float v = gl[s >> K] + d * d;
    if (v < m) { m = v; sb = s; }
  }
  if (sb < 0) return FLT_MAX;
  st[255] = (uint16_t)sb;
  for (int i = 255; i > 0; i--) {     // re-form step i-1's candidates of group g; viterbi.c: init j=0, then strict <
    const int g = st[i] >> K; int bj = 0; float bv = cost_at(xt, cb, G, K, roll, c, i - 1, g);
    for (int j = 1; j < NJ; j++) { const float v = cost_at(xt, cb, G, K, roll, c, i - 1, (j << SH) | g); if (v < bv) { bv = v; bj = j; } }
    st[i - 1] = (uint16_t)((bj << SH) | g);
  }
  return m;
}

static void tiles_tb3(const float* x, float* out, uint16_t* st_out, int ntiles, const float* cb, int K, int ntry) {
  static const int rolls[16] = {128, 64, 192, 32, 96, 160, 224, 16, 48, 80, 112, 144, 176, 208, 240, 0};
#pragma omp parallel
  {
    const int NS = 65536, NG = NS >> K;
    float* G = malloc(sizeof(float) * ((size_t)256 * NG + 16));   /* +16: the K <= 2 SIMD load reads 8 */
    uint16_t st[256], best[256];
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256; float bm = FLT_MAX;
      for (int r = 0; r < ntry && r < 16; r++) {
        tb_run3(xt, cb, K, rolls[r], -1, G, st);                      /* free run, rolled */
        const int c = st[(256 - rolls[r]) & 255] >> K;                          /* wrap group at position 0 */
        const float m = tb_run3(xt, cb, K, 0, c, G, st);               /* constrained, unrolled */
        if (m < bm) { bm = m; memcpy(best, st, sizeof st); }
      }
      for (int i = 0; i < 256; i++) out[(size_t)t * 256 + i] = cb[best[i]];
      if (st_out) memcpy(st_out + (size_t)t * 256, best, sizeof best);
    }
    free(G);
  }
}

void viterbi_tiles_tb(const float* x, float* out, int ntiles, const float* cb, int K, int ntry) {
  tiles_tb3(x, out, NULL, ntiles, cb, K, ntry);
}
void viterbi_tiles_tb_st(const float* x, float* out, uint16_t* st_out, int ntiles, const float* cb, int K, int ntry) {
  tiles_tb3(x, out, st_out, ntiles, cb, K, ntry);
}
