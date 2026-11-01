// viterbi.c -- E3 trellis check: EXL3/QTIP-style bitshift trellis quantization of 16x16 tiles (CPU, OpenMP).
// State s: 16 bits; each step shifts in K bits: s' = ((s << K) | x) & 0xffff; the value emitted at a step is cb[s'].
// A tile is 256 values (row-major); the Viterbi minimises the plain squared error over the tile (exllamav3's), with a
// free initial state (exllamav3 is tail-biting: start state = end state; same for every codebook compared here).
// The predecessors of s are (s >> K) | (j << (16 - K)), j < 2^K, so the min over them is one pass over 2^(16-K)
// groups: 2 x 65536 ops a step whatever K.
//   gcc -O3 -march=native -fopenmp -shared -fPIC viterbi.c -o libviterbi.so
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// x: [ntiles][256] targets, out: [ntiles][256] quantized values, cb: 65536 codebook values, K: bits a step
void viterbi_tiles(const float* x, float* out, int ntiles, const float* cb, int K) {
  const int NS = 65536, NG = NS >> K, SH = 16 - K, NJ = 1 << K;
#pragma omp parallel
  {
    float* cost = malloc(sizeof(float) * NS);
    float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * NG);
    uint8_t* bp = malloc((size_t)256 * NG);   // the j of the best predecessor group member, per step and group
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256;
      for (int s = 0; s < NS; s++) { const float d = xt[0] - cb[s]; cost[s] = d * d; }
      for (int i = 1; i < 256; i++) {
        uint8_t* b = bp + (size_t)i * NG;
        // j outer, g inner: contiguous loads, compiles to compare + select vectors
        for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; b[g] = 0; }
        for (int j = 1; j < NJ; j++) {
          const float* cj = cost + (j << SH);
          for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; b[g] = lt ? (uint8_t)j : b[g]; }
        }
        const float xi = xt[i];
        for (int s = 0; s < NS; s++) { const float d = xi - cb[s]; ncost[s] = gmin[s >> K] + d * d; }
        float* tmp = cost; cost = ncost; ncost = tmp;
      }
      int s = 0; float m = FLT_MAX;
      for (int q = 0; q < NS; q++) if (cost[q] < m) { m = cost[q]; s = q; }
      float* ot = out + (size_t)t * 256;
      for (int i = 255; i >= 0; i--) {
        ot[i] = cb[s];
        if (i) { const int g = s >> K; s = g | ((int)bp[(size_t)i * NG + g] << SH); }
      }
    }
    free(cost); free(ncost); free(gmin); free(bp);
  }
}

// Fractional K (V3 design §6b): step i shifts pat[i % plen] bits (0..8). A 0-bit step keeps the state and emits
// cb[s ^ salt(i)] (a step-specific XOR salt, so consecutive 0-bit steps do not repeat a value); steps of >= 1 bit
// emit cb[s] as in viterbi_tiles. K = sum(pat) / plen bits a weight. Free initial state, as viterbi_tiles.
static inline int salt_of(int i, int k) { return k ? 0 : (int)((0x9E37u * (unsigned)((i & 15) + 1)) & 0xffff); }
/* fp16 emulation of exllamav3's GPU search (costs, inputs and codebook values in half; err = fma(dh, dh, cost)
 * rounded to half): viterbi_set_f16(1). Only viterbi_tiles_pat. */
static int F16 = 0;
void viterbi_set_f16(int on) { F16 = on; }
static inline float H(float v) { return F16 ? (float)(_Float16)v : v; }
void viterbi_tiles_pat(const float* x, float* out, int ntiles, const float* cb, const int* pat, int plen) {
  const int NS = 65536;
  int kk[256];
  for (int i = 0; i < 256; i++) kk[i] = pat[i % plen];
#pragma omp parallel
  {
    float* cost = malloc(sizeof(float) * NS);
    float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * NS);
    uint8_t* bp = malloc((size_t)256 * (NS >> 1));   // largest group count of a >= 1-bit step
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256;
      { const int sa = salt_of(0, kk[0]); for (int s = 0; s < NS; s++) { const float d = H(H(cb[s ^ sa]) - H(xt[0])); cost[s] = H(d * d); } }
      for (int i = 1; i < 256; i++) {
        const int K = kk[i], sa = salt_of(i, K);
        const float xi = H(xt[i]);
        if (K == 0) {
          for (int s = 0; s < NS; s++) { const float d = H(H(cb[s ^ sa]) - xi); ncost[s] = H(cost[s] + d * d); }
        } else {
          const int NG = NS >> K, SH = 16 - K, NJ = 1 << K;
          uint8_t* b = bp + (size_t)i * (NS >> 1);
          for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; b[g] = 0; }
          for (int j = 1; j < NJ; j++) {
            const float* cj = cost + (j << SH);
            for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; b[g] = lt ? (uint8_t)j : b[g]; }
          }
          for (int s = 0; s < NS; s++) { const float d = H(H(cb[s]) - xi); ncost[s] = H(gmin[s >> K] + d * d); }
        }
        float* tmp = cost; cost = ncost; ncost = tmp;
      }
      int s = 0; float m = FLT_MAX;
      for (int q = 0; q < NS; q++) if (cost[q] < m) { m = cost[q]; s = q; }
      float* ot = out + (size_t)t * 256;
      for (int i = 255; i >= 0; i--) {
        const int K = kk[i];
        ot[i] = cb[s ^ salt_of(i, K)];
        if (i && K) { const int g = s >> K; s = g | ((int)bp[(size_t)i * (NS >> 1) + g] << (16 - K)); }
      }
    }
    free(cost); free(ncost); free(gmin); free(bp);
  }
}

/* ---- tail-biting search (EXL3's actual constraint: the state entering position 0 is the last position's window,
 * i.e. (state_0 >> K) == (state_255 & (2^(16-K) - 1))), integer K only. One run = a Viterbi on the tile rolled by
 * `roll`, start group c (-1: free), end state required to wrap onto c. exllamav3's heuristic = NTRY 1: a free run
 * rolled by L/2 gives the wrap group c, then one constrained run. NTRY > 1 tries the wrap groups found by more
 * rolls (L/2, L/4, 3L/4, ...) and keeps the best constrained result. */
static float tb_run(const float* xt, const float* cb, int K, int roll, int c, float* cost, float* ncost,
                    float* gmin, uint8_t* bj, uint8_t* bp, uint16_t* st) {
  const int NS = 65536, SH = 16 - K, NG = NS >> K, NJ = 1 << K;
  { const float x0 = xt[roll & 255];
    for (int s = 0; s < NS; s++) { const float d = x0 - cb[s]; cost[s] = (c < 0 || (s >> K) == c) ? d * d : FLT_MAX; } }
  for (int i = 1; i < 256; i++) {
    const float xi = xt[(i + roll) & 255];
    for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; bj[g] = 0; }
    for (int j = 1; j < NJ; j++) { const float* cj = cost + (j << SH);
      for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; bj[g] = lt ? (uint8_t)j : bj[g]; } }
    memcpy(bp + (size_t)i * NG, bj, NG);
    for (int s = 0; s < NS; s++) { const float d = xi - cb[s]; ncost[s] = gmin[s >> K] + d * d; }
    float* t = cost; cost = ncost; ncost = t;
  }
  int sb = -1; float m = FLT_MAX;
  for (int s = 0; s < NS; s++) if ((c < 0 || (s & (NG - 1)) == c) && cost[s] < m) { m = cost[s]; sb = s; }
  if (sb < 0) return FLT_MAX;
  st[255] = (uint16_t)sb;
  for (int i = 255; i > 0; i--) { const int g = st[i] >> K; st[i - 1] = (uint16_t)((bp[(size_t)i * NG + g] << SH) | g); }
  return m;
}
void viterbi_tiles_tb(const float* x, float* out, int ntiles, const float* cb, int K, int ntry) {
  static const int rolls[16] = {128, 64, 192, 32, 96, 160, 224, 16, 48, 80, 112, 144, 176, 208, 240, 0};
#pragma omp parallel
  {
    const int NS = 65536, NG = NS >> K;
    float* cost = malloc(sizeof(float) * NS); float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * NG); uint8_t* bj = malloc(NG); uint8_t* bp = malloc((size_t)256 * NG);
    uint16_t st[256], best[256];
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256; float bm = FLT_MAX;
      for (int r = 0; r < ntry && r < 16; r++) {
        tb_run(xt, cb, K, rolls[r], -1, cost, ncost, gmin, bj, bp, st);            /* free run, rolled */
        const int c = st[(256 - rolls[r]) & 255] >> K;                              /* wrap group at position 0 */
        const float m = tb_run(xt, cb, K, 0, c, cost, ncost, gmin, bj, bp, st);     /* constrained, unrolled */
        if (m < bm) { bm = m; memcpy(best, st, sizeof st); }
      }
      for (int i = 0; i < 256; i++) out[(size_t)t * 256 + i] = cb[best[i]];
    }
    free(cost); free(ncost); free(gmin); free(bj); free(bp);
  }
}
