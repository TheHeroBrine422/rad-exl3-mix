// ring102.c -- w102 #136-01 (../w102.md): tail-biting trellis search at any rate, sub-1 included (KA = 0 steps).
// A copy of v3/e3/chain/viterbi.c's viterbi_chain_tb (#102) generalised to 0-bit steps, as qtf2 / viterbi.c's
// viterbi_tiles_pat define them: a 0-bit step keeps the 16-bit state and emits cb[s ^ salt(i)], salt(i) =
// (0x9E37 ((i & 15) + 1)) & 0xffff (i = ring position); a k >= 1 bit step shifts k new bits in and emits cb[s].
// One ring = L = 256 n positions (n = 1: EXL3's tile; n > 1: #102's chain of n row tiles). Ring constraint
// (s_0 >> k_0) == (s_{L-1} & (2^(16 - k_0) - 1)); at k_0 = 0 that is s_0 == s_{L-1}. Search = exllamav3's (NTRY 1): a
// free run rolled by L/2 gives the wrap group at position 0, then one constrained unrolled run; ntry > 1 also tries
// the wrap groups of the rolls L/4, 3L/4, ... (#57's order) and keeps the best constrained result.
//   gcc -O3 -march=native -fopenmp -shared -fPIC ring102.c -o libring102.so
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static inline int salt_of(int i, int k) { return k ? 0 : (int)((0x9E37u * (unsigned)((i & 15) + 1)) & 0xffff); }

static float ring_run(const float* xt, int L, const float* cb, const int* kk, int roll, int c, float* cost,
                      float* ncost, float* gmin, uint8_t* bj, uint8_t* bp, uint16_t* st) {
  const int NS = 65536;
  { const int r0 = roll % L, k0 = kk[r0], sa = salt_of(r0, k0); const float x0 = xt[r0];
    for (int s = 0; s < NS; s++) { const float d = x0 - cb[s ^ sa]; cost[s] = (c < 0 || (s >> k0) == c) ? d * d : FLT_MAX; } }
  for (int i = 1; i < L; i++) {
    const int ri = (i + roll) % L, K = kk[ri];
    const float xi = xt[ri];
    if (K == 0) {
      const int sa = salt_of(ri, 0);
      for (int s = 0; s < NS; s++) { const float d = xi - cb[s ^ sa]; ncost[s] = cost[s] + d * d; }
    } else {
      const int SH = 16 - K, NG = NS >> K, NJ = 1 << K;
      for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; bj[g] = 0; }
      for (int j = 1; j < NJ; j++) { const float* cj = cost + (j << SH);
        for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; bj[g] = lt ? (uint8_t)j : bj[g]; } }
      memcpy(bp + (size_t)i * (NS >> 1), bj, NG);
      for (int s = 0; s < NS; s++) { const float d = xi - cb[s]; ncost[s] = gmin[s >> K] + d * d; }
    }
    float* t = cost; cost = ncost; ncost = t;
  }
  const int M = (NS >> kk[0]) - 1;
  int sb = -1; float m = FLT_MAX;
  for (int s = 0; s < NS; s++) if ((c < 0 || (s & M) == c) && cost[s] < m) { m = cost[s]; sb = s; }
  if (sb < 0) return FLT_MAX;
  st[L - 1] = (uint16_t)sb;
  for (int i = L - 1; i > 0; i--) {
    const int K = kk[(i + roll) % L];
    if (K == 0) { st[i - 1] = st[i]; continue; }
    const int g = st[i] >> K;
    st[i - 1] = (uint16_t)((bp[(size_t)i * (NS >> 1) + g] << (16 - K)) | g);
  }
  return m;
}

// x: [nrings][L] targets (L = 256 n), out: [nrings][L] values; idx (16-bit states) if not NULL; pat: bits per step,
// step i shifts pat[i % plen] (plen divides 16 for the salt to follow the tile position). ntry 0 = free start
// (one unconstrained run, no wrap: the old CPU proxy's reference).
void ring_tb(const float* x, float* out, uint16_t* idx, int nrings, int L, const float* cb, const int* pat, int plen,
             int ntry) {
  static const int rolls[16] = {128, 64, 192, 32, 96, 160, 224, 16, 48, 80, 112, 144, 176, 208, 240, 0};
  int* kk = malloc(sizeof(int) * L);
  for (int i = 0; i < L; i++) kk[i] = pat[i % plen];
#pragma omp parallel
  {
    const int NS = 65536;
    float* cost = malloc(sizeof(float) * NS); float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * (NS >> 1)); uint8_t* bj = malloc(NS >> 1);
    uint8_t* bp = malloc((size_t)L * (NS >> 1));
    uint16_t* st = malloc(sizeof(uint16_t) * L); uint16_t* best = malloc(sizeof(uint16_t) * L);
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < nrings; t++) {
      const float* xt = x + (size_t)t * L; float bm = FLT_MAX;
      if (ntry <= 0) { ring_run(xt, L, cb, kk, 0, -1, cost, ncost, gmin, bj, bp, best); }
      for (int r = 0; r < ntry && r < 16; r++) {
        const int roll = (int)((long)rolls[r] * L / 256);
        ring_run(xt, L, cb, kk, roll, -1, cost, ncost, gmin, bj, bp, st);          /* free run, rolled */
        const int c = st[(L - roll) % L] >> kk[0];                                 /* wrap group at position 0 */
        const float m = ring_run(xt, L, cb, kk, 0, c, cost, ncost, gmin, bj, bp, st); /* constrained, unrolled */
        if (m < bm) { bm = m; memcpy(best, st, sizeof(uint16_t) * L); }
      }
      for (int i = 0; i < L; i++) out[(size_t)t * L + i] = cb[best[i] ^ salt_of(i, kk[i])];
      if (idx) memcpy(idx + (size_t)t * L, best, sizeof(uint16_t) * L);
    }
    free(cost); free(ncost); free(gmin); free(bj); free(bp); free(st); free(best);
  }
  free(kk);
}

// decoder check: the values a 16-bit state sequence decodes to, and whether it is a valid ring (every transition and
// the wrap obey the shift rule). Returns the number of violations.
int ring_check(const uint16_t* idx, int nrings, int L, const int* pat, int plen) {
  int bad = 0;
  for (int t = 0; t < nrings; t++) {
    const uint16_t* s = idx + (size_t)t * L;
    for (int i = 0; i < L; i++) {
      const int k = pat[i % plen], prev = s[(i + L - 1) % L], cur = s[i];
      if (k == 0) bad += cur != prev;
      else bad += (cur >> k) != (prev & ((1 << (16 - k)) - 1));
    }
  }
  return bad;
}
