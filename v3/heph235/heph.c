// heph.c -- HEPH's grid-aligned tail-biting bitshift-trellis Viterbi (w235), generic L / T / shift pattern, per-element weights.
//
// A sequence is T consecutive weights of one output row (targets z_i = the MXFP4 atoms, weights w_i = diag-H branch weights).
// State s_i (L bits) decodes position i as cb[s_i ^ salt] (salt = 0 unless the step is a 0-bit step); the step into position i
// shifts kk[i] bits: s_i = ((s_{i-1} << kk[i]) | b) mod 2^L, i.e. s_i >> kk[i] == s_{i-1} & (2^(L-kk[i]) - 1). A 0-bit step keeps the
// state and reads cb[s ^ salt(i)], salt(i) = (0x9E37 * ((i & 15) + 1)) & (2^L - 1), the same rule as v3/e3/viterbi.c viterbi_tiles_pat.
// Integer K: kk[] = K everywhere. Fractional K = a/b: kk[i] = pat[i % b] (trellis.pattern).
// Branch cost w_i (cb[s ^ salt] - z_i)^2; exact Viterbi (state cost = min over the 2^kk predecessor group + branch cost).
// Tail biting = QTIP / HEPH two passes: (1) free-boundary pass on the sequence rolled by T/2 gives the group c = s_0 >> kk[0] the
// seam wants; (2) the unrolled pass starts at positions 0 with s_0 >> kk[0] == c and must end with s_{T-1} & (2^(L-kk[0]) - 1) == c.
//   gcc -O3 -march=native -fopenmp -shared -fPIC heph.c -o libheph.so
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BIGV 1e30f

static inline int salt_of(int pos, int k, int mask) { return k ? 0 : (int)((0x9E37u * (unsigned)((pos & 15) + 1)) & (unsigned)mask); }

// One pass on positions rolled by `roll` (step j = position (j + roll) % T). c < 0: free start and end; else (roll must be 0)
// s_0 >> kk[0] == c and s_{T-1} & (2^(L-kk[0]) - 1) == c. st[j] = state of step j. Returns the path cost.
static float run_pass(const float* x, const float* w, const int* kk, int T, int L, const float* cb, int roll, int c,
                      float* cost, float* ncost, float* gmin, uint8_t* bp, uint16_t* st) {
  const int NS = 1 << L, mask = NS - 1, HALF = NS >> 1;
  { const int p = roll % T, k = kk[p], sa = salt_of(p, k, mask); const float xi = x[p], wi = w[p];
    for (int s = 0; s < NS; s++) { const float d = cb[s ^ sa] - xi; cost[s] = (c >= 0 && (s >> k) != c) ? BIGV : wi * d * d; } }
  for (int j = 1; j < T; j++) {
    const int p = (j + roll) % T, k = kk[p], sa = salt_of(p, k, mask); const float xi = x[p], wi = w[p];
    if (k == 0) {
      for (int s = 0; s < NS; s++) { const float d = cb[s ^ sa] - xi; ncost[s] = cost[s] + wi * d * d; }
    } else {
      const int NG = NS >> k, SH = L - k, NJ = 1 << k, R = 1 << k; uint8_t* b = bp + (size_t)j * HALF;
      for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; b[g] = 0; }
      for (int jj = 1; jj < NJ; jj++) {
        const float* cj = cost + ((size_t)jj << SH);
        for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; b[g] = lt ? (uint8_t)jj : b[g]; }
      }
      for (int g = 0; g < NG; g++) { const float gv = gmin[g]; const float* cbg = cb + ((size_t)g << k); float* nc = ncost + ((size_t)g << k);
        for (int r = 0; r < R; r++) { const float d = cbg[r] - xi; nc[r] = gv + wi * d * d; } }
    }
    float* t = cost; cost = ncost; ncost = t;
  }
  // `cost` now holds the last step (after an odd / even number of swaps the local pointers still track it)
  const int k0 = kk[0]; const int lowmask = (1 << (L - k0)) - 1;
  int sb = -1; float m = BIGV * 2;
  for (int s = 0; s < NS; s++) { if (c >= 0 && (s & lowmask) != c) continue; if (cost[s] < m) { m = cost[s]; sb = s; } }
  if (sb < 0) return BIGV * 2;
  int s = sb;
  for (int j = T - 1; j >= 0; j--) {
    const int p = (j + roll) % T, k = kk[p]; st[j] = (uint16_t)s;
    if (j > 0 && k > 0) { const int g = s >> k; s = ((int)bp[(size_t)j * HALF + g] << (L - k)) | g; }
  }
  return m;
}

// z, w: [nseq * T]; kk: [T]; cb: [2^L]; out: reconstructed codewords [nseq * T]; st_out (nullable): final states by position;
// seq_cost (nullable): [nseq] weighted squared error of the constrained path. Returns the number of infeasible sequences.
int heph_encode(const float* z, const float* w, const int* kk, int nseq, int T, int L, const float* cb, float* out,
                uint16_t* st_out, float* seq_cost) {
  int bad = 0;
#pragma omp parallel
  {
    const int NS = 1 << L;
    float* cost = malloc(sizeof(float) * NS); float* ncost = malloc(sizeof(float) * NS); float* gmin = malloc(sizeof(float) * NS);
    uint8_t* bp = malloc((size_t)T * (NS >> 1)); uint16_t* st = malloc(sizeof(uint16_t) * T);
    const int mask = NS - 1, roll = T / 2;
#pragma omp for schedule(dynamic, 16) reduction(+ : bad)
    for (int n = 0; n < nseq; n++) {
      const float* x = z + (size_t)n * T; const float* wt = w + (size_t)n * T;
      run_pass(x, wt, kk, T, L, cb, roll, -1, cost, ncost, gmin, bp, st);
      const int j0 = (T - roll) % T, c = st[j0] >> kk[0];     // step j0 is position 0
      float m = run_pass(x, wt, kk, T, L, cb, 0, c, cost, ncost, gmin, bp, st);
      if (m >= BIGV) bad++;
      for (int j = 0; j < T; j++) { out[(size_t)n * T + j] = cb[st[j] ^ salt_of(j, kk[j], mask)]; if (st_out) st_out[(size_t)n * T + j] = st[j]; }
      if (seq_cost) seq_cost[n] = m;
    }
    free(cost); free(ncost); free(gmin); free(bp); free(st);
  }
  return bad;
}

// free (non-tail-biting) single pass on roll 0, for the brute-force tests: returns cost, fills st / out.
float heph_free_pass(const float* x, const float* w, const int* kk, int T, int L, const float* cb, int c, uint16_t* st) {
  const int NS = 1 << L;
  float* cost = malloc(sizeof(float) * NS); float* ncost = malloc(sizeof(float) * NS); float* gmin = malloc(sizeof(float) * NS);
  uint8_t* bp = malloc((size_t)T * (NS >> 1));
  float m = run_pass(x, w, kk, T, L, cb, 0, c, cost, ncost, gmin, bp, st);
  free(cost); free(ncost); free(gmin); free(bp);
  return m;
}
