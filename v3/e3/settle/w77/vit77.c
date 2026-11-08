// vit77.c -- w77 (#114) gate helper: viterbi.c's pattern trellis (fractional and sub-1 K, salted 0-bit steps) with
// the GPU quantizer's tail-biting heuristic (qtf2.cuh: a free pass on the ring rolled by L / 2 gives the node entering
// position 0, then one pass from position 0 pinned to start and end on that node), plus viterbi.c's fp16 cost
// emulation. Used only by gate77.py to compare against the GPU's numbers; the sweep itself uses settle's free start
// (ratios between dampings unaffected, settle README).
//   gcc -O3 -march=native -shared -fPIC vit77.c -o libvit77.so
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static int F16 = 0;
void vit77_set_f16(int on) { F16 = on; }
static inline float H(float v) { return F16 ? (float)(_Float16)v : v; }
static inline int salt_of(int i, int k) { return k ? 0 : (int)((0x9E37u * (unsigned)((i & 15) + 1)) & 0xffff); }

// one pass over ring positions (i + roll) & 255, i = 0..255; c < 0: free, else the node of width 16 - D(roll) that
// enters position `roll` (start: s >> D(roll) == c; end: s & (2^(16 - D(roll)) - 1) == c). st[i] = state at step i.
static float run(const float* xt, const float* cb, const int* kk, int roll, int c, float* cost, float* ncost,
                 float* gmin, uint8_t* bp, uint16_t* st) {
  const int NS = 65536, r0 = roll & 255, D0 = kk[r0];
  { const int sa = salt_of(r0, D0); const float x0 = H(xt[r0]);
    for (int s = 0; s < NS; s++) {
      const float d = H(H(cb[s ^ sa]) - x0);
      cost[s] = (c < 0 || (s >> D0) == c) ? H(d * d) : FLT_MAX;
    } }
  for (int i = 1; i < 256; i++) {
    const int ri = (i + roll) & 255, K = kk[ri], sa = salt_of(ri, K);
    const float xi = H(xt[ri]);
    if (K == 0) {
      for (int s = 0; s < NS; s++) { const float d = H(H(cb[s ^ sa]) - xi); ncost[s] = cost[s] == FLT_MAX ? FLT_MAX : H(cost[s] + d * d); }
    } else {
      const int NG = NS >> K, SH = 16 - K, NJ = 1 << K;
      uint8_t* b = bp + (size_t)i * (NS >> 1);
      for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; b[g] = 0; }
      for (int j = 1; j < NJ; j++) {
        const float* cj = cost + (j << SH);
        for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; b[g] = lt ? (uint8_t)j : b[g]; }
      }
      for (int s = 0; s < NS; s++) { const float d = H(H(cb[s]) - xi); ncost[s] = gmin[s >> K] == FLT_MAX ? FLT_MAX : H(gmin[s >> K] + d * d); }
    }
    float* t = cost; cost = ncost; ncost = t;
  }
  const int mask = (1 << (16 - D0)) - 1;
  int sb = -1; float m = FLT_MAX;
  for (int s = 0; s < NS; s++) if ((c < 0 || (s & mask) == c) && cost[s] < m) { m = cost[s]; sb = s; }
  if (sb < 0) { sb = c < 0 ? 0 : c; }
  st[255] = (uint16_t)sb;
  for (int i = 255; i > 0; i--) {
    const int K = kk[(i + roll) & 255]; int s = st[i];
    if (K) { const int g = s >> K; s = g | ((int)bp[(size_t)i * (NS >> 1) + g] << (16 - K)); }
    st[i - 1] = (uint16_t)s;
  }
  return m;
}

// x [ntiles][256] -> out (values) and idx (states), pattern pat[i % plen] bits at position i; tb: 1 = GPU heuristic
void vit77_tiles(const float* x, float* out, uint16_t* idx, int ntiles, const float* cb, const int* pat, int plen, int tb) {
  int kk[256];
  for (int i = 0; i < 256; i++) kk[i] = pat[i % plen];
  float* cost = malloc(sizeof(float) * 65536); float* ncost = malloc(sizeof(float) * 65536);
  float* gmin = malloc(sizeof(float) * 65536); uint8_t* bp = malloc((size_t)256 * 32768);
  uint16_t st[256];
  for (int t = 0; t < ntiles; t++) {
    const float* xt = x + (size_t)t * 256;
    if (tb) {
      run(xt, cb, kk, 128, -1, cost, ncost, gmin, bp, st);
      const int c = st[128] >> kk[0];                       // step 128 of the rolled pass = ring position 0
      run(xt, cb, kk, 0, c, cost, ncost, gmin, bp, st);
    } else run(xt, cb, kk, 0, -1, cost, ncost, gmin, bp, st);
    for (int i = 0; i < 256; i++) {
      out[(size_t)t * 256 + i] = cb[st[i] ^ salt_of(i, kk[i])];
      if (idx) idx[(size_t)t * 256 + i] = st[i];
    }
  }
  free(cost); free(ncost); free(gmin); free(bp);
}
