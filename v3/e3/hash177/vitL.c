// vitL.c -- viterbi.c's viterbi_tiles (bitshift trellis over 16x16 tiles, free initial state, plain squared error) with
// the state width L a parameter (16 = viterbi.c's exactly; 17 / 18 for #177's longer-state variants) and, optionally, the
// chosen state of every position (for Lloyd fits of LUT codebooks). State s: L bits; a step shifts in K bits,
// s' = ((s << K) | x) mod 2^L, emitting cb[s'].  (w139, #177)
//   gcc -O3 -march=native -fopenmp -shared -fPIC vitL.c -o libvitL.so
#include <stdint.h>
#include <stdlib.h>
#include <float.h>

// x: [ntiles][256] targets, out: [ntiles][256] values, st (may be NULL): [ntiles][256] states, cb: 2^L values
void viterbi_tiles_L(const float* x, float* out, int32_t* st, int ntiles, const float* cb, int K, int L) {
  const int NS = 1 << L, NG = NS >> K, SH = L - K, NJ = 1 << K;
#pragma omp parallel
  {
    float* cost = malloc(sizeof(float) * NS);
    float* ncost = malloc(sizeof(float) * NS);
    float* gmin = malloc(sizeof(float) * NG);
    uint8_t* bp = malloc((size_t)256 * NG);
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256;
      for (int s = 0; s < NS; s++) { const float d = xt[0] - cb[s]; cost[s] = d * d; }
      for (int i = 1; i < 256; i++) {
        uint8_t* b = bp + (size_t)i * NG;
        for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; b[g] = 0; }
        for (int j = 1; j < NJ; j++) {
          const float* cj = cost + ((size_t)j << SH);
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
        if (st) st[(size_t)t * 256 + i] = s;
        if (i) { const int g = s >> K; s = g | ((int)bp[(size_t)i * NG + g] << SH); }
      }
    }
    free(cost); free(ncost); free(gmin); free(bp);
  }
}
