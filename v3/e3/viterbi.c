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
      { const int sa = salt_of(0, kk[0]); for (int s = 0; s < NS; s++) { const float d = xt[0] - cb[s ^ sa]; cost[s] = d * d; } }
      for (int i = 1; i < 256; i++) {
        const int K = kk[i], sa = salt_of(i, K);
        const float xi = xt[i];
        if (K == 0) {
          for (int s = 0; s < NS; s++) { const float d = xi - cb[s ^ sa]; ncost[s] = cost[s] + d * d; }
        } else {
          const int NG = NS >> K, SH = 16 - K, NJ = 1 << K;
          uint8_t* b = bp + (size_t)i * (NS >> 1);
          for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; b[g] = 0; }
          for (int j = 1; j < NJ; j++) {
            const float* cj = cost + (j << SH);
            for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; b[g] = lt ? (uint8_t)j : b[g]; }
          }
          for (int s = 0; s < NS; s++) { const float d = xi - cb[s]; ncost[s] = gmin[s >> K] + d * d; }
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
