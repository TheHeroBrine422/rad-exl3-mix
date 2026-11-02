// test_decode.hip -- exl3i_decode_frag on the GPU against the oracle: every fractional / sub-1 rate of the V3 menu
// (KA 0-4) and both codebooks, 256 random tiles each; one wave a tile, every lane decodes its fragment, positions
// scattered back to ring order, compared bit for bit with exl3i_decode_tile. Build (GPU box):
//   hipcc -O3 --offload-arch=gfx1201 -ffp-contract=off -I<plugin>/exl3r4d -I<radiance>/include test_decode.hip ../ref/exl3i_ref.c
#include "exl3r4d_common.hip.h"
#include "exl3i_decode.hip.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
int exl3i_frac_mask(int n);
int exl3i_decode_tile(const uint32_t* words, int ka, int mask, int i8, uint16_t* states, uint16_t* values);
}

template <int CB, int KA, unsigned MASK>
__global__ void k_decode(const unsigned* words, unsigned short* out) {   // out [tile][256] fp16 bits, ring order
  constexpr int W = Exl3iLanePlan<KA, MASK>::WORDS;
  const int lane = threadIdx.x, tile = blockIdx.x;
  const Exl3iLanePlan<KA, MASK> P(lane);
  const unsigned* tw = words + (size_t)tile * W;
  unsigned w[2][2];
  for (int r = 0; r < 2; r++) for (int x = 0; x < 2; x++) w[r][x] = tw[P.woff[r][x]];
  union { v8h f; unsigned short h[8]; } u;
  u.f = exl3i_decode_frag<CB, KA, MASK>(w, P);
  const int n = lane & 15, hh = lane >> 4, base = 32 * (n & 7) + 4 * (n >> 3) + 16 * hh;
  // element e: run r = (e >> 1) & 1, position within the run (e & 1) + 2 (e >> 2)
  for (int e = 0; e < 8; e++) {
    const int r = (e >> 1) & 1, i = (e & 1) + 2 * (e >> 2);
    out[(size_t)tile * 256 + base + 8 * r + i] = u.h[e];
  }
}

template <int CB, int KA, unsigned MASK>
static int run1() {
  constexpr int W = Exl3iLanePlan<KA, MASK>::WORDS, T = 256;
  std::vector<unsigned> h(W * T);
  for (auto& x : h) x = ((unsigned)rand() << 16) ^ (unsigned)rand();
  unsigned* dw; unsigned short* dout;
  hipMalloc(&dw, h.size() * 4); hipMalloc(&dout, T * 256 * 2);
  hipMemcpy(dw, h.data(), h.size() * 4, hipMemcpyHostToDevice);
  k_decode<CB, KA, MASK><<<T, 32>>>(dw, dout);
  std::vector<unsigned short> o(T * 256);
  hipMemcpy(o.data(), dout, o.size() * 2, hipMemcpyDeviceToHost);
  hipFree(dw); hipFree(dout);
  int bad = 0;
  for (int t = 0; t < T; t++) {
    uint16_t st[256], va[256];
    exl3i_decode_tile(&h[(size_t)t * W], KA, (int)MASK, CB, st, va);
    for (int p = 0; p < 256; p++) bad += o[(size_t)t * 256 + p] != va[p];
  }
  printf("cb %d KA %d mask %04x: %s (%d values differ)\n", CB, KA, MASK, bad ? "FAIL" : "ok", bad);
  return bad;
}

template <int CB, int KA>
static int rates() {   // the seven even fractions of KA (KA = 0: all sub-1 rates)
  return run1<CB, KA, 0x8080u>() + run1<CB, KA, 0x8888u>() + run1<CB, KA, 0xa4a4u>() + run1<CB, KA, 0xaaaau>() +
         run1<CB, KA, 0xdadau>() + run1<CB, KA, 0xeeeeu>() + run1<CB, KA, 0xfefeu>();
}

int main() {
  srand(3);
  int bad = 0;
  bad += rates<0, 0>() + rates<0, 1>() + rates<0, 2>() + rates<0, 3>() + rates<0, 4>();
  bad += rates<1, 0>() + rates<1, 1>() + rates<1, 2>() + rates<1, 3>() + rates<1, 4>();
  printf("DECODE %s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
