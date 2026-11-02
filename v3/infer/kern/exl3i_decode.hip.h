// exl3i_decode.hip.h -- EXL3-V3 fragment decode for the exl3i plugin: exl3r4d's decode_frag generalized to fractional
// rates (K = KA + n/16, KA <= 4) and sub-1 (salted 0-bit reads), for both codebooks (mul1, mul1i8). Integer K keeps
// exl3r4d's Plan<BITS> / decode_run (its tuned fast paths); this file is the fractional path.
// Bit plan: exl3i_plan.h (host-verified against the oracle, v3/infer/ref). Values: the oracle's exl3i_codebook --
//   mul1:   v = fp16 fma(fp16(1024 + bytesum(state * 0x83DCD12D)), K_INV, K_BIAS)
//   mul1i8: fp16(clamp(rint(f32(v) / (4 K_INV)), +-127) * 4 K_INV)   (what the quantizer searched with)
// Needs exl3r4d_common.hip.h first (v2h / v8h, sad_lo / sad_hi, hash_pair, EXL3_HASH).
#pragma once
#include "exl3i_plan.h"

enum Exl3iCb { EXL3I_CB_MUL1 = 0, EXL3I_CB_MUL1I8 = 1 };

// int8 rounding of a pair of mul1 values, exactly codebook.cuh's mul1_i8_round (multiply by inv, rint, clamp, fp16)
static __device__ __forceinline__ unsigned exl3i_i8_pair(unsigned p) {
  const float s4 = 4.0f * (float)__builtin_bit_cast(_Float16, (unsigned short)0x1EEE), inv = 1.0f / s4;
  const v2h v = __builtin_bit_cast(v2h, p);
  const float q0 = fminf(fmaxf(rintf((float)v[0] * inv), -127.f), 127.f);
  const float q1 = fminf(fmaxf(rintf((float)v[1] * inv), -127.f), 127.f);
  return __builtin_bit_cast(unsigned, (v2h){(_Float16)(q0 * s4), (_Float16)(q1 * s4)});
}

// two states (s0 low, s1 high) -> fp16 pair, the same arithmetic as exl3r4d's K = 4 / R23 form
template <int CB>
static __device__ __forceinline__ unsigned exl3i_value_pair(unsigned s0, unsigned s1) {
  unsigned h0, h1;
  hash_pair(s0 | s1 << 16, h0, h1);
  const _Float16 kinv = __builtin_bit_cast(_Float16, (unsigned short)0x1EEE);
  const _Float16 kbias = __builtin_bit_cast(_Float16, (unsigned short)0xC931);
  const unsigned a = sad_hi(h1, sad_lo(h0, 0x64006400u));
  const unsigned p = __builtin_bit_cast(unsigned, __builtin_elementwise_fma(__builtin_bit_cast(v2h, a), (v2h){kinv, kinv},
                                                                            (v2h){kbias, kbias}));
  if constexpr (CB == EXL3I_CB_MUL1I8) return exl3i_i8_pair(p);
  return p;
}

// One run of four positions from its two words: lo = the 32 ring bits ending at the run's e3, state i = lo[off_i, +16),
// XOR the position's salt (0 unless a 0-bit sub-1 step). SALTED is false for KA >= 1 (no 0-bit steps): no XOR at all.
template <int CB, bool SALTED>
static __device__ __forceinline__ void exl3i_decode_run(unsigned w0, unsigned w1, unsigned shift, const unsigned (&off)[4],
                                                        const unsigned (&xr)[4], unsigned& p01, unsigned& p23) {
  const unsigned lo = __builtin_amdgcn_alignbit(w0, w1, shift);
  unsigned st[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    st[i] = __builtin_amdgcn_ubfe(lo, off[i], 16);
    if constexpr (SALTED) st[i] ^= xr[i];
  }
  p01 = exl3i_value_pair<CB>(st[0], st[1]);
  p23 = exl3i_value_pair<CB>(st[2], st[3]);
}

// The lane's plan, kept in registers by the kernel (built once per lane, as exl3r4d's Plan)
template <int KA, unsigned MASK>
struct Exl3iLanePlan {
  static constexpr bool SALTED = KA == 0;
  static constexpr int WORDS = (16 * KA + __builtin_popcount(MASK)) / 2;
  int woff[2][2];
  unsigned shift[2], off[2][4], xr[2][4];
  __device__ __forceinline__ explicit Exl3iLanePlan(int lane) {
    Exl3iPlan P;
    exl3i_plan(&P, lane, KA, MASK);
#pragma unroll
    for (int r = 0; r < 2; r++) {
      woff[r][0] = P.woff[r][0]; woff[r][1] = P.woff[r][1]; shift[r] = P.shift[r];
#pragma unroll
      for (int i = 0; i < 4; i++) { off[r][i] = P.off[r][i]; xr[r][i] = P.xr[r][i]; }
    }
  }
};

// The lane's 16-bit A fragment of one tile (element order as exl3r4d's decode_frag): w[r][x] = the tile's words
// woff[r][x], loaded by the caller (tile base + 4 woff bytes, as exl3r4d's kernels do).
template <int CB, int KA, unsigned MASK>
static __device__ __forceinline__ v8h exl3i_decode_frag(const unsigned (&w)[2][2], const Exl3iLanePlan<KA, MASK>& P) {
  union { unsigned u[4]; v8h f; } x;
  exl3i_decode_run<CB, Exl3iLanePlan<KA, MASK>::SALTED>(w[0][0], w[0][1], P.shift[0], P.off[0], P.xr[0], x.u[0], x.u[2]);
  exl3i_decode_run<CB, Exl3iLanePlan<KA, MASK>::SALTED>(w[1][0], w[1][1], P.shift[1], P.off[1], P.xr[1], x.u[1], x.u[3]);
  return x.f;
}
