// cb208.hip -- w208 (2026-11-20): what a codebook's decode costs inside an int8 (w8a8) / int4 (w4a4) expert GEMV and a
// WMMA prefill GEMM on gfx1201, integer and fractional K, against the same kernel without a codebook. See README.md.
//
// One binary a candidate: hipcc -O3 --offload-arch=gfx1201 -DCAND=Mad cb208.hip -o cb208_Mad   (build.sh builds all)
//
// Stream (the exl3i ring, plugin exl3i `common/exl3i_decode.h`, = v3/e3 viterbi_tiles_pat's trellis): a tile is 256
// positions at rate K = MM / 16, MM / 2 u32 words, one MSB-first bit ring of 16 MM bits (word w = ring bits
// [32 w, 32 w + 32), bit 31 first). Position i takes D(i) = floor((i%16 + 1) MM / 16) - floor((i%16) MM / 16) new bits
// (trellis.py `pattern`); its 16-bit state is the ring bits [S(i) - 16, S(i)), S(i) = D(0) + .. + D(i), circular; a
// 0-bit step reads the codebook at state ^ salt(i), salt(i) = 0x9E37 (i%16 + 1) mod 2^16 (viterbi.c salt_of).
// Scan (this bench's own; a quantizer would encode in it): position i = tile row i / 16 (output), column i % 16 (input).
//
// GEMV (M = 1, 2, 4, 8 tokens): a lane owns one tile row (16 positions = MM bits + 16 bits of look-back), a wave two
// tiles a step; activations int8 (4 a dword) or int4 (8 a dword) for M tokens interleaved; one sudot4 / sudot8 a packed
// weight dword a token; int32 sums over the whole row; KW waves split the columns (reduced through LDS).
// GEMM (M = 256): a lane owns half a tile row (the gfx12 WMMA A fragment), one wave decodes a tile and multiplies it
// into 16 / TS token tiles (iu8 16x16x16 or iu4 16x16x16), activation fragments straight from global.
// Weights are decoded as unsigned grid indices n (value = n - BIAS2 / 2); the epilogue subtracts BIAS2 / 2 * sum(x).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "tables.h"

typedef uint32_t u32;
typedef uint64_t u64;
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define DEV static __device__ __forceinline__
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------- the fractional-K pattern (compile time)
template <int MM> struct Pat {
  static constexpr int KA = MM / 16, NX = MM % 16, TW = MM / 2;
  static constexpr int D(int i) { i &= 15; return KA + ((i + 1) * NX) / 16 - (i * NX) / 16; }
  static constexpr int S(int i) { int s = 0; for (int j = 0; j < i; j++) s += D(j); return s; }   // bits of positions < i
  static constexpr int NA = (16 + MM + 31) / 32;        // aligned words of a row lane (look-back + 16 positions)
  static constexpr int NA8 = (16 + MM / 2 + 31) / 32;   // of a half-row lane
};
static constexpr int KCEIL(int MM) { return (MM + 15) / 16; }

// the eight states of a run from its 64-bit window (hi, lo): window bit 0 (MSB of hi) = 16 bits before the run's first
// position; state c ends S(c + 1) bits later. sbase = the run's first column (0 or 8; the salt's position).
// ST8 (the CPTC family): the state is taken by a v_bfe_u32 whose offset is opaque to the compiler (constant + z, z a
// run-time zero), so the extract survives as a bit-field node with 16 known bits and the 16 x 16 product selects
// v_mul_u32_u24. A plain shift + mask does not: a CPTC index needs only the product's low 16 bits, so LLVM drops the
// state's mask as redundant, the operand is then a full 32-bit value and half the products of a run became
// v_mul_lo_u32 (the full multiply; measured cost in RESULTS.md). The offsets are loop-invariant registers.
template <int MM, bool ST8> DEV void states(u32 hi, u32 lo, u32 sbase, u32 z, u32 (&st)[8]) {
  const u32 mid = (hi << 16) | (lo >> 16);
#pragma unroll
  for (int c = 0; c < 8; c++) {
    const int k = 48 - Pat<MM>::S(c + 1);   // st = (window >> k) & 0xFFFF, 16 <= k <= 48
    if constexpr (ST8) {
      if (k == 48) st[c] = hi >> 16;
      else st[c] = k >= 32 ? __builtin_amdgcn_ubfe(hi, (u32)(k - 32) + z, 16u) : __builtin_amdgcn_ubfe(mid, (u32)(k - 16) + z, 16u);
    } else st[c] = k >= 32 ? (hi >> (k - 32)) & 0xFFFFu : (mid >> (k - 16)) & 0xFFFFu;
    if (Pat<MM>::D(c) == 0) st[c] ^= (0x9E37u * (sbase + c + 1u)) & 0xFFFFu;
  }
}

// ---------------------------------------------------------------- candidates
struct NoCtx { __device__ NoCtx(const u32*, const u32*) {} };
template <bool LDS> struct TabCtx {
  const uint8_t* t;
  __device__ TabCtx(const u32* g, const u32* l) : t((const uint8_t*)(LDS ? l : g)) {}
};
struct CBDef { static constexpr bool FOLD = false, RAW = false, SGN = false, ST8 = false; static constexpr int CUSTOM = 0, TABB = 0, TABG = 0; };
#define CB_COMMON(nm, pl) static constexpr const char* NAME = nm; static constexpr const char* PLACE = pl;
static u32 h_bytesum(u32 h) { return (h & 255) + (h >> 8 & 255) + (h >> 16 & 255) + (h >> 24); }

// baseline: the trellis without a codebook (shift + mask), the paper's INT4-baseline analogue
struct Base : CBDef {
  CB_COMMON("base", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return true; }
  static constexpr int bias2(int) { return 0; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) { return st & (A4 ? 15u : 255u); }
  static void host(int, int a4, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = s & (a4 ? 15 : 255); }
};
// raw: no per-value work at all, the window words are the packed weights (loads + dots only)
struct Raw : Base { static constexpr const char* NAME = "raw"; static constexpr bool RAW = true; };

#define MAD_A 0xD0474Bu
#define MAD_B 0x2545F491u
struct Mad : CBDef {   // production: bytesum((s A + B) & 0x3F3F3F3F) - 126; n = S + 2, BIAS 128
  CB_COMMON("mad", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    return __builtin_amdgcn_udot4((st * MAD_A + MAD_B) & 0x3F3F3F3Fu, 0x01010101u, 2u, false);
  }
  static void host(int, int, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = h_bytesum((s * MAD_A + MAD_B) & 0x3F3F3F3Fu) + 2; }
};
struct MadF : CBDef {   // mad folded into the dot (madgemv #177): sudot4(h, x * 0x01010101) = x * bytesum(h); n = S, BIAS 126
  CB_COMMON("mad", "fold") typedef NoCtx Ctx;
  static constexpr bool FOLD = true;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 252; }
  DEV u32 hash(u32 st) { return (st * MAD_A + MAD_B) & 0x3F3F3F3Fu; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) { return __builtin_amdgcn_udot4(hash(st), 0x01010101u, 0u, false); }
  static void host(int, int, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = h_bytesum((s * MAD_A + MAD_B) & 0x3F3F3F3Fu); }
};
// mad4 (this bench's proposal): the same hash, three top fields of 3 + 3 + 1 bits summed by one udot8:
//   n = ((h >> 28) & 7) + ((h >> 24) & 7) + ((h >> 20) & 1), h = (s A + B) mod 2^32;  value = n - 8 in [-8, 7] (16 levels)
#define MAD4_MASK 0x77100000u
#define MAD4_ONES 0x11100000u
struct Mad4 : CBDef {
  CB_COMMON("mad4", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return true; }
  static constexpr int bias2(int) { return 16; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    return __builtin_amdgcn_udot8((st * MAD_A + MAD_B) & MAD4_MASK, MAD4_ONES, 0u, false);
  }
  static void host(int, int, uint8_t* n) {
    for (u32 s = 0; s < 65536; s++) { const u32 h = s * MAD_A + MAD_B; n[s] = ((h >> 28) & 7) + ((h >> 24) & 7) + ((h >> 20) & 1); }
  }
};
struct Mad4F : CBDef {   // mad4 folded: sudot8(h & mask, x * 0x11100000) = x * n  (w4a4 only; x as a replicated nibble a column)
  CB_COMMON("mad4", "fold") typedef NoCtx Ctx;
  static constexpr bool FOLD = true;
  static constexpr bool i4(int) { return true; }
  static constexpr int bias2(int) { return 16; }
  DEV u32 hash(u32 st) { return (st * MAD_A + MAD_B) & MAD4_MASK; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) { return __builtin_amdgcn_udot8(hash(st), MAD4_ONES, 0u, false); }
  static void host(int mm, int a4, uint8_t* n) { Mad4::host(mm, a4, n); }
};

// host: the standardised mul1 byte sum of trellis.py `codebook`
static void mul1_std(std::vector<double>& v) {
  v.resize(65536); double m = 0, q = 0;
  for (u32 s = 0; s < 65536; s++) { v[s] = h_bytesum(s * 0x83DCD12Du); m += v[s]; }
  m /= 65536; for (u32 s = 0; s < 65536; s++) q += (v[s] - m) * (v[s] - m);
  q = sqrt(q / 65536); for (u32 s = 0; s < 65536; s++) v[s] = (v[s] - m) / q;
}
struct Mul1I8 : CBDef {   // exllamav3 mul1 on the int8 grid (+-127 = 3.5 sd): n = q + 128
  CB_COMMON("mul1_i8", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    const u32 bs = __builtin_amdgcn_udot4(st * 0x83DCD12Du, 0x01010101u, 0u, false) & 1023u;   // the mask lets the map be a mad24
    return (bs * MUL1I8_MUL + MUL1I8_ADD) >> MUL1I8_SH;
  }
  static void host(int, int, uint8_t* n) {
    std::vector<double> v; mul1_std(v);
    for (u32 s = 0; s < 65536; s++) n[s] = (uint8_t)(std::min(127.0, std::max(-127.0, nearbyint(v[s] / (3.5 / 127)))) + 128);
  }
};
struct Int4Grid : CBDef {   // mul1 on 16 levels (+-7.5 = 2.6 sd): n = level index 0..15, value = n - 7.5
  CB_COMMON("int4grid", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return true; }
  static constexpr int bias2(int) { return 15; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    const int bs = (int)(__builtin_amdgcn_udot4(st * 0x83DCD12Du, 0x01010101u, 0u, false) & 1023u);
    const int t = (bs * I4G_MUL + I4G_ADD) >> I4G_SH;
    return (u32)(t < 0 ? 0 : t > 15 ? 15 : t);
  }
  static void host(int, int, uint8_t* n) {
    std::vector<double> v; mul1_std(v);
    for (u32 s = 0; s < 65536; s++)
      n[s] = (uint8_t)(std::min(7.5, std::max(-7.5, nearbyint(v[s] / (2.6 / 7.5) - 0.5) + 0.5)) + 7.5);
  }
};
struct Qtip1Mad : CBDef {   // QTIP 1MAD: (bytesum(s * 34038481 + 76625530) - 510) / 147.8005, on the int8 grid
  CB_COMMON("qtip_1mad", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    const u32 bs = __builtin_amdgcn_udot4(st * 34038481u + 76625530u, 0x01010101u, 0u, false) & 1023u;
    return (bs * Q1MAD_MUL + Q1MAD_ADD) >> Q1MAD_SH;
  }
  static void host(int, int, uint8_t* n) {
    for (u32 s = 0; s < 65536; s++) {
      const double v = ((double)h_bytesum(s * 34038481u + 76625530u) - 510.0) / 147.800537109375;
      n[s] = (uint8_t)(std::min(127.0, std::max(-127.0, nearbyint(v / (3.5 / 127)))) + 128);
    }
  }
};
struct Qtip3Inst : CBDef {   // QTIP 3INST: x = s * 89226354 + 64248484; fp16(lo ^ m) + fp16(hi ^ m) (half arithmetic), int8 grid
  CB_COMMON("qtip_3inst", "-") typedef NoCtx Ctx;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  static __host__ __device__ inline float sum16(u32 st) {
    const u32 r = ((st * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 996162400u;
    const _Float16 a = __builtin_bit_cast(_Float16, (unsigned short)(r >> 16)), b = __builtin_bit_cast(_Float16, (unsigned short)r);
    return (float)(_Float16)(a + b);
  }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    return (u32)((int)__builtin_rintf(sum16(st) * Q3INST_C) + 128);
  }
  static void host(int, int, uint8_t* n) {
    double m = 0, q = 0; std::vector<float> r(65536);
    for (u32 s = 0; s < 65536; s++) { r[s] = sum16(s); m += r[s]; }
    m /= 65536; for (u32 s = 0; s < 65536; s++) q += (r[s] - m) * (r[s] - m);
    const float c = (float)(127.0 / (3.5 * sqrt(q / 65536)));
    if (c != Q3INST_C) { printf("qtip_3inst: host scale %a != tables.h %a\n", c, (float)Q3INST_C); exit(1); }
    for (u32 s = 0; s < 65536; s++) n[s] = (uint8_t)((int)nearbyintf(r[s] * c) + 128);
  }
};

// CPTC (arXiv 2611.22019), the unified L=16 pair; state products are u24 multiplies (16 x 16 bits)
// Plain shift + mask, NOT __builtin_amdgcn_ubfe: LLVM rewrites a constant ubfe as shl + lshr and folds the shl into the
// multiplier ((s * (a << 16)) >> 26), which no longer fits the 24-bit multiply and selects v_mul_lo_u32 -- measured
// 1.2-1.3x slower GEMV for cptc5 on gfx1201. This form keeps v_mul_u32_u24 + v_bfe_u32.
DEV u32 top_q(u32 prod, int Q) { return (prod >> (16 - Q)) & ((1u << Q) - 1u); }
template <int Q> DEV u32 cptc_idx(u32 st) { return top_q(st * CPTC_A0, Q) + top_q(st * CPTC_A1, Q); }
static u32 cptc_idx_h(u32 s, int Q) { return (((s * CPTC_A0) & 0xFFFF) >> (16 - Q)) + (((s * CPTC_A1) & 0xFFFF) >> (16 - Q)); }
template <bool LDS> struct Cptc6 : CBDef {   // Q = 6, 64-entry lower half, mirror fold (Algorithm 1)
  CB_COMMON("cptc6", LDS ? "lds" : "glb") typedef TabCtx<LDS> Ctx;
  static constexpr bool ST8 = true;
  static constexpr bool SGN = true;
  static constexpr int TABB = LDS ? 64 : 0, TABG = 64;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  static const void* tab(int kc) { return T6[kc - 1]; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx& c) {
    const u32 i = cptc_idx<6>(st), j = i < 126u - i ? i : 126u - i;
    const int v = (int)(int8_t)c.t[j], m = (int)(i >> 6);
    return (u32)((v ^ -m) + m);
  }
  static void host(int mm, int, uint8_t* n) {
    for (u32 s = 0; s < 65536; s++) { const u32 i = cptc_idx_h(s, 6); n[s] = (uint8_t)((i < 64 ? T6[KCEIL(mm) - 1][i] : -T6[KCEIL(mm) - 1][126 - i]) + 128); }
  }
};
// the same, products by the paper's Eq. (3) recurrence inside a run: x_c = (x_{c-1} << D) + a * code_c (a D-bit x 16 product)
struct Cptc6Eq3 : Cptc6<true> {
  static constexpr const char* PLACE = "lds+eq3";
  static constexpr int CUSTOM = 1;
  template <int MM> DEV void dec8(const u32 (&st)[8], const Ctx& c, u32 (&v)[8]) {
    u32 x0 = 0, x1 = 0;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int D = Pat<MM>::D(e);
      u32 i;
      if (D == 0) i = cptc_idx<6>(st[e]);   // a salted read: direct, the chain keeps the unsalted state
      else {
        bool first = true;   // the first unsalted state of the run starts the chain with two full products
        for (int f = 0; f < e; f++) first = first && Pat<MM>::D(f) == 0;
        if (first) { x0 = st[e] * CPTC_A0; x1 = st[e] * CPTC_A1; }
        else { const u32 code = st[e] & ((1u << D) - 1u); x0 = (x0 << D) + code * CPTC_A0; x1 = (x1 << D) + code * CPTC_A1; }
        i = top_q(x0, 6) + top_q(x1, 6);
      }
      const u32 j = i < 126u - i ? i : 126u - i;
      const int t = (int)(int8_t)c.t[j], m = (int)(i >> 6);
      v[e] = (u32)((t ^ -m) + m);
    }
  }
};
template <bool LDS> struct Cptc6F : CBDef {   // Q = 6, the 127 entries stored in full (no fold)
  CB_COMMON("cptc6full", LDS ? "lds" : "glb") typedef TabCtx<LDS> Ctx;
  static constexpr bool ST8 = true;
  static constexpr int TABB = LDS ? 128 : 0, TABG = 127;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  static const void* tab(int kc) { return T6F[kc - 1]; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx& c) { return c.t[cptc_idx<6>(st)]; }
  static void host(int mm, int, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = T6F[KCEIL(mm) - 1][cptc_idx_h(s, 6)]; }
};
template <bool LDS> struct Cptc5 : CBDef {   // Q = 5, 63 entries in full
  CB_COMMON("cptc5", LDS ? "lds" : "glb") typedef TabCtx<LDS> Ctx;
  static constexpr bool ST8 = true;
  static constexpr int TABB = LDS ? 64 : 0, TABG = 64;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  static const void* tab(int kc) { return T5[kc - 1]; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx& c) { return c.t[cptc_idx<5>(st)]; }
  static void host(int mm, int, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = T5[KCEIL(mm) - 1][cptc_idx_h(s, 5)]; }
};
struct RegCtx {   // the 64 table bytes in 16 registers
  u32 r[16];
  __device__ RegCtx(const u32* g, const u32*) {
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = g[i];
  }
};
struct Cptc5Reg : Cptc5<false> {   // table in registers: 8 byte permutes (8 entries each) + a 3-level byte select, 4 values at once
  static constexpr const char* PLACE = "reg";
  typedef RegCtx Ctx;
  static constexpr int CUSTOM = 2;
  DEV u32 look4(u32 I, const Ctx& c) {
    const u32 sel = I & 0x07070707u;
    u32 r[8];
#pragma unroll
    for (int k = 0; k < 8; k++) r[k] = __builtin_amdgcn_perm(c.r[2 * k + 1], c.r[2 * k], sel);
#pragma unroll
    for (int b = 0; b < 3; b++) {
      const u32 m = ((I >> (3 + b)) & 0x01010101u) * 0xFFu;
#pragma unroll
      for (int k = 0; k < (4 >> b); k++) r[k] = (r[2 * k + 1] & m) | (r[2 * k] & ~m);
    }
    return r[0];
  }
  template <int MM, int A4> DEV void pk8(const u32 (&st)[8], const Ctx& c, u32 (&P)[2]) {
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const u32 I = cptc_idx<5>(st[4 * q]) + (cptc_idx<5>(st[4 * q + 1]) << 8) + (cptc_idx<5>(st[4 * q + 2]) << 16) +
                    (cptc_idx<5>(st[4 * q + 3]) << 24);
      P[q] = look4(I, c);
    }
  }
};
template <int L, bool LDS> struct CptcSt : CBDef {   // state table: the codebook indexed by the low L state bits
  CB_COMMON(L == 6 ? "cptcst6" : L == 8 ? "cptcst8" : L == 10 ? "cptcst10" : "cptcst12", LDS ? "lds" : "glb") typedef TabCtx<LDS> Ctx;
  static constexpr int TABB = LDS ? (1 << L) : 0, TABG = 1 << L;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  // L 6 and 8 are locality probes of this bench (the first 2^L bytes of the L 10 table): same decode, smaller table
  static const void* tab(int kc) { return L <= 10 ? (const void*)ST10[kc - 1] : (const void*)ST12[kc - 1]; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx& c) { return c.t[st & ((1u << L) - 1u)]; }
  static void host(int mm, int, uint8_t* n) {
    for (u32 s = 0; s < 65536; s++) n[s] = L <= 10 ? ST10[KCEIL(mm) - 1][s & ((1u << L) - 1u)] : ST12[KCEIL(mm) - 1][s & 4095];
  }
};
struct CptcMin : CBDef {   // minimal codebook: Q = ceil(K) + 1, the 2^(Q+1) - 1 levels are the index itself (no table); int4 for K <= 2
  CB_COMMON("cptcmin", "-") typedef NoCtx Ctx;
  static constexpr bool ST8 = true;
  static constexpr int Q(int mm) { return KCEIL(mm) + 1; }
  static constexpr bool i4(int mm) { return mm <= 32; }
  static constexpr int bias2(int mm) { return 2 * ((1 << Q(mm)) - 1); }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx&) {
    return top_q(st * CPTC_A0, Q(MM)) + top_q(st * CPTC_A1, Q(MM));
  }
  static void host(int mm, int, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = cptc_idx_h(s, Q(mm)); }
};
template <bool LDS> struct Lc1 : CBDef {   // one multiply (a0 = 0): cell = top 6 bits of a1 s mod 2^16, 64-entry table, no fold
  CB_COMMON("lc1", LDS ? "lds" : "glb") typedef TabCtx<LDS> Ctx;
  static constexpr bool ST8 = true;
  static constexpr int TABB = LDS ? 64 : 0, TABG = 64;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  static const void* tab(int kc) { return TLC[kc - 1]; }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx& c) { return c.t[top_q(st * CPTC_A1, 6)]; }
  static void host(int mm, int, uint8_t* n) { for (u32 s = 0; s < 65536; s++) n[s] = TLC[KCEIL(mm) - 1][((s * CPTC_A1) & 0xFFFF) >> 10]; }
};
typedef Cptc6<true> Cptc6Lds; typedef Cptc6<false> Cptc6Glb; typedef Cptc6F<true> Cptc6FLds; typedef Cptc6F<false> Cptc6FGlb;
// CPTC with both 16 x 16 products in ONE packed multiply (lead 11-21): v_pk_mul_lo_u16 of the state broadcast to both halves
// times (a0, a1); the top Q bits of each half lie in bytes 1 and 3, so a mask + udot4 sums them, already scaled by 2^(8-Q)
// (x4 for Q = 6, x8 for Q = 5): the table is laid out at that stride, so the dot's result IS the byte address. 3 ALU + 1 load.
typedef unsigned short pk_us2 __attribute__((ext_vector_type(2)));
template <int Q> DEV u32 cptc_addr_pk(u32 st) {
  const pk_us2 s2 = (pk_us2)((unsigned short)st);
  const pk_us2 c = {(unsigned short)CPTC_A0, (unsigned short)CPTC_A1};
  const pk_us2 p = s2 * c;
  const u32 m = __builtin_bit_cast(u32, p) & (Q == 6 ? 0xFC00FC00u : 0xF800F800u);
  return __builtin_amdgcn_udot4(m, 0x01000100u, 0u, false);
}
template <int Q, bool LDS> struct CptcPk : CBDef {
  CB_COMMON(Q == 6 ? "cptc6pk" : "cptc5pk", LDS ? "lds" : "glb") typedef TabCtx<LDS> Ctx;
  static constexpr bool ST8 = true;
  static constexpr int STR = 1 << (8 - Q), NE = (1 << (Q + 1)) - 1;
  static constexpr int TABB = LDS ? 512 : 0, TABG = 512;
  static constexpr bool i4(int) { return false; }
  static constexpr int bias2(int) { return 256; }
  static const void* tab(int kc) {
    static uint8_t X[4][512];
    for (int i = 0; i < NE; i++) X[kc - 1][i * STR] = Q == 6 ? T6F[kc - 1][i] : T5[kc - 1][i];
    return X[kc - 1];
  }
  template <int MM, int A4> DEV u32 n(u32 st, const Ctx& c) { return c.t[cptc_addr_pk<Q>(st)]; }
  static void host(int mm, int, uint8_t* n) {
    for (u32 s = 0; s < 65536; s++) n[s] = Q == 6 ? T6F[KCEIL(mm) - 1][cptc_idx_h(s, 6)] : T5[KCEIL(mm) - 1][cptc_idx_h(s, 5)];
  }
};
typedef CptcPk<6, true> Cptc6PkLds; typedef CptcPk<6, false> Cptc6PkGlb; typedef CptcPk<5, true> Cptc5PkLds; typedef CptcPk<5, false> Cptc5PkGlb;
typedef Cptc5<true> Cptc5Lds; typedef Cptc5<false> Cptc5Glb; typedef Lc1<true> Lc1Lds; typedef Lc1<false> Lc1Glb;
typedef CptcSt<6, true> St6Lds; typedef CptcSt<6, false> St6Glb; typedef CptcSt<8, true> St8Lds; typedef CptcSt<8, false> St8Glb;
typedef CptcSt<10, true> St10Lds; typedef CptcSt<10, false> St10Glb; typedef CptcSt<12, true> St12Lds; typedef CptcSt<12, false> St12Glb;

#ifndef CAND
#define CAND Mad
#endif
typedef CAND CBT;
template <class T, class = void> struct HasTab { static const void* get(int) { return nullptr; } };
template <class T> struct HasTab<T, decltype((void)T::tab(1))> { static const void* get(int kc) { return T::tab(kc); } };

// eight states -> packed weights: w8a8 P[0], P[1] (4 bytes each); w4a4 P[0] (8 nibbles)
template <class CB, int MM, int A4> DEV void run8(const u32 (&st)[8], const typename CB::Ctx& ctx, u32 (&P)[2]) {
  if constexpr (CB::CUSTOM == 2) CB::template pk8<MM, A4>(st, ctx, P);
  else {
    u32 v[8];
    if constexpr (CB::CUSTOM == 1) CB::template dec8<MM>(st, ctx, v);
    else {
#pragma unroll
      for (int c = 0; c < 8; c++) v[c] = CB::template n<MM, A4>(st[c], ctx);
    }
    if constexpr (A4) {
      P[0] = v[0] + (v[1] << 4) + (v[2] << 8) + (v[3] << 12) + (v[4] << 16) + (v[5] << 20) + (v[6] << 24) + (v[7] << 28);
      P[1] = 0;
      if constexpr (CB::SGN) P[0] += 0x88888888u;
    } else {
      P[0] = v[0] + (v[1] << 8) + (v[2] << 16) + (v[3] << 24);
      P[1] = v[4] + (v[5] << 8) + (v[6] << 16) + (v[7] << 24);
      if constexpr (CB::SGN) { P[0] += 0x80808080u; P[1] += 0x80808080u; }
    }
  }
}

// ---------------------------------------------------------------- GEMV (M tokens; M = 0 dumps the decoded weights)
#define WPB 16u   // waves a GEMV block (so KW <= 16)
template <class CB, int MM, int M, int A4>
__global__ __launch_bounds__(32 * WPB) void gemv(const u32* __restrict__ W, const int* __restrict__ X, const int* __restrict__ bias2,
                                            int* __restrict__ Y, const u32* __restrict__ tab, u32 NT, u32 KT, u32 kwl, u32 J) {
  typedef Pat<MM> P_;
  constexpr int MA = M ? M : 1, NA = P_::NA, NL = NA + 1, TW = P_::TW;
  const u32 lane = threadIdx.x & 31u, wv = threadIdx.x >> 5, row = lane & 15u, tsel = lane >> 4;
  __shared__ int red[WPB][16][MA];
  __shared__ u32 ltab[CB::TABB ? CB::TABB / 4 : 1];
  if constexpr (CB::TABB != 0) {
    for (u32 i = threadIdx.x; i < (u32)(CB::TABB / 4); i += 32u * WPB) ltab[i] = tab[i];
    __syncthreads();
  }
  const typename CB::Ctx ctx(tab, ltab);
  const u32 b = (row * MM + 16u * MM - 16u) % (16u * MM), d0 = b >> 5, sh = 32u - (b & 31u);
  u32 woff[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) woff[i] = (d0 + i) % TW;
  const u32 KW = 1u << kwl, G8 = gridDim.x * WPB;
  for (u32 it = 0; it < J; ++it) {
    const u32 u = it * G8 + blockIdx.x * WPB + wv, tr = u >> kwl, kw = u & (KW - 1u);
    int acc[MA] = {};
    if (tr < NT) {
      const u32* trow = W + (size_t)tr * KT * TW;
      for (u32 g = kw; g < KT / 2u; g += KW) {
        const u32 kt = 2u * g + tsel;
        const u32* tw = trow + (size_t)kt * TW;
        u32 w[NL], n[5] = {};
#pragma unroll
        for (int i = 0; i < NL; i++) w[i] = tw[woff[i]];
#pragma unroll
        for (int i = 0; i < NA; i++) n[i] = (u32)((((u64)w[i] << 32) | w[i + 1]) >> sh);
        u32 PP[4], hh[16];
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int o = q * MM / 2, ix = o / 32, s = o % 32;
          const u32 hi = s ? (n[ix] << s) | (n[ix + 1] >> ((32 - s) & 31)) : n[ix];
          const u32 lo = s ? (n[ix + 1] << s) | (n[ix + 2] >> ((32 - s) & 31)) : n[ix + 1];
          u32 P[2];
          if constexpr (CB::RAW) { P[0] = hi; P[1] = lo; }
          else {
            u32 st[8];
            states<MM, CB::ST8>(hi, lo, 8u * q, J >> 31, st);
            if constexpr (CB::FOLD) {
#pragma unroll
              for (int c = 0; c < 8; c++) hh[8 * q + c] = CB::hash(st[c]);
              if constexpr (M == 0) run8<CB, MM, A4>(st, ctx, P);
            } else run8<CB, MM, A4>(st, ctx, P);
          }
          if constexpr (A4) PP[q] = P[0];
          else { PP[2 * q] = P[0]; PP[2 * q + 1] = P[1]; }
        }
        constexpr int NP = A4 ? 2 : 4;
        if constexpr (M == 0) {
#pragma unroll
          for (int j = 0; j < NP; j++) Y[(size_t)(tr * 16u + row) * (KT * NP) + kt * NP + j] = (int)PP[j];
        } else if constexpr (CB::FOLD) {
          const int* xp = X + (size_t)kt * 16u * M;
#pragma unroll
          for (int c = 0; c < 16; c++)
#pragma unroll
            for (int t = 0; t < M; t++) {
              if constexpr (A4) acc[t] = __builtin_amdgcn_sudot8(false, (int)hh[c], true, xp[c * M + t], acc[t], false);
              else acc[t] = __builtin_amdgcn_sudot4(false, (int)hh[c], true, xp[c * M + t], acc[t], false);
            }
        } else {
          const int* xp = X + (size_t)kt * NP * M;
#pragma unroll
          for (int j = 0; j < NP; j++)
#pragma unroll
            for (int t = 0; t < M; t++) {
              if constexpr (A4) acc[t] = __builtin_amdgcn_sudot8(false, (int)PP[j], true, xp[j * M + t], acc[t], false);
              else acc[t] = __builtin_amdgcn_sudot4(false, (int)PP[j], true, xp[j * M + t], acc[t], false);
            }
        }
      }
    }
    if constexpr (M != 0) {
#pragma unroll
      for (int t = 0; t < M; t++) {
        acc[t] += __shfl_xor(acc[t], 16);
        if (lane < 16u) red[wv][lane][t] = acc[t];
      }
      __syncthreads();
      if (kw == 0u && tr < NT && lane < 16u) {
#pragma unroll
        for (int t = 0; t < M; t++) {
          int s = 0;
          for (u32 k = 0; k < KW; k++) s += red[wv + k][lane][t];
          Y[(size_t)(tr * 16u + lane) * M + t] = 2 * s - bias2[t];
        }
      }
      if (J > 1u) __syncthreads();
    }
  }
}

// ---------------------------------------------------------------- GEMM (256 tokens, WMMA)
template <class CB, int MM, int A4, int TS>
__global__ __launch_bounds__(256) void gemm(const u32* __restrict__ W, const uint8_t* __restrict__ X, const int* __restrict__ bias2,
                                            int* __restrict__ Y, const u32* __restrict__ tab, u32 NT, u32 KT) {
  typedef Pat<MM> P_;
  constexpr int NA = P_::NA8, NL = NA + 1, TW = P_::TW, NTT = 16 / TS;
  const u32 lane = threadIdx.x & 31u, wv = threadIdx.x >> 5, row = lane & 15u, h = lane >> 4;
  __shared__ u32 ltab[CB::TABB ? CB::TABB / 4 : 1];
  if constexpr (CB::TABB != 0) {
    for (u32 i = threadIdx.x; i < (u32)(CB::TABB / 4); i += 256u) ltab[i] = tab[i];
    __syncthreads();
  }
  const typename CB::Ctx ctx(tab, ltab);
  const u32 b = (row * MM + h * (MM / 2) + 16u * MM - 16u) % (16u * MM), d0 = b >> 5, sh = 32u - (b & 31u);
  u32 woff[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) woff[i] = (d0 + i) % TW;
  const u32 u = blockIdx.x * 8u + wv, tr = u / TS, ts = u % TS;
  if (tr >= NT) return;
  v8i C[NTT];
#pragma unroll
  for (int j = 0; j < NTT; j++) C[j] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
  const u32* trow = W + (size_t)tr * KT * TW;
  for (u32 kt = 0; kt < KT; kt++) {
    const u32* tw = trow + (size_t)kt * TW;
    u32 w[NL], n[2] = {};
#pragma unroll
    for (int i = 0; i < NL; i++) w[i] = tw[woff[i]];
#pragma unroll
    for (int i = 0; i < NA; i++) n[i] = (u32)((((u64)w[i] << 32) | w[i + 1]) >> sh);
    u32 P[2];
    if constexpr (CB::RAW) { P[0] = n[0]; P[1] = n[1] ^ n[0]; }
    else {
      u32 st[8];
      states<MM, CB::ST8>(n[0], n[1], 8u * h, KT >> 31, st);
      run8<CB, MM, A4>(st, ctx, P);
    }
#pragma unroll
    for (int j = 0; j < NTT; j++) {
      const u32 tok = (ts * NTT + j) * 16u + row;
      if constexpr (A4) {
        const int B = *(const int*)(X + ((size_t)tok * KT + kt) * 8u + 4u * h);
        C[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(false, (int)P[0], true, B, C[j], false);
      } else {
        const v2i B = *(const v2i*)(X + ((size_t)tok * KT + kt) * 16u + 8u * h);
        C[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, (v2i){(int)P[0], (int)P[1]}, true, B, C[j], false);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NTT; j++) {
    const u32 tok = (ts * NTT + j) * 16u + row;
    const int bz = bias2[tok];
#pragma unroll
    for (int e = 0; e < 8; e++) Y[(size_t)(tr * 16u + 8u * h + e) * 256u + tok] = 2 * C[j][e] - bz;
  }
}

// context kernels (Base binary): an empty launch of the GEMV's grid, and a pure read of the weights
__global__ void k_empty(int* Y) { if (blockIdx.x == 0xFFFFFFFFu) Y[0] = 1; }
__global__ void k_read(const u32* __restrict__ W, u32* __restrict__ O, u32 nq) {   // nq = 16-byte vectors
  typedef u32 v4u __attribute__((ext_vector_type(4)));
  const u32 id = blockIdx.x * blockDim.x + threadIdx.x, tot = gridDim.x * blockDim.x;
  v4u a = {0, 0, 0, 0};
  for (u32 i = id; i < nq; i += tot) a ^= *(const v4u*)(W + 4 * (size_t)i);
  O[id] = a.x ^ a.y ^ a.z ^ a.w;
}

// ---------------------------------------------------------------- host
static const int MMS[] = {8, 16, 24, 32, 40, 48, 64};
#define FOR_MM(X) X(8) X(16) X(24) X(32) X(40) X(48) X(64)
struct GArgs { const u32* W; const int* X; const int* bias2; int* Y; const u32* tab; u32 NT, KT, kwl, J, grid; };

template <int MM, int M, int A4> static void launch_gemv_1(const GArgs& a) {
  if constexpr (!A4 || (CBT::i4(MM) && !(M == 0 && false))) {
    if (A4 == 0 && CBT::FOLD && std::string(CBT::NAME) == "mad4") return;
    hipLaunchKernelGGL((gemv<CBT, MM, M, A4>), dim3(a.grid), dim3(32 * WPB), 0, 0, a.W, a.X, a.bias2, a.Y, a.tab, a.NT, a.KT, a.kwl, a.J);
  }
}
static void launch_gemv(int MM, int M, int A4, const GArgs& a) {
#define LM(mm, m) if (MM == mm && M == m) { if (A4) launch_gemv_1<mm, m, 1>(a); else launch_gemv_1<mm, m, 0>(a); return; }
#define LMM(mm) LM(mm, 0) LM(mm, 1) LM(mm, 2) LM(mm, 4) LM(mm, 8)
  FOR_MM(LMM)
  printf("no gemv kernel MM %d M %d\n", MM, M); exit(1);
}
template <int MM, int A4, int TS> static void launch_gemm_1(const GArgs& a) {
  if constexpr (!A4 || CBT::i4(MM))
    hipLaunchKernelGGL((gemm<CBT, MM, A4, TS>), dim3(a.grid), dim3(256), 0, 0, a.W, (const uint8_t*)a.X, a.bias2, a.Y, a.tab, a.NT, a.KT);
}
static void launch_gemm(int MM, int A4, int TS, const GArgs& a) {
#define LG(mm) if (MM == mm) { if (A4) { if (TS == 1) launch_gemm_1<mm, 1, 1>(a); else launch_gemm_1<mm, 1, 2>(a); } \
                               else { if (TS == 1) launch_gemm_1<mm, 0, 1>(a); else launch_gemm_1<mm, 0, 2>(a); } return; }
  FOR_MM(LG)
  printf("no gemm kernel MM %d\n", MM); exit(1);
}
static bool has_i4(int MM) {
#define HI(mm) if (MM == mm) return CBT::i4(mm);
  FOR_MM(HI) return false;
}
static int bias2_of(int MM) {
#define HB(mm) if (MM == mm) return CBT::bias2(mm);
  FOR_MM(HB) return 0;
}

// the oracle's states: [N][Kin] u16 (row-major), from the tiles' rings, bit by bit
static void host_states(const std::vector<u32>& W, int MM, u32 NT, u32 KT, std::vector<uint16_t>& st) {
  const u32 R = 16 * MM, TW = MM / 2, Kin = KT * 16;
  st.resize((size_t)NT * 16 * Kin);
  std::vector<uint8_t> bit(R);
  for (u32 tr = 0; tr < NT; tr++)
    for (u32 kt = 0; kt < KT; kt++) {
      const u32* tw = &W[((size_t)tr * KT + kt) * TW];
      for (u32 i = 0; i < R; i++) bit[i] = (tw[i / 32] >> (31 - i % 32)) & 1;
      u32 end = 0;
      for (u32 i = 0; i < 256; i++) {
        const u32 p = i % 16, D = ((p + 1) * MM) / 16 - (p * MM) / 16;
        end += D;
        u32 s = 0;
        for (u32 j = 0; j < 16; j++) s = (s << 1) | bit[(end + R - 16 + j) % R];
        if (D == 0) s ^= (0x9E37u * (p + 1)) & 0xFFFF;
        st[(size_t)(tr * 16 + i / 16) * Kin + kt * 16 + p] = (uint16_t)s;
      }
    }
}

struct Timer {
  std::vector<double> us;
  template <class F> void run(F&& launch, int reps, int batch, int& ctr) {
    us.clear();
    for (int r = 0; r < reps; r++) {
      CK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < batch; i++) launch(ctr++);
      CK(hipDeviceSynchronize());
      us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / batch);
    }
    std::sort(us.begin(), us.end());
  }
  double med() const { return us[us.size() / 2]; }
  double q(double f) const { return us[std::min(us.size() - 1, (size_t)(f * us.size()))]; }
};

static std::vector<int> ilist(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) { v.push_back((int)lround(atof(p) * 1)); while (*p && *p != ',') p++; if (*p) p++; }
  return v;
}

int main(int argc, char** argv) {
  std::vector<int> mms = {16, 24, 32, 40, 48, 64}, ms = {1, 2, 4, 8, 256};
  std::string shapes = "gate_up,down", outp;
  int reps = 31, batch = 128, check = 1, ctx_runs = 0;
  double rot_mb = 256, warm_s = 0.4, warm_cfg = 0.15;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) { printf("missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
    if (a == "--mm") mms = ilist(nx());
    else if (a == "--M") ms = ilist(nx());
    else if (a == "--shapes") shapes = nx();
    else if (a == "--out") outp = nx();
    else if (a == "--reps") reps = atoi(nx());
    else if (a == "--batch") batch = atoi(nx());
    else if (a == "--check") check = atoi(nx());
    else if (a == "--ctx") ctx_runs = atoi(nx());
    else if (a == "--rot-mb") rot_mb = atof(nx());
    else if (a == "--warm") warm_s = atof(nx());
    else if (a == "--warm-cfg") warm_cfg = atof(nx());
    else { printf("usage: %s [--mm 16,24,..] [--M 1,2,4,8,256] [--shapes gate_up,down] [--out f.jsonl] [--reps 25] [--batch 64] "
                  "[--check 1] [--ctx 1] [--rot-mb 256]\n", argv[0]); return 1; }
  }
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
  if (!strstr(pr.gcnArchName, "gfx1201")) { printf("device 0 is %s (%s), not gfx1201: refusing\n", pr.name, pr.gcnArchName); return 1; }
  FILE* fo = outp.empty() ? nullptr : fopen(outp.c_str(), "a");
  char place[32]; snprintf(place, sizeof place, "%s", CBT::PLACE);
  printf("# cand %s place %s on %s (%s)\n", CBT::NAME, place, pr.name, pr.gcnArchName);
  int nbad = 0;
  {   // warm-up: ~0.4 s of launches so the card leaves its idle clocks before anything is timed
    u32* ww; u32* wo; CK(hipMalloc(&ww, 8 << 20)); CK(hipMalloc(&wo, 256 * 256 * 4)); CK(hipMemset(ww, 1, 8 << 20));
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
      for (int i = 0; i < 64; i++) hipLaunchKernelGGL(k_read, dim3(256), dim3(256), 0, 0, ww, wo, (u32)((8 << 20) / 16));
      CK(hipDeviceSynchronize());
    }
    CK(hipFree(ww)); CK(hipFree(wo));
  }
  auto emit = [&](const std::string& js) { printf("%s\n", js.c_str()); fflush(stdout); if (fo) { fprintf(fo, "%s\n", js.c_str()); fflush(fo); } };
  struct Shape { const char* name; u32 N, Kin; };
  const Shape all[] = {{"gate_up", 4608, 5120}, {"down", 5120, 2304}, {"tiny", 48, 96}};
  std::mt19937 rng(208);
  for (const Shape& sp : all) {
    if (shapes.find(sp.name) == std::string::npos) continue;
    const u32 NT = sp.N / 16, KT = sp.Kin / 16, N = sp.N, Kin = sp.Kin;
    for (int MM : mms) {
      const u32 TW = MM / 2;
      const size_t nw = (size_t)NT * KT * TW;
      std::vector<u32> W(nw); for (auto& v : W) v = rng();
      const int NC = std::min(64, std::max(1, (int)ceil(rot_mb * 1e6 / (nw * 4.0))));
      std::vector<u32*> cp(NC);
      for (int i = 0; i < NC; i++) {
        CK(hipMalloc(&cp[i], nw * 4 + 64));
        if (i == 0) CK(hipMemcpy(cp[0], W.data(), nw * 4, hipMemcpyHostToDevice));
        else CK(hipMemcpy(cp[i], cp[0], nw * 4, hipMemcpyDeviceToDevice));
      }
      std::vector<uint16_t> st;
      if (check && !CBT::RAW) host_states(W, MM, NT, KT, st);
      // the table (a table candidate's; 16 KB of zeros otherwise so the pointer is always valid)
      std::vector<uint8_t> tabh(16384, 0);
      if (const void* t = HasTab<CBT>::get(KCEIL(MM))) memcpy(tabh.data(), t, CBT::TABG);
      u32* dtab; CK(hipMalloc(&dtab, tabh.size())); CK(hipMemcpy(dtab, tabh.data(), tabh.size(), hipMemcpyHostToDevice));
      if (ctx_runs && CBT::RAW) {   // context: empty launch, pure read
        int* dy; CK(hipMalloc(&dy, 64)); u32* dob; CK(hipMalloc(&dob, 4096 * 256 * 4));
        Timer T; int ctr = 0;
        for (int grid : {36, 144, 288}) {
          T.run([&](int) { hipLaunchKernelGGL(k_empty, dim3(grid), dim3(256), 0, 0, dy); }, reps, batch, ctr);
          char js[512]; snprintf(js, sizeof js, "{\"cand\":\"empty\",\"shape\":\"%s\",\"K\":%g,\"grid\":%d,\"us\":%.3f,\"us_min\":%.3f}", sp.name, MM / 16.0, grid, T.med(), T.us[0]);
          emit(js);
        }
        double best = 1e30; int bg = 0;
        for (int grid : {64, 128, 256, 512, 1024, 2048}) {
          T.run([&](int c) { hipLaunchKernelGGL(k_read, dim3(grid), dim3(256), 0, 0, cp[c % NC], dob, (u32)(nw / 4)); }, reps, batch, ctr);
          if (T.med() < best) { best = T.med(); bg = grid; }
        }
        char js[512]; snprintf(js, sizeof js, "{\"cand\":\"read\",\"shape\":\"%s\",\"K\":%g,\"grid\":%d,\"us\":%.3f,\"wbytes\":%zu,\"GBps\":%.1f}", sp.name, MM / 16.0, bg, best, nw * 4, nw * 4 / best / 1e3);
        emit(js);
        CK(hipFree(dy)); CK(hipFree(dob));
      }
      bool cold = true;   // host work since the last launch: the first configuration warms up longer
      for (int A4 = 0; A4 < 2; A4++) {
        if (A4 && !has_i4(MM)) continue;
        if (!A4 && CBT::FOLD && std::string(CBT::NAME) == "mad4") continue;
        const int NP = A4 ? 2 : 4, bias2 = bias2_of(MM);
        std::vector<uint8_t> ntab(65536), nref;
        if (check && !CBT::RAW) {
          CBT::host(MM, A4, ntab.data());
          nref.resize(st.size());
          for (size_t i = 0; i < st.size(); i++) nref[i] = ntab[st[i]];
          // dump: every decoded weight
          const size_t nd = (size_t)N * KT * NP;
          int* dd; CK(hipMalloc(&dd, nd * 4));
          GArgs a{cp[0], nullptr, nullptr, dd, dtab, NT, KT, 0, 1, (NT + WPB - 1) / WPB};
          launch_gemv(MM, 0, A4, a); CK(hipDeviceSynchronize());
          std::vector<u32> got(nd); CK(hipMemcpy(got.data(), dd, nd * 4, hipMemcpyDeviceToHost)); CK(hipFree(dd));
          size_t bad = 0;
          for (u32 r = 0; r < N; r++)
            for (u32 c = 0; c < Kin; c++) {
              const u32 g = A4 ? (got[(size_t)r * KT * 2 + c / 8] >> (4 * (c % 8))) & 15 : (got[(size_t)r * KT * 4 + c / 4] >> (8 * (c % 4))) & 255;
              bad += g != nref[(size_t)r * Kin + c];
            }
          char js[512]; snprintf(js, sizeof js, "{\"cand\":\"%s\",\"place\":\"%s\",\"test\":\"decode\",\"shape\":\"%s\",\"K\":%g,\"mode\":\"%s\",\"weights\":%zu,\"bad\":%zu}",
                                 CBT::NAME, place, sp.name, MM / 16.0, A4 ? "w4a4" : "w8a8", (size_t)N * Kin, bad);
          emit(js); nbad += bad != 0; cold = true;
        }
        for (int M : ms) {
          // activations x[t][c]: int8 in [-127, 127] or int4 in [-8, 7]
          std::vector<int8_t> x((size_t)M * Kin);
          for (auto& v : x) v = A4 ? (int8_t)((int)(rng() % 16) - 8) : (int8_t)((int)(rng() % 255) - 127);
          std::vector<int> b2(std::max(M, 1), 0);
          for (int t = 0; t < M; t++) { long s = 0; for (u32 c = 0; c < Kin; c++) s += x[(size_t)t * Kin + c]; b2[t] = (int)(bias2 * s); }
          int* db2; CK(hipMalloc(&db2, b2.size() * 4)); CK(hipMemcpy(db2, b2.data(), b2.size() * 4, hipMemcpyHostToDevice));
          const bool isdec = M == 0;   // the weight-decode pass alone (decoded int8 / int4 written to memory)
          int* dy; CK(hipMalloc(&dy, isdec ? (size_t)N * KT * NP * 4 : (size_t)N * M * 4));
          std::vector<int> y((size_t)N * M);
          int ctr = 0; Timer T;
          char cfg[64]; double best = 1e30; GArgs ba{};
          std::vector<int> xl; std::vector<uint8_t> xg;
          const bool isgemm = M == 256;
          if (isdec) xl.assign(16, 0);
          else if (!isgemm) {
            if (CBT::FOLD) {
              xl.resize((size_t)Kin * M);
              for (u32 c = 0; c < Kin; c++) for (int t = 0; t < M; t++)
                xl[(size_t)c * M + t] = A4 ? (int)(((u32)x[(size_t)t * Kin + c] & 15u) * MAD4_ONES) : (int)((u32)(uint8_t)x[(size_t)t * Kin + c] * 0x01010101u);
            } else {
              const int per = A4 ? 8 : 4, bits = A4 ? 4 : 8;
              xl.assign((size_t)Kin / per * M, 0);
              for (u32 c = 0; c < Kin; c++) for (int t = 0; t < M; t++)
                xl[(size_t)(c / per) * M + t] |= (int)(((u32)x[(size_t)t * Kin + c] & ((1u << bits) - 1)) << (bits * (c % per)));
            }
          } else {
            if (A4) { xg.assign((size_t)M * Kin / 2, 0); for (size_t i = 0; i < x.size(); i++) xg[i / 2] |= (uint8_t)((x[i] & 15) << (4 * (i % 2))); }
            else { xg.resize(x.size()); memcpy(xg.data(), x.data(), x.size()); }
          }
          const size_t xbytes = isgemm ? xg.size() : xl.size() * 4;
          int* dx; CK(hipMalloc(&dx, xbytes + 64));
          CK(hipMemcpy(dx, isgemm ? (const void*)xg.data() : (const void*)xl.data(), xbytes, hipMemcpyHostToDevice));
          const int rp = reps, bt = isgemm ? 4 : batch;
          {   // the host work above let the card idle (clocks): load it until the launch time is steady -- blocks of
              // ~60 ms, at least warm_cfg seconds, until two consecutive blocks agree within 1% (cap 5 s)
            GArgs a{cp[0], dx, db2, dy, dtab, NT, KT, 3, 1, isgemm ? (NT * 2 + 7) / 8 : ((NT << 3) + WPB - 1) / WPB};
            const auto t0 = std::chrono::steady_clock::now();
            double prev = 0, el = 0; int nb = isgemm ? 8 : 512;
            for (;;) {
              const auto b0 = std::chrono::steady_clock::now();
              for (int i = 0; i < nb; i++) { a.W = cp[ctr++ % NC]; if (isgemm) launch_gemm(MM, A4, 2, a); else launch_gemv(MM, M, A4, a); }
              CK(hipDeviceSynchronize());
              const auto b1 = std::chrono::steady_clock::now();
              const double bt_ = std::chrono::duration<double>(b1 - b0).count(), per = bt_ / nb;
              el = std::chrono::duration<double>(b1 - t0).count();
              if (bt_ < 0.03) { nb *= 2; prev = 0; continue; }
              if ((el >= (cold && warm_cfg >= 0.1 ? std::max(warm_cfg, 0.8) : warm_cfg) && prev > 0 && fabs(per / prev - 1) < 0.01) || el > 5) break;   // --warm-cfg < 0.1: an exactness-only run
              prev = per;
            }
            cold = check != 0;   // the oracle's product after each timed configuration is an idle gap too
          }
          if (!isgemm) {
            for (int pass = 0; pass < 2; pass++)   // pass 0 also warms the card up (clocks), pass 1 decides
              for (u32 kwl = 1; kwl < 5; kwl++)
                for (u32 J : {1u, 2u}) {
                  GArgs a{cp[0], dx, db2, dy, dtab, NT, KT, kwl, J, ((NT << kwl) + WPB * J - 1) / (WPB * J)};
                  T.run([&](int c) { a.W = cp[c % NC]; launch_gemv(MM, M, A4, a); }, 5, bt / 2, ctr);
                  if (pass && T.med() < best) { best = T.med(); ba = a; }
                }
            snprintf(cfg, sizeof cfg, "\"kw\":%u,\"J\":%u", 1u << ba.kwl, ba.J);
            for (int i = 0; i < 20; i++) { ba.W = cp[ctr++ % NC]; launch_gemv(MM, M, A4, ba); }
            T.run([&](int c) { ba.W = cp[c % NC]; launch_gemv(MM, M, A4, ba); }, rp, bt, ctr);
          } else {
            int bts = 1;
            for (int pass = 0; pass < 2; pass++)
              for (int TS : {1, 2}) {
                GArgs a{cp[0], dx, db2, dy, dtab, NT, KT, 0, 1, (NT * TS + 7) / 8};
                T.run([&](int c) { a.W = cp[c % NC]; launch_gemm(MM, A4, TS, a); }, 5, bt, ctr);
                if (pass && T.med() < best) { best = T.med(); ba = a; bts = TS; }
              }
            snprintf(cfg, sizeof cfg, "\"ts\":%d", bts);
            for (int i = 0; i < 4; i++) { ba.W = cp[ctr++ % NC]; launch_gemm(MM, A4, bts, ba); }
            T.run([&, bts](int c) { ba.W = cp[c % NC]; launch_gemm(MM, A4, bts, ba); }, rp, bt, ctr);
            ba.J = bts;
          }
          // exactness of the timed configuration
          long bad = -1;
          if (check && !CBT::RAW && !isdec) {
            CK(hipMemset(dy, 0x55, (size_t)N * M * 4));
            ba.W = cp[NC - 1];
            if (isgemm) launch_gemm(MM, A4, ba.J, ba); else launch_gemv(MM, M, A4, ba);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(y.data(), dy, (size_t)N * M * 4, hipMemcpyDeviceToHost));
            bad = 0;
            const u32 rstep = isgemm ? 53 : 1;   // the GEMM: every 53rd row (all tile rows mod 16 over the run)
            for (u32 r = 0; r < N; r += rstep)
              for (int t = 0; t < M; t++) {
                long s = 0; const uint8_t* nr = &nref[(size_t)r * Kin]; const int8_t* xr = &x[(size_t)t * Kin];
                for (u32 c = 0; c < Kin; c++) s += (long)nr[c] * xr[c];
                bad += (long)y[(size_t)r * M + t] != 2 * s - b2[t];
              }
            nbad += bad != 0;
          }
          char js[768];
          snprintf(js, sizeof js, "{\"cand\":\"%s\",\"place\":\"%s\",\"shape\":\"%s\",\"N\":%u,\"Kin\":%u,\"K\":%g,\"M\":%d,\"mode\":\"%s\",%s,"
                   "\"us\":%.3f,\"us_min\":%.3f,\"us_p90\":%.3f,\"reps\":%d,\"batch\":%d,\"wbytes\":%zu,\"copies\":%d,\"bad\":%ld}",
                   CBT::NAME, place, sp.name, N, Kin, MM / 16.0, M, A4 ? "w4a4" : "w8a8", cfg, T.med(), T.us[0], T.q(0.9), rp, bt, nw * 4, NC, bad);
          emit(js);
          CK(hipFree(dx)); CK(hipFree(dy)); CK(hipFree(db2));
        }
      }
      CK(hipFree(dtab));
      for (auto p : cp) CK(hipFree(p));
    }
  }
  if (fo) fclose(fo);
  printf("# %s: %s\n", CBT::NAME, nbad ? "MISMATCH" : "all exact");
  return nbad ? 2 : 0;
}
