/* Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and
 * radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE. */
// exl3r4d_nt.hip -- the dense EXL3 linear (exl3_dense / gemm_exl3_nt) on gfx1201's f16 matrix cores.
//
// Two forms, both byte-identical in output:
//   * nt_pass: the trellis decoded in registers (Plan / decode_frag), f16 WMMA, the 128-point output rotation x svh.
//     One 16-row tile a workgroup at M <= 16 (decode), EXL3R4D_NT_MT tiles past it.
//   * T2 (from EXL3R4D_T2_MIN rows): the weight decoded once into f16 WMMA fragments (wdec), then a 128x256 GEMM.
// The input rotation (a x suh, 128-point Hadamard) is the prep kernel's, into the op's scratch.
//
// LINEAGE: the decode, rotation and epilogue are Dyluhn's (radiance-exl3) exl3-moe kernel (exl3_moe_wmma.hip) by way of the radiance
// exl3r4d MoE library this file was cut from (that library's history: git log before this file existed).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdlib>

#include "exl3r4d_common.hip.h"

#ifndef EXL3R4D_NT          // nontemporal weight loads (decode band)
#define EXL3R4D_NT 1
#endif
#ifndef EXL3R4D_BUF         // buffer loads for the weight / activation fragments
#define EXL3R4D_BUF 1
#endif
#ifndef EXL3R4D_EPIC        // the epilogue's paired tile sums (adopted; T2 uses the same order)
#define EXL3R4D_EPIC 1
#endif

// ============================================================================= gemm_exl3_nt (interface v8): the dense linear
//
// One weight [N, K], no routing: moe_exl3_down's arithmetic over token-ordered rows in P1c's GSCR form (each
// workgroup rotates its rows into the rotated-rows region, the K loop reads buffer loads from it), K 1..8, mul1,
// any M (one 16-row tile a workgroup at M <= 16, EXL3R4D_NT_MT tiles past it -- below), split-K 1 (the down's
// fit logic, launch.c L7, is the tuning arm).
//
// THE SEAM: when the weight has plane 3 (suh.1 -- the operand's RadTensor.data is null when it does not),
// the output rows [0, N/2) are scaled by suh and [N/2, N) by suh.1 (the stacked gate_up's rule; the
// reference's launch_nt makes the same two-half split, its 128-point rotation demanding N/2 % 128 -- the
// row enforces N % 256 in that case). Every 128-column workgroup lies wholly on one side, so it rotates
// with one scale vector and reads one half of the region. The region is [T][2K] f16 (one shared copy, not
// P1c's per-column-block copies: same-side workgroups write the SAME bytes, and a workgroup reads only the
// rows it rotated itself, its own barrier ordering them). Without plane 3, planes_of sets suh1 = suh and
// the selection is inactive (both halves rotate identically; the read takes the workgroup's own half).
//
// THE ARITHMETIC, word for word the decode band's: a[m] x suh (or suh.1) rotated by rotate128, the W_hat
// trellis decoded in registers (Plan), f16 WMMAs, the split-K LDS reduction (sk > 1), the 128-point output
// rotation x 1/sqrt(128) x svh, bf16 / f16 / f32 out (the epilogue_block MATS 1 arm, EPIC as adopted for down;
// EPIB refuted for down). One dispatch, no prep kernel, no counters, no scratch past the region.
#if EXL3R4D_NCLS == 2 && !EXL3R4D_GD4
// The levers (first pass): the adopted down's configuration (EXL3_PF 2, buffer loads, KX_BUF; EPIC on,
// EPIB off, no SGB).
#ifndef EXL3R4D_NT_PF
#define EXL3R4D_NT_PF 2
#endif
#ifndef EXL3R4D_NT_KX_BUF
#define EXL3R4D_NT_KX_BUF 1
#endif
#ifndef EXL3R4D_NT_EPIC
#define EXL3R4D_NT_EPIC 1
#endif

#ifndef EXL3R4D_NT_WPE
#define EXL3R4D_NT_WPE 0
#endif
#if EXL3R4D_NT_WPE > 0
#define EXL3R4D_NT_WPE_ATTR __attribute__((amdgpu_waves_per_eu(EXL3R4D_NT_WPE)))
#else
#define EXL3R4D_NT_WPE_ATTR
#endif

// MT (the prefill form): a workgroup owns MT consecutive 16-row tiles, so one decoded weight fragment feeds MT
// WMMAs (exl3r4d_prefill.hip's MTX idea without its run map: one weight, token-ordered rows). MT 1 is the decode
// band's arithmetic exactly; every tile's sums run in the same order whatever MT is, so the bits do not depend on it
// (measured: the 2k-token KLD record identical at MT 1, 2, 4, 8). Measured on gfx1201, Qwen3.8-27B 4.00 bpw, 2143-token
// prefill, exl3_dense total: MT 2 2805 ms, MT 4 2454 ms, MT 8 4038 ms (spills) -- 4 adopted.
#ifndef EXL3R4D_NT_MT
#define EXL3R4D_NT_MT 4
#endif
// ALDS (prefill form): the activation slab [16 MT rows][64 features] staged once a workgroup in LDS (coalesced 16-B
// loads, double-buffered, one barrier a slab) instead of each wave loading the same fragments from global -- the 4
// waves of a workgroup read identical A, and a 16-row fragment load uses a quarter of each line. Same values, same
// sums in the same order.
// The row's own launch bound: it launches at most 8 waves (EXL3R4D_LB's 1024 caps every wave's registers).
#ifndef EXL3R4D_NT_LB
#define EXL3R4D_NT_LB 256
#endif
// RG (prefill form, ALDS): row groups a workgroup -- RG sets of 8 / NPW waves over the same 128 columns, each its own
// 16 MT rows: one staged slab and one column block's weight stream serve RG x the rows, each wave's registers as at RG 1.
#ifndef EXL3R4D_NT_RG
#define EXL3R4D_NT_RG 1
#endif
#ifndef EXL3R4D_NT_ALDS
#define EXL3R4D_NT_ALDS 1
#endif
#define EXL3R4D_NT_KC 4                       // k steps a slab (64 features)
#define EXL3R4D_NT_AP (16 * EXL3R4D_NT_KC + 8)  // the slab's LDS row pitch, halfs (+16 B against bank conflicts)

template <int BITS, int NPW, int MT, bool PREP, int RG>
static __device__ __forceinline__ void nt_pass(const Exl3r4dGemm& p, const Planes& w, int cb, int rc, int r1, float* lds) {
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  constexpr int PF = EXL3R4D_NT_PF;
  const int SK = p.sk;
  const int tid = (int)threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  static_assert(RG == 1 || (PREP && EXL3R4D_NT_ALDS), "row groups are the staged-slab form's");
  constexpr int WVC = 8 / NPW;           // waves a row group (the 128 columns)
  const int rg = RG > 1 ? wave / WVC : 0, wc = RG > 1 ? wave % WVC : wave;
  const int col_local = wc / SK, ksplit = wc % SK;
  const int h = lane >> 4, nl = lane & 15;
  const int width = w.rows;              // [N, K]: one matrix, no gate/up split
  const int ntiles = width / 16, ksteps = w.cols / 16;
  // The seam (above): this workgroup's half of the region, and its scale vector.
  const int mat_a = (w.suh1 != w.suh && cb * 128 >= width / 2) ? 1 : 0;
  const P plan(lane);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];

  // The record: one [N, K] matrix (no mat offset -- the stacked form's `mat * ksteps * ntiles * WORDS`
  // is 0 here).
  gu8 wt[NPW];
  const unsigned* wexp = w.codes;
  const size_t kps = ksteps / SK;
  const size_t ks0 = (size_t)ksplit * kps;
  const int vl = col_local * NPW;
  const int nt0 = cb * 8 + vl % 8;
  const size_t tile0 = ks0 * ntiles + nt0;
#pragma unroll
  for (int j = 0; j < NPW; j++) wt[j] = (gu8)(const unsigned char*)(wexp + (tile0 + j) * WORDS);
  const size_t kbytes = (size_t)ntiles * WORDS * 4;

  // GSCR: rotate this workgroup's rows into the shared region [T][2K], this workgroup's half, a batch of 4 with
  // every load of the batch first (the down's P1c loop, the sorted indirection gone -- dense row m is token m).
  // PREP: exl3r4d_nt_prep_kernel rotated every row once before this dispatch (the same function, the same bits);
  // in-workgroup, every column block would rotate the same rows again (272 times over at gate_up).
  if constexpr (!PREP) {
    const int nrows = min(EXL3_BLOCK * MT, r1 - rc);
    const int blocks = w.cols / 128;
    const int tasks = nrows * blocks;
    const int nwaves = (int)EXL3R4D_BDIM / WAVE32;
    constexpr int NB = 4;
    typedef const __attribute__((address_space(1))) v2u* gv2;
    const gu8 xb = (gu8)(const unsigned char*)p.x;
    const gu8 sx = (gu8)(const unsigned char*)(mat_a ? w.suh1 : w.suh);
    _Float16* xo = (_Float16*)p.xr + (size_t)mat_a * w.cols;
    for (int t0 = wave; t0 < tasks; t0 += NB * nwaves) {
      int bi[NB], bk[NB];
      v2u bx[NB], bs[NB];
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const int t1 = t0 + b * nwaves, tsk = t1 < tasks ? t1 : t0;
        bi[b] = tsk / blocks;
        bk[b] = tsk - bi[b] * blocks;
      }
#pragma unroll
      for (int b = 0; b < NB; b++) {
        bx[b] = *(gv2)(xb + ((size_t)(rc + bi[b]) * p.x_pitch + bk[b] * 128 + 4 * lane) * 2);
        bs[b] = *(gv2)(sx + (size_t)(bk[b] * 128 + 4 * lane) * 2);
      }
#pragma unroll
      for (int b = 0; b < NB; b++)
        if (t0 + b * nwaves < tasks)
          *(v2u*)(xo + (size_t)(rc + bi[b]) * p.xr_pitch + bk[b] * 128 + 4 * lane) =
              rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane);
    }
    __syncthreads();   // the workgroup's release/acquire on global: its rows are visible to its reads
  }

  // Tile t's rows: rc + 16 t + nl; a row past the run reads the run's first row (valid memory) and is not stored.
  int row[MT];
  bool in_run[MT];
  unsigned avo[MT];
#pragma unroll
  for (int t = 0; t < MT; t++) {
    row[t] = rc + EXL3_BLOCK * (MT * rg + t) + nl;
    in_run[t] = EXL3_BLOCK * (MT * rg + t) + nl < r1 - rc;
    avo[t] = (unsigned)((size_t)(in_run[t] ? row[t] : rc) * p.xr_pitch + 4 * h) * 2;
  }
  gu8 abase = (gu8)(const unsigned char*)((const _Float16*)p.xr + (size_t)mat_a * w.cols + (size_t)ks0 * 16);

  v8f acc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[t][j] = (v8f)(0.f);
  [[maybe_unused]] v8h svp[epi_opw<NPW, 1>()][1];   /* the epilogue's svh-early form (SVH_EARLY 0: unused) */

  // The K loop: the down's adopted PF-stage buffer form (V1b; the three-word form is K = 4's, every other K
  // the four-word one, R23).
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t wrs =
      __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t ars =
      __builtin_amdgcn_make_buffer_rsrc((void*)abase, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const unsigned bso = (unsigned)(tile0 * WORDS * 4);
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr bool kBuf = EXL3R4D_BUF && NW == 2 && (BITS == 4 || EXL3R4D_NT_KX_BUF);
  constexpr int kAux = EXL3R4D_NT ? 1 : 0;

#define NT_LOAD(ks, W_, A_)                                                        \
  do {                                                                               \
    if constexpr (kBuf) {                                                            \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
        if constexpr (BITS == 4) {                                                   \
          unsigned m_, z_, q_;                                                       \
          m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);  \
          z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);  \
          q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);  \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;   \
        } else {                                                                     \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                               \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        }                                                                            \
      }                                                                              \
      _Pragma("unroll") for (int t = 0; t < MT; t++) {                                 \
        union { v2u w[2]; v8h f; } x_;                                               \
        x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[t], (unsigned)(ks) * 32u, 0);      \
        x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[t] + 16u, (unsigned)(ks) * 32u, 0); \
        A_[t] = x_.f;                                                                \
      }                                                                              \
    } else {                                                                         \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
        const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                                 \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                \
        _Pragma("unroll") for (int x = 0; x < NW; x++)                               \
          W_[j][r][x] = EXL3R4D_NT ? __builtin_nontemporal_load((gu32)(tb_ + wvo[r][x])) \
                                   : *(gu32)(tb_ + wvo[r][x]);                        \
      }                                                                              \
      _Pragma("unroll") for (int t = 0; t < MT; t++) {                                 \
        union { v2u w[2]; v8h f; } x_;                                               \
        const unsigned char* ap_ = (const unsigned char*)abase + avo[t] + (unsigned)(ks) * 32u; \
        x_.w[0] = *(const v2u*)ap_;                                                  \
        x_.w[1] = *(const v2u*)(ap_ + 16u);                                          \
        A_[t] = x_.f;                                                                \
      }                                                                              \
    }                                                                                \
  } while (0)

#define NT_COMPUTE(W_, A_)                                                         \
  do {                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
      const v8h wf = decode_frag<BITS, NW>(W_[j], plan.shift);                       \
      _Pragma("unroll") for (int t = 0; t < MT; t++) acc[t][j] = mma(wf, A_[t], acc[t][j]); \
    }                                                                                \
  } while (0)

  if constexpr (PREP && EXL3R4D_NT_ALDS) {
    constexpr int KC = EXL3R4D_NT_KC, AP = EXL3R4D_NT_AP, ROWS = 16 * MT * RG, SLAB = ROWS * AP;
    static_assert(KC % PF == 0, "the weight stages index by the slab's step");
    _Float16* const abuf = (_Float16*)lds;          // [2][16 MT][AP] (the epilogue reuses it after the loop)
    constexpr int nth = WVC * RG * WAVE32;
    constexpr int NLD = (ROWS * KC * 2 + nth - 1) / nth;   // 16-B loads a thread
    const _Float16* const xrow = (const _Float16*)p.xr + (size_t)mat_a * w.cols;
    v4u stg[NLD];
    auto slab_load = [&](int c) {
#pragma unroll
      for (int i = 0; i < NLD; i++) {
        const int idx = tid + nth * i, r = idx / (2 * KC), q = idx % (2 * KC);
        const int rr = rc + r < r1 ? rc + r : rc;   // a row past the run reads the run's first (not stored)
        stg[i] = idx < ROWS * 2 * KC ? *(const v4u*)(xrow + (size_t)rr * p.xr_pitch + c * KC * 16 + q * 8) : (v4u)(0u);
      }
    };
    auto slab_store = [&](int b) {
#pragma unroll
      for (int i = 0; i < NLD; i++) {
        const int idx = tid + nth * i, r = idx / (2 * KC), q = idx % (2 * KC);
        if (idx < ROWS * 2 * KC) *(v4u*)(abuf + b * SLAB + r * AP + q * 8) = stg[i];
      }
    };
#define NT_LOADW(ks, W_)                                                           \
    do {                                                                             \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
        if constexpr (kBuf && BITS == 4) {                                           \
          unsigned m_, z_, q_;                                                       \
          m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);  \
          z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);  \
          q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);  \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;   \
        } else if constexpr (kBuf) {                                                 \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                               \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        } else {                                                                     \
          const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                               \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                              \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                             \
            W_[j][r][x] = *(gu32)(tb_ + wvo[r][x]);                                  \
        }                                                                            \
      }                                                                              \
    } while (0)
    unsigned wb[PF][NPW][2][NW];
#pragma unroll
    for (int u = 0; u < PF - 1; u++) NT_LOADW(u < kps ? u : kps - 1, wb[u]);
    const int nchunks = (int)kps / KC;
    slab_load(0);
    slab_store(0);
    lds_barrier();   // LDS only: the rows came from an earlier dispatch
    for (int c = 0; c < nchunks; c++) {
      if (c + 1 < nchunks) slab_load(c + 1);
      const _Float16* const ab_ = abuf + (c & 1) * SLAB;
#pragma unroll
      for (int u = 0; u < KC; u++) {
        const int ks = c * KC + u;
        const int ls = ks + PF - 1 < (int)kps ? ks + PF - 1 : (int)kps - 1;
        NT_LOADW(ls, wb[(u + PF - 1) % PF]);
        v8h af[MT];
#pragma unroll
        for (int t = 0; t < MT; t++) af[t] = frag_lds(ab_ + (16 * (MT * rg + t) + nl) * AP + u * 16 + 4 * h);
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const v8h wf = decode_frag<BITS, NW>(wb[u % PF][j], plan.shift);
#pragma unroll
          for (int t = 0; t < MT; t++) acc[t][j] = mma(wf, af[t], acc[t][j]);
        }
      }
      if (c + 1 < nchunks) slab_store((c + 1) & 1);
      lds_barrier();   // LDS only: the rows came from an earlier dispatch
    }
#undef NT_LOADW
  } else {
  // The loads are unconditional (the step index clamps to the last one) -- a branch around the prefetch
  // serialised it with the math (exl3_moe_wmma.hip:714-716; FACTS-exl3 trap 11).
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF][MT];
#pragma unroll
  for (int u = 0; u < PF - 1; u++)
    NT_LOAD(u < kps ? u : kps - 1, wb[u], ab[u]);
  int ks = 0;
  for (; ks + PF <= kps; ks += PF) {
#pragma unroll
    for (int u = 0; u < PF; u++) {
      const int ls = ks + u + PF - 1 < kps ? ks + u + PF - 1 : kps - 1;
      NT_LOAD(ls, wb[(u + PF - 1) % PF], ab[(u + PF - 1) % PF]);
      NT_COMPUTE(wb[u], ab[u]);
    }
  }
#pragma unroll
  for (int u = 0; u < PF - 1; u++)
    if (ks + u < kps) NT_COMPUTE(wb[u], ab[u]);
  }
#undef NT_LOAD
#undef NT_COMPUTE

#pragma unroll
  for (int t = 0; t < MT; t++) {
    if (SK > 1) splitk_reduce<NPW>(acc[t], lds, col_local, ksplit, SK, lane);
    epilogue_block<NPW, 1, false, false, EXL3R4D_NT_EPIC>(p, w, acc[t], col_local, ksplit, lane, cb, row[t],
                                                        in_run[t], true, lds + rg * 4096, svp);   // 16 KB a group
  }
}

// One workgroup = (128-column block blockIdx.x, MT 16-row tiles from blockIdx.y * 16 MT): one pass, its own rows.
// Column blocks fastest: the resident workgroups share one row block's activations. REFUTED, the swap (row blocks
// fastest, to share a column block's weights): 2k prefill exl3_dense 2454 -> 3590 ms, ffn_down (K 17408, a 37 MB
// activation) 4.4 -> 8.1 ms a call.
template <int BITS, int NPW, int MT, bool PREP, int RG>
__global__ void __launch_bounds__(EXL3R4D_NT_LB) EXL3R4D_NT_WPE_ATTR exl3r4d_gemm_nt_kernel(Exl3r4dGemm p) {
  EXL3R4D_KARG_PREFETCH();
  extern __shared__ float lds[];
  const Planes w = planes_of(p, 0);   // the one weight
  const int rc = (int)blockIdx.y * EXL3_BLOCK * MT * RG;
  if (rc < p.T) nt_pass<BITS, NPW, MT, PREP, RG>(p, w, (int)blockIdx.x, rc, min(rc + EXL3_BLOCK * MT * RG, (int)p.T), lds);
}

// The prefill form's input rotation, once a row: one wave a (row, 128-feature block, half) task -- whole waves
// (rotate128_math's DPP stages need every lane), the region [T][2K] as the GEMM reads it (half 1 only with suh.1).
__global__ void __launch_bounds__(256) exl3r4d_nt_prep_kernel(Exl3r4dGemm p) {
  const Planes w = planes_of(p, 0);
  const int lane = (int)threadIdx.x % WAVE32;
  const int task = (int)blockIdx.x * (256 / WAVE32) + (int)threadIdx.x / WAVE32;
  const int blocks = w.cols / 128, mats = w.suh1 != w.suh ? 2 : 1;
  if (task >= p.T * blocks * mats) return;
  const int mat = task % mats, rest = task / mats, bk = rest % blocks, row = rest / blocks;
  typedef const __attribute__((address_space(1))) v2u* gv2;
  const gu8 xb = (gu8)(const unsigned char*)p.x;
  const gu8 sx = (gu8)(const unsigned char*)(mat ? w.suh1 : w.suh);
  const v2u bx = *(gv2)(xb + ((size_t)row * p.x_pitch + bk * 128 + 4 * lane) * 2);
  const v2u bs = *(gv2)(sx + (size_t)(bk * 128 + 4 * lane) * 2);
  *(v2u*)((_Float16*)p.xr + (size_t)row * p.xr_pitch + (size_t)mat * w.cols + bk * 128 + 4 * lane) =
      rotate128_math(bx, bs, p.x_bf16 == 0, lane);
}

// ============================================================================= T2: the two-pass prefill form (2026-10-30)
// Pass 1 (wdec): every 16 x 16 tile of a column chunk of the weight decoded ONCE into its WMMA fragment order,
// [ksteps][chunk tiles][32 lanes] v8h (decode_frag's bits), in the op's scratch. Pass 2 (t2 GEMM): a workgroup is
// 128 rows x 256 columns, 8 waves (2 along M x 4 along N), each 64 rows x 64 columns (MT 4 x NPW 4 WMMA tiles):
// the rotated activation rows (the prep kernel's region) staged through LDS in fragment order (one 16-B read a
// fragment), the decoded weight fragments read straight from global a chunk (KC steps) ahead into registers, one
// barrier a chunk; row blocks grouped (GROUP) so the workgroups in flight share a column block's weight in L2.
// Per output: one WMMA a 16 k in ascending order from zero, the same fragments as nt_pass; the epilogue is
// epilogue_block with nt_pass's EPIC sum order, so the output is BYTE-IDENTICAL to nt_pass (checked: ntbench
// out-hash at 600..2048 rows, every K, gate_up/down/GDN/head shapes; calib-A KLD scores byte-identical at TP1 and TP2,
// and on a 64k-token document). Below EXL3R4D_T2_MIN rows nt_pass is faster (the whole weight's decode is a fixed
// cost; crossover ~450-512 rows). EXL3R4D_T2_OFF=1 (env) forces nt_pass.
// gfx1201: gate_up 2048x34816x5120 k4 8.08 -> 6.74 ms, down 4.33 -> 3.65, head 248320 rows 65.9 -> 48.1; the GEMM core
// alone ~130-137 TFLOPS (85% of the card's sustained f16 WMMA peak, 158). Standalone core: v3/bench/tlgemm.hip.
#ifndef EXL3R4D_T2
#define EXL3R4D_T2 1
#endif
#define T2_MT 4
#define T2_NPW 4
#define T2_WM 2
#define T2_WN 4
#define T2_KC 2
#define T2_GROUP 8
#define T2_BM (16 * T2_MT * T2_WM)
#define T2_BN (16 * T2_NPW * T2_WN)
#define T2_CHUNK_BYTES EXL3R4D_T2_CHUNK_BYTES   // the decoded weight's scratch: a column chunk of at most this
#if EXL3R4D_NCLS == 2 && !EXL3R4D_GD4
// p.raw: the decoded chunk; p.n_expert (unused by the dense op): the chunk's first column; p.top_k: its columns.
template <int BITS>
__global__ void __launch_bounds__(256) exl3r4d_wdec_kernel(Exl3r4dGemm p) {
  const Planes w = planes_of(p, 0);
  using P = Plan<BITS>;
  constexpr int NW = P::NW, WORDS = P::WORDS;
  const int lane = (int)threadIdx.x % WAVE32;
  const int n0 = p.n_expert, ntc = p.top_k / 16, ntiles = w.rows / 16;
  const size_t tile = (size_t)blockIdx.x * 8 + threadIdx.x / WAVE32;
  if (tile >= (size_t)ntc * (w.cols / 16)) return;
  const size_t ks = tile / ntc, nt = tile % ntc;
  const P plan(lane);
  const gu8 tb = (gu8)(const unsigned char*)w.codes + (ks * ntiles + n0 / 16 + nt) * WORDS * 4;
  unsigned wd[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wd[r][x] = *(gu32)(tb + 4u * (unsigned)plan.woff[r][x]);
  ((v8h*)p.raw)[tile * WAVE32 + lane] = decode_frag<BITS, NW>(wd, plan.shift);
}

__global__ void __launch_bounds__(256) exl3r4d_gemm_t2_kernel(Exl3r4dGemm p) {
  extern __shared__ float lds[];
  const Planes w = planes_of(p, 0);
  constexpr int MT = T2_MT, NPW = T2_NPW, WN = T2_WN, KC = T2_KC, BM = T2_BM, BN = T2_BN, NTH = 256;
  constexpr int RT = BM / 16, ASLAB = RT * KC * WAVE32 * 8;   // halfs, [RT][KC][32] v8h
  constexpr int NLA = BM * KC * 2 / NTH;
  static_assert(BM * KC * 2 % NTH == 0, "every thread loads the same number of activation chunks");
  const int tid = (int)threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int wm = wave / WN, wn = wave % WN;
  const int M = p.T, n0 = p.n_expert, NC = p.top_k, ntc = NC / 16;
  const int nmb = (M + BM - 1) / BM, nnb = NC / BN, lin = (int)blockIdx.x;
  const int per = T2_GROUP * nnb, first = lin / per * T2_GROUP, r0 = lin % per;
  const int gm = min(nmb - first, T2_GROUP);
  const int mb = first + r0 % gm, nb = r0 / gm;
  const int m0 = mb * BM, ntile0 = nb * (BN / 16), ksteps = w.cols / 16;
  const int width = w.rows;
  const int col0 = n0 + nb * BN;                       // this workgroup's first output column
  const int mat_a = (w.suh1 != w.suh && col0 >= width / 2) ? 1 : 0;
  const _Float16* const xrow = (const _Float16*)p.xr + (size_t)mat_a * w.cols;
  const v8h* const wdec = (const v8h*)p.raw;
  _Float16* const abuf = (_Float16*)lds;
  v4u sa[NLA];
  auto load = [&](int c) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, r = idx / (2 * KC), q = idx % (2 * KC);
      const int gr = m0 + r < M ? m0 + r : M - 1;
      sa[i] = *(const v4u*)(xrow + (size_t)gr * p.xr_pitch + c * KC * 16 + q * 8);
    }
  };
  auto store = [&](int b) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, r = idx / (2 * KC), q = idx % (2 * KC);
      {
        const int rt = r / 16, n = r % 16, u = q / 2, qh = q % 2;
        _Float16* const t0 = abuf + b * ASLAB + (rt * KC + u) * WAVE32 * 8 + qh * 4;
        *(v2u*)(t0 + n * 8) = (v2u){sa[i][0], sa[i][1]};
        *(v2u*)(t0 + (n + 16) * 8) = (v2u){sa[i][2], sa[i][3]};
      }
    }
  };
  v8h wq[KC][NPW];
  auto wfetch = [&](int ks, v8h (&o)[NPW]) {
#pragma unroll
    for (int j = 0; j < NPW; j++) o[j] = wdec[((size_t)ks * ntc + ntile0 + wn * NPW + j) * WAVE32 + lane];
  };
  v8f acc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[t][j] = (v8f)(0.f);
#pragma unroll
  for (int u = 0; u < KC; u++) wfetch(u, wq[u]);
  const int nch = ksteps / KC;
  // chunk c + 2's rows are loaded at the end of iteration c and stored at the end of c + 1: every wait in the loop
  // targets loads a whole iteration old (gfx12's load counter retires in order). tlgemm.hip: 137 vs 125 TFLOPS.
  load(0);
  store(0);
  load(min(1, nch - 1));
  lds_barrier();
  for (int c = 0; c < nch; c++) {
    const v8h* const ab = (const v8h*)(abuf + (c & 1) * ASLAB);
#pragma unroll
    for (int u = 0; u < KC; u++) {
      v8h af[MT];
#pragma unroll
      for (int t = 0; t < MT; t++) af[t] = ab[((wm * MT + t) * KC + u) * WAVE32 + lane];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int t = 0; t < MT; t++) acc[t][j] = mma(wq[u][j], af[t], acc[t][j]);
      wfetch(min((c + 1) * KC + u, ksteps - 1), wq[u]);
    }
    __builtin_amdgcn_sched_barrier(0);   // the store stays after the WMMAs (the compiler hoisted it: a full stall)
    store((c + 1) & 1);
    load(min(c + 2, nch - 1));
    lds_barrier();
  }
  // The epilogue: each 128-column block of the workgroup (waves wn 0..1 and 2..3) rotated as nt_pass's, per row group
  const int nl = lane & 15;
#pragma unroll
  for (int t = 0; t < MT; t++) {
    const int row = m0 + 16 * (wm * MT + t) + nl;
    epilogue_block<NPW, 1, false, false, EXL3R4D_NT_EPIC>(p, w, acc[t], wn % 2, 0, lane, col0 / 128 + wn / 2, row, row < M, true,
                                                lds + (wm * 2 + wn / 2) * 4096);
  }
}
template <int B>
static const void* wdec_fn() { return (const void*)exl3r4d_wdec_kernel<B>; }
#endif

template <int B>
static const void* nt_fn(int npw, int mt) {
  if (mt == EXL3R4D_NT_MT && EXL3R4D_NT_MT != 1)
    return npw == 2 ? (const void*)exl3r4d_gemm_nt_kernel<B, 2, EXL3R4D_NT_MT, true, EXL3R4D_NT_RG> : nullptr;
  if (mt != 1) return nullptr;
  switch (npw) {
    case 1: return (const void*)exl3r4d_gemm_nt_kernel<B, 1, 1, false, 1>;
    case 2: return (const void*)exl3r4d_gemm_nt_kernel<B, 2, 1, false, 1>;
    case 4: return (const void*)exl3r4d_gemm_nt_kernel<B, 4, 1, false, 1>;
    default: return nullptr;
  }
}

// The dense linear over one weight, any M (MT 16-row tiles a workgroup past 16), K 1..8, mul1. `a` carries the GSCR region
// (the row's scratch: [T][2 * K] f16) and a->sk the split (1, first pass); a->sorted / a->off / a->n_expert
// / a->top_k are not read (no routing), a->cls[1] not either (one weight). RAD_OK, or a negative RAD_E_*
// naming what is refused; launch errors come back as RAD_E_DEVICE. Must not allocate and must not
// synchronise.

// ============================================================================= E1b DIAGNOSTIC (v3/e1b, never shipped)
// Activation fake-quant of the rotated rows: after the prep kernel, before the GEMM (both prefill forms), each row of
// the region (each half with suh.1) goes through a symmetric int quantiser and back to f16:
//   s = clip * amax / Q, x' = clamp(rint(x / s), -Q, Q) * s, Q = 2^(b-1) - 1, amax over the whole row (EXL3_AQ_GROUP 0,
//   the per-token scale an int GEMM applies in its epilogue) or over groups of EXL3_AQ_GROUP features.
// Which linears: each weight gets an index on its first launch (keyed by its suh plane's address; the first forward
// step issues them in layer order). EXL3_AQ_MASK holds one char an index: '8' int8, '4' int4, any digit 2..8 that
// width, anything else off. EXL3_AQ_LOG (path): the index -> N x K map, appended on first sight. Rows of a decode
// step (T <= 16, in-GEMM rotation) are not quantised: KLD scoring runs prefill steps only.
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include <algorithm>
__global__ void __launch_bounds__(256) e1b_aq_kernel(_Float16* xr, int pitch, int K, int mats, int bits, int group,
                                                     float clip) {
  const int row = (int)blockIdx.x / mats, mat = (int)blockIdx.x % mats;
  _Float16* const x = xr + (size_t)row * pitch + (size_t)mat * K;
  const float Q = (float)((1 << (bits - 1)) - 1);
  const int tid = (int)threadIdx.x;
  if (group > 0) {
    for (int g = tid; g < K / group; g += 256) {
      _Float16* const xg = x + (size_t)g * group;
      float m = 0.f;
      for (int i = 0; i < group; i++) m = fmaxf(m, fabsf((float)xg[i]));
      const float sc = m > 0.f ? clip * m / Q : 1.f, inv = 1.f / sc;
      for (int i = 0; i < group; i++) xg[i] = (_Float16)(fminf(fmaxf(rintf((float)xg[i] * inv), -Q), Q) * sc);
    }
    return;
  }
  __shared__ float red[256];
  float m = 0.f;
  for (int i = tid; i < K; i += 256) m = fmaxf(m, fabsf((float)x[i]));
  red[tid] = m;
  __syncthreads();
  for (int w = 128; w > 0; w >>= 1) {
    if (tid < w) red[tid] = fmaxf(red[tid], red[tid + w]);
    __syncthreads();
  }
  m = red[0];
  const float sc = m > 0.f ? clip * m / Q : 1.f, inv = 1.f / sc;
  for (int i = tid; i < K; i += 256) x[i] = (_Float16)(fminf(fmaxf(rintf((float)x[i] * inv), -Q), Q) * sc);
}
// the weight's index in issue order (registered on first sight; -1 past 4096 weights); EXL3_AQ_LOG gets the map
static int e1b_index(const Exl3r4dGemm* a) {
  static std::mutex mu;
  static const void* seen[4096];
  static int n = 0;
  std::lock_guard<std::mutex> lk(mu);
  const void* key = a->cls[0].suh;
  int i = 0;
  while (i < n && seen[i] != key) i++;
  if (i == n) {
    if (n == 4096) return -1;
    seen[n++] = key;
    if (const char* lp = getenv("EXL3_AQ_LOG")) {
      if (FILE* f = fopen(lp, "a")) {
        fprintf(f, "%d %d %d %d\n", i, a->cls[0].rows, a->cls[0].cols, a->cls[0].suh1 ? 2 : 1);
        fclose(f);
      }
    }
  }
  return i;
}
// the width for weight i (0: off) from EXL3_AQ_MASK
static int e1b_aq_bits(int i) {
  static const char* mask = getenv("EXL3_AQ_MASK");
  if (!mask || i < 0 || i >= (int)strlen(mask)) return 0;
  const char c = mask[i];
  return c >= '2' && c <= '8' ? c - '0' : 0;
}
// EXL3_AQ_DUMP=DIR: the GEMM inputs as they arrive (before the rotation), for CPU-side activation-quant studies.
// Weight i gets DIR/i.x: up to EXL3_AQ_DUMP_ROWS (default 512) rows, EXL3_AQ_DUMP_STEP (default 64) of them a prefill
// step, evenly spaced over the step's rows, as raw 16-bit values ([rows][K], the input's dtype); DIR/i.meta: "K bf16 mats";
// DIR/i.suh (and i.suh1): the f16 input scales. Synchronous (diagnostic only); decode steps (T <= 16) are skipped.
// EXL3_AQ_DUMP_ONLY=i: dump weight i only (e.g. 336 = the head, for the vocab-tier study).
static void e1b_dump(const Exl3r4dGemm* a, int i, void* stream) {
  static const char* dir = getenv("EXL3_AQ_DUMP");
  static const int cap = getenv("EXL3_AQ_DUMP_ROWS") ? atoi(getenv("EXL3_AQ_DUMP_ROWS")) : 512;
  static const int per = getenv("EXL3_AQ_DUMP_STEP") ? atoi(getenv("EXL3_AQ_DUMP_STEP")) : 64;
  static const int only = getenv("EXL3_AQ_DUMP_ONLY") ? atoi(getenv("EXL3_AQ_DUMP_ONLY")) : -1;
  static int done[4096];
  static std::mutex mu;
  if (!dir || i < 0 || a->T <= 16 || (only >= 0 && i != only)) return;
  std::lock_guard<std::mutex> lk(mu);
  if (done[i] >= cap) return;
  const int K = a->cls[0].cols, take = std::min(std::min(per, cap - done[i]), (int)a->T);
  char fn[512];
  if (done[i] == 0) {
    snprintf(fn, sizeof fn, "%s/%d.meta", dir, i);
    if (FILE* f = fopen(fn, "w")) { fprintf(f, "%d %d %d\n", K, a->x_bf16 ? 1 : 0, a->cls[0].suh1 ? 2 : 1); fclose(f); }
    std::vector<unsigned short> sv(K);
    for (int m = 0; m < (a->cls[0].suh1 ? 2 : 1); m++) {
      hipStreamSynchronize((hipStream_t)stream);
      hipMemcpy(sv.data(), m ? a->cls[0].suh1 : a->cls[0].suh, (size_t)K * 2, hipMemcpyDeviceToHost);
      snprintf(fn, sizeof fn, "%s/%d.suh%s", dir, i, m ? "1" : "");
      if (FILE* f = fopen(fn, "wb")) { fwrite(sv.data(), 2, K, f); fclose(f); }
    }
  }
  std::vector<unsigned short> rows((size_t)take * K);
  hipStreamSynchronize((hipStream_t)stream);
  for (int r = 0; r < take; r++) {
    const int64_t src = (int64_t)r * a->T / take;
    hipMemcpy(rows.data() + (size_t)r * K, (const unsigned short*)a->x + src * a->x_pitch, (size_t)K * 2,
              hipMemcpyDeviceToHost);
  }
  snprintf(fn, sizeof fn, "%s/%d.x", dir, i);
  if (FILE* f = fopen(fn, "ab")) { fwrite(rows.data(), 2, rows.size(), f); fclose(f); }
  done[i] += take;
}
// EXL3_AQ_GROUPS: one char a weight (same indexing as EXL3_AQ_MASK) picking its scale group: '0' per token, '1' 1024,
// '2' 512, '3' 256, '4' 128, '5' 64; overrides EXL3_AQ_GROUP for the weights it covers.
static int e1b_aq_group(int i) {
  static const char* gs = getenv("EXL3_AQ_GROUPS");
  static const int g0 = getenv("EXL3_AQ_GROUP") ? atoi(getenv("EXL3_AQ_GROUP")) : 0;
  if (!gs || i < 0 || i >= (int)strlen(gs)) return g0;
  static const int tab[6] = {0, 1024, 512, 256, 128, 64};
  const int c = gs[i] - '0';
  return c >= 0 && c < 6 ? tab[c] : g0;
}
static int e1b_aq_launch(const Exl3r4dGemm* a, int aqb, int group, void* stream) {
  static const float clip = getenv("EXL3_AQ_CLIP") ? (float)atof(getenv("EXL3_AQ_CLIP")) : 1.0f;
  const int K = a->cls[0].cols, mats = a->cls[0].suh1 ? 2 : 1;
  if (group < 0 || (group > 0 && K % group)) return RAD_E_SHAPE;
  hipLaunchKernelGGL(e1b_aq_kernel, dim3((unsigned)(a->T * mats)), dim3(256), 0, (hipStream_t)stream,
                     (_Float16*)a->xr, (int)a->xr_pitch, K, mats, aqb, group, clip);
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int exl3r4d_launch_nt(const Exl3r4dGemm* a, int bits, int npw, void* stream) {
  const int e1bi = e1b_index(a);
  const int aqb = e1b_aq_bits(e1bi);
  e1b_dump(a, e1bi, stream);
  if (bits < 1 || bits > 8) return RAD_E_DTYPE;
  if (npw != 1 && npw != 2 && npw != 4) return RAD_E_SHAPE;
  const int sk = a->sk > 0 && a->sk <= 8 ? a->sk : 1;
  if (sk != 1) return RAD_E_UNSUPPORTED;   // first pass: the split-K 1 form (the fit logic is the tuning arm)
  const int wv = 8 / npw;
  const int K = a->cls[0].cols;
  int64_t smem = 8 * 256 * 4;   // the epilogue's tiles (one matrix, split-K 1)
  (void)K;
  if (smem < 0) return (int)smem;
  // MT: one 16-row tile a workgroup at decode, EXL3R4D_NT_MT of them past 16 rows (NPW 2, the shipped arm)
  const int mt = a->T > 16 && npw == 2 ? EXL3R4D_NT_MT : 1;
  const int rg = mt > 1 ? EXL3R4D_NT_RG : 1;
  const int rows_wg = 16 * mt * rg;
  if (a->T < 1 || (a->T + rows_wg - 1) / rows_wg > 65535) return RAD_E_UNSUPPORTED;
  if (K % 128 || a->y_cols % 128) return RAD_E_SHAPE;
  if (wv * sk * WAVE32 * (mt > 1 ? EXL3R4D_NT_RG : 1) > (int)EXL3R4D_NT_LB) return RAD_E_SHAPE;
  if (!a->xr) return RAD_E_INVAL;   // the GSCR region (the row's scratch hook)
  Exl3r4dGemm p = *a;
  p.sk = sk;
  void* args[] = {&p};
  const void* fn = bits == 1 ? nt_fn<1>(npw, mt) : bits == 2 ? nt_fn<2>(npw, mt) : bits == 3 ? nt_fn<3>(npw, mt) : bits == 4 ? nt_fn<4>(npw, mt)
                  : bits == 5 ? nt_fn<5>(npw, mt) : bits == 6 ? nt_fn<6>(npw, mt) : bits == 7 ? nt_fn<7>(npw, mt)
                  : nt_fn<8>(npw, mt);
  if (!fn) return RAD_E_SHAPE;
#if EXL3R4D_T2
  // The two-pass form: needs the decoded chunk's scratch past the GSCR region (exl3r4d_nt_scratch sizes both).
  if (a->T >= EXL3R4D_T2_MIN && a->raw && !getenv("EXL3R4D_T2_OFF") && a->y_cols % T2_BN == 0 && K % (16 * T2_KC) == 0) {
    const int mats = a->cls[0].suh1 ? 2 : 1;
    const long long tasks = (long long)a->T * (K / 128) * mats;
    if (hipLaunchKernel((const void*)exl3r4d_nt_prep_kernel, dim3((unsigned)((tasks + 7) / 8)), dim3(256), args, 0,
                        (hipStream_t)stream) != hipSuccess)
      return RAD_E_DEVICE;
    if (aqb) { const int e = e1b_aq_launch(a, aqb, e1b_aq_group(e1bi), stream); if (e != RAD_OK) return e; }
    const void* df = bits == 1 ? wdec_fn<1>() : bits == 2 ? wdec_fn<2>() : bits == 3 ? wdec_fn<3>() : bits == 4 ? wdec_fn<4>()
                   : bits == 5 ? wdec_fn<5>() : bits == 6 ? wdec_fn<6>() : bits == 7 ? wdec_fn<7>() : wdec_fn<8>();
    int nc = (int)(T2_CHUNK_BYTES / ((long long)K * 2)) / T2_BN * T2_BN;
    if (nc < T2_BN) nc = T2_BN;
    p.raw = a->raw;
    for (int c0 = 0; c0 < a->y_cols; c0 += nc) {
      const int ncols = min(nc, a->y_cols - c0);
      p.n_expert = c0;
      p.top_k = ncols;
      const size_t tiles = (size_t)(ncols / 16) * (K / 16);
      if (hipLaunchKernel(df, dim3((unsigned)((tiles + 7) / 8)), dim3(256), args, 0, (hipStream_t)stream) != hipSuccess)
        return RAD_E_DEVICE;
      const unsigned grid = (unsigned)(((a->T + T2_BM - 1) / T2_BM) * (ncols / T2_BN));
      if (hipLaunchKernel((const void*)exl3r4d_gemm_t2_kernel, dim3(grid), dim3(256), args, (size_t)65536,
                          (hipStream_t)stream) != hipSuccess)
        return RAD_E_DEVICE;
    }
    return RAD_OK;
  }
#endif
  if (mt > 1 && EXL3R4D_NT_ALDS) {   // the slab's two buffers (the epilogue's tiles reuse them)
    const int64_t slab = 2LL * 16 * mt * rg * EXL3R4D_NT_AP * 2;
    if (smem < slab) smem = slab;
    if (smem < 16384LL * rg) smem = 16384LL * rg;   // each row group's epilogue tiles
  }
  if (mt > 1) {   // the prefill form: the rows rotated once, by their own dispatch (stream-ordered before the GEMM)
    const int mats = a->cls[0].suh1 ? 2 : 1;
    const long long tasks = (long long)a->T * (K / 128) * mats;
    if (hipLaunchKernel((const void*)exl3r4d_nt_prep_kernel, dim3((unsigned)((tasks + 7) / 8)), dim3(256), args, 0,
                        (hipStream_t)stream) != hipSuccess)
      return RAD_E_DEVICE;
    if (aqb) { const int e = e1b_aq_launch(a, aqb, e1b_aq_group(e1bi), stream); if (e != RAD_OK) return e; }
  }
  dim3 grid((unsigned)(a->y_cols / 128), (unsigned)((a->T + rows_wg - 1) / rows_wg));
  dim3 block((unsigned)(wv * sk * WAVE32 * rg));
  if (hipLaunchKernel(fn, grid, block, args, (size_t)smem, (hipStream_t)stream) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}
#endif  // EXL3R4D_NCLS == 2 && !EXL3R4D_GD4
