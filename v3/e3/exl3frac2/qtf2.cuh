#pragma once
// qtf2 -- fractional-rate trellis quantizer, second version (V3 kernel pass, 2026-11-01).
//
// Same search as the fork's quantize_tiles_frac_kernel (branch rxm-frac-quarter): a 256-weight tail-biting ring,
// position i reads D(i) = KA + bit (i mod 16) of MASK new bits, the node between positions i-1 and i is the
// (16 - D(i))-bit overlap of their windows, two passes (rolled by L / 2, then pinned to the first pass's end
// state). For every existing (KA >= 1, MASK) instance the output (indices and values) must equal the old kernel's,
// bit for bit: same candidate order, same half arithmetic, same strict-less tie rule, same argmin ranking.
//
// What changed:
//   - history holds the branch k (KOUT <= 8 bits) as one byte, not the 16-bit predecessor: half the traffic
//     (the predecessor is recomputed in traceback: state = (k << (16 - KOUT)) | out, in = state >> KIN)
//   - cost arrays live in LDS when two of them fit (KA >= 2; at KA = 2 they fill the 64 KB exactly, so the
//     input weight is read from global and the argmin scratch aliases the dead costs, as the integer kernel does)
//   - KA = 0: sub-1 rates (K = n / 16, n even). A 0-bit step keeps the state and reads the codebook at
//     state ^ salt(i), salt(i) = (0x9E37 * ((i & 15) + 1)) & 0xFFFF (i = ring position), so consecutive 0-bit steps
//     do not repeat a value; v3/e3/viterbi.c (CPU simulator) uses the same rule. Decoders must apply it too.
//   - argmin ranking widened by one bit when the last node space is 65536 wide (KA = 0); identical otherwise.
//   - v5 (packed history, KA 1-5): each thread's branches of one position go to ONE 16- or 32-bit store (entry
//     2 j + h at bits (2 j + h) KOUT, j = the thread's pair slot, h = low / high state of the pair) in a row of
//     4 NT bytes, instead of a byte per state: 2-8x less history traffic and no sub-word stores. The traceback reads
//     word (edge >> 1) % NT, slot (edge >> 1) / NT. Same branches, so the same output. KA 0 and KA >= 6 keep bytes.
//   - v5 also tried an LDS-only step barrier (~+1%): REVERTED in v8. The cost arrays are reached through generic
//     pointers (flat loads / stores), which an LDS-scoped fence does not wait for: nondeterministic output at KA 1
//     (qtf2race.py). Every barrier is __syncthreads() again. Test build QTF2_BYTEHIST forces the byte history.
//   - v6: the uncached path (KA >= 5) takes the cached path's packed minimum instead of the strict-less scan with
//     per-candidate branches (same pick, ties to the lowest k). QTF2_SCANMIN restores the scan.
//   - v10 (#142): KA = 0 forward pass with the 0-bit steps fused (qtf2_forward0): each thread keeps its own states'
//     costs in registers between branching positions, global memory and barriers only where costs cross threads.
//     Same operations in the same order per state, so the same output. QTF2_NOFUSE restores the per-step path.

#include <cuda_fp16.h>
#include <cstdint>
#include "util.h"              // exllamav3_ext/: half_uint16 etc., before codebook.cuh (as the fork's kernels do)
#include "util.cuh"
#include "codebook.cuh"
#include <cstring>

// half2 <-> its 32 bits. HIP: __builtin_bit_cast (the gated build, unchanged). CUDA: cuda_fp16's half2 is not trivially
// copyable, which __builtin_bit_cast refuses; memcpy compiles to a register move.
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define QTF2_H2U(x) __builtin_bit_cast(uint32_t, (x))
#define QTF2_U2H(x) __builtin_bit_cast(half2, (x))
#else
__device__ __forceinline__ uint32_t qtf2_h2u(const half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ half2 qtf2_u2h(const uint32_t u) { half2 h; memcpy(&h, &u, 4); return h; }
#define QTF2_H2U(x) qtf2_h2u(x)
#define QTF2_U2H(x) qtf2_u2h(x)
#endif

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

// forward-step barrier: a full __syncthreads() (see the v8 note above; an LDS-only fence raced)
__device__ __forceinline__ void qtf2_lds_barrier() { __syncthreads(); }

// v9 (QTF2_HIOPT): at KA 6-7 the forward / step lambdas were outlined as device functions (84-199 KB each, a 152-byte
// stack frame, 249 VGPRs at KA 7) -- the mixed-rate KA 6 / 7 kernels ran at 17k / 4.5k tiles/s vs 93k / 87k for the
// integer ones. Forced inlining and a bounded candidate unroll at KOUT >= 6; same operations in the same order.
#ifdef QTF2_HIOPT
#define QTF2_INLINE __attribute__((always_inline))
#else
#define QTF2_INLINE
#endif

template <bool B> struct qtf2_bool { static constexpr bool value = B; };

__device__ __forceinline__ uint32_t qtf2_salt(int i) { return (0x9E37u * (uint32_t) ((i & 15) + 1)) & 0xFFFFu; }

// mul1 value of a 16-bit state (decode_3inst<2> = what the old kernel's traceback writes)
__device__ __forceinline__ half qtf2_value(uint32_t state) { return decode_3inst<2>(state); }

// packed history word: BITS used bits -> a 16- or 32-bit store at the thread's slot of a 4 NT-byte row
template <int BITS>
__device__ __forceinline__ void qtf2_hist_store(uint8_t* __restrict__ hist_row, const int thread, const uint32_t word)
{
    if constexpr (BITS <= 16) reinterpret_cast<uint16_t*>(hist_row)[thread] = (uint16_t) word;
    else reinterpret_cast<uint32_t*>(hist_row)[thread] = word;
}

template <int NT, int KIN, int KOUT, bool PACK>
__device__ __forceinline__ void qtf2_step
(
    const half w,
    const half* __restrict__ costs_in,
    half* __restrict__ costs_out,
    uint8_t* __restrict__ hist_row,
    const bool first,
    const int pre_state,
    const bool write_hist,
    const uint32_t salt,              // KIN == 0 only (the 0-bit step's codebook read)
    const int thread,
    const half* __restrict__ lut      // KIN == 0: decode_3inst<2> of all 65536 states (same values), or null
)
{
    constexpr int Kr = 16 - KOUT;
    constexpr int max_q = 1 << KOUT;
    constexpr int edges_out = 65536 >> KOUT;
    const half2 w2 = __half2half2(w);
    const half2 inf2 = __half2half2(H_INF);

    // cost of candidate pair (state0, state0 + 1), state0 even. KIN >= 1: both share in-node state0 >> KIN, the old
    // kernel's exact operations. KIN == 0: in-node = state, each half has its own predecessor; values at state ^ salt
    auto pair_err = [&](int state0, int in0, uint32_t p0, uint32_t p1) -> half2
    {
        half2 dec;
        if constexpr (KIN == 0)
            dec = lut ? __halves2half2(__ldg(lut + ((uint32_t) state0 ^ salt)), __ldg(lut + ((uint32_t) (state0 + 1) ^ salt)))
                      : decode_mul1_product_2(p0, p1);
        else dec = decode_mul1_product_2(p0, p1);
        const half2 dh2 = __hsub2(dec, w2);
        if constexpr (KIN > 0)
        {
            if (first)
            {
                half2 e = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in0 != pre_state) e = inf2;
                return e;
            }
            return __hfma2(dh2, dh2, __half2half2(costs_in[in0]));
        }
        else
        {
            if (first)
            {
                const half2 e = __hmul2(dh2, dh2);
                if (pre_state < 0) return e;
                return __halves2half2(in0 == pre_state ? __low2half(e) : H_INF,
                                      in0 + 1 == pre_state ? __high2half(e) : H_INF);
            }
            return __halves2half2(__hfma(__low2half(dh2), __low2half(dh2), costs_in[in0]),
                                  __hfma(__high2half(dh2), __high2half(dh2), costs_in[in0 + 1]));
        }
    };

    constexpr int ITERS = (edges_out + 2 * NT - 1) / (2 * NT);
    static_assert(!PACK || max_q == 1 || 2 * ITERS * KOUT <= 32, "packed history: one word a thread");
    uint32_t word = 0;
    int it = 0;
    for (int out_edge_idx = 2 * thread; out_edge_idx < edges_out; out_edge_idx += 2 * NT, ++it)
    {
        uint32_t product0, product1;
        if constexpr (KIN == 0)
        {
            product0 = ((uint32_t) out_edge_idx ^ salt) * 0x83DCD12Du;
            product1 = ((uint32_t) (out_edge_idx + 1) ^ salt) * 0x83DCD12Du;
        }
        else
        {
            product0 = out_edge_idx * 0x83DCD12Du;
            product1 = product0 + 0x83DCD12Du;
        }
        half2 min_err2 = pair_err(out_edge_idx, out_edge_idx >> KIN, product0, product1);
        int min_k0 = 0, min_k1 = 0;
#ifndef QTF2_SCANMIN
        if constexpr (KIN > 0)
        {
            // v6: packed minimum (cost bits << 16 | k), as the cached path (v4): for non-negative, non-NaN halves the
            // bit patterns order like the values and ties go to the lowest k -- the strict-less scan's pick, without
            // its per-candidate branches (the uncached path's main cost: K5.5 and up)
            const uint32_t e0 = QTF2_H2U(min_err2);
#ifdef QTF2_HIOPT
            constexpr int QTF2_UNR = max_q > 32 ? 16 : max_q;     // v9: bounded unroll at KOUT >= 6 (code size)
#else
            constexpr int QTF2_UNR = max_q;
#endif
            uint32_t best0 = e0 << 16, best1 = e0 & 0xffff0000u;
            #pragma unroll QTF2_UNR
            for (int k = 1; k < max_q; ++k)
            {
                const int state0 = (k << Kr) | out_edge_idx;
                constexpr uint32_t product_step = 0x83DCD12Du << Kr;
                product0 += product_step;
                product1 += product_step;
                const uint32_t eb = QTF2_H2U(pair_err(state0, state0 >> KIN, product0, product1));
                best0 = min(best0, (eb << 16) | (uint32_t) k);
                best1 = min(best1, (eb & 0xffff0000u) | (uint32_t) k);
            }
            min_err2 = QTF2_U2H((best0 >> 16) | (best1 & 0xffff0000u));
            min_k0 = (int) (best0 & 0xffffu); min_k1 = (int) (best1 & 0xffffu);
        }
        else
#endif
        #pragma unroll
        for (int k = 1; k < max_q; ++k)
        {
            const int state0 = (k << Kr) | out_edge_idx;
            if constexpr (KIN == 0)
            {
                product0 = ((uint32_t) state0 ^ salt) * 0x83DCD12Du;
                product1 = ((uint32_t) (state0 + 1) ^ salt) * 0x83DCD12Du;
            }
            else
            {
                constexpr uint32_t product_step = 0x83DCD12Du << Kr;
                product0 += product_step;
                product1 += product_step;
            }
            const half2 err2 = pair_err(state0, state0 >> KIN, product0, product1);
            if (__hlt(__low2half(err2), __low2half(min_err2)))
            {
                min_err2 = __halves2half2(__low2half(err2), __high2half(min_err2));
                min_k0 = k;
            }
            if (__hlt(__high2half(err2), __high2half(min_err2)))
            {
                min_err2 = __halves2half2(__low2half(min_err2), __high2half(err2));
                min_k1 = k;
            }
        }

        reinterpret_cast<half2*>(costs_out)[out_edge_idx >> 1] = min_err2;
        if constexpr (max_q > 1)
        {
            if constexpr (PACK) word |= ((uint32_t) min_k0 | ((uint32_t) min_k1 << KOUT)) << (2 * it * KOUT);
            else if (write_hist)
            {
                hist_row[out_edge_idx] = (uint8_t) min_k0;
                hist_row[out_edge_idx + 1] = (uint8_t) min_k1;
            }
        }
    }
    if constexpr (PACK && max_q > 1)
        if (write_hist && 2 * thread < edges_out) qtf2_hist_store<2 * ITERS * KOUT>(hist_row, thread, word);
}

template <int KA>
struct Qtf2Geom
{
    static constexpr int edges_max = 65536 >> KA;
    // CACHED (KA 1-4): every codebook value a thread needs, decoded once into registers (two state sets: one per
    // out-width), 1024 threads; KA 1 keeps ONE cost buffer in LDS (64 KB, next costs staged in registers), KA 2-4 two.
    static constexpr bool cached = KA >= 1 && KA <= 4;
    static constexpr bool single = KA == 1;
    static constexpr bool lds = KA >= 1;
    static constexpr int nt = cached ? 1024 : 512;
    // v5 packed history rows of 4 nt bytes (KA 1-5: <= the byte layout's edges_max, so the scratch still fits).
    // v8: used where it measured faster with full barriers (qtf2v7split): KA 1, and integer K (MASK 0) at KA <= 3;
    // the mixed KA 2-5 kernels are faster with the byte history (packing costs more ALU than the stores it saves)
    template <uint32_t MASK>
#ifdef QTF2_BYTEHIST
    static constexpr bool pack = false;
#else
    static constexpr bool pack = KA >= 1 && 4 * nt <= edges_max && (KA == 1 || (MASK == 0 && KA <= 3));
#endif
    template <uint32_t MASK>
    static constexpr int row_bytes = pack<MASK> ? 4 * nt : edges_max;
    static constexpr int shmem = !lds ? 32 * (int) sizeof(int) : (single ? 1 : 2) * edges_max * (int) sizeof(half);
};

// Cached step (KIN, KOUT >= 1): identical arithmetic and candidate order to qtf2_step, values from registers.
// vals[j][k] = codebook pair of states (k << (16 - KOUT)) | out_j, out_j = 2 (thread + j NT). STAGE: the new costs go to
// stage[] (written to the single buffer by the caller after a barrier) instead of costs_out.
template <int NT, int KIN, int KOUT, bool STAGE, int NPS, bool PACK>
__device__ __forceinline__ void qtf2_step_cached
(
    const half w,
    const half* __restrict__ costs_in,
    half* __restrict__ costs_out,
    uint8_t* __restrict__ hist_row,
    const bool first,
    const int pre_state,
    const bool write_hist,
    const int thread,
    const half2 (&vals)[(65536 >> KOUT) / (2 * NT)][1 << KOUT],
    half2 (&stage)[NPS]
)
{
    constexpr int Kr = 16 - KOUT;
    constexpr int NP = (65536 >> KOUT) / (2 * NT);
    constexpr int NC = 1 << KOUT;
    static_assert(!PACK || 2 * NP * KOUT <= 32, "packed history: one word a thread");
    const half2 w2 = __half2half2(w);
    const half2 inf2 = __half2half2(H_INF);
    uint32_t word = 0;
    #pragma unroll
    for (int j = 0; j < NP; ++j)
    {
        const int out = 2 * (thread + j * NT);
        // v4: packed minimum (cost bits << 16 | k): for non-negative, non-NaN halves the bit patterns order like the
        // values and ties go to the lowest k -- the same pick as the strict-less scan (qtf2check verifies it)
        uint32_t best0 = 0xffffffffu, best1 = 0xffffffffu;
        #pragma unroll
        for (int k = 0; k < NC; ++k)
        {
            const int in_edge_idx = ((k << Kr) | out) >> KIN;
            const half2 dh2 = __hsub2(vals[j][k], w2);
            half2 err2;
            if (first)
            {
                err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in_edge_idx != pre_state) err2 = inf2;
            }
            else
                err2 = __hfma2(dh2, dh2, __half2half2(costs_in[in_edge_idx]));
            const uint32_t eb = QTF2_H2U(err2);
            best0 = min(best0, (eb << 16) | (uint32_t) k);
            best1 = min(best1, (eb & 0xffff0000u) | (uint32_t) k);
        }
        const half2 min_err2 = QTF2_U2H((best0 >> 16) | (best1 & 0xffff0000u));
        if constexpr (STAGE) stage[j] = min_err2;
        else reinterpret_cast<half2*>(costs_out)[out >> 1] = min_err2;
        if constexpr (PACK) word |= ((best0 & 0xffu) | ((best1 & 0xffu) << KOUT)) << (2 * j * KOUT);
        else if (write_hist)
            *reinterpret_cast<uint16_t*>(hist_row + out) = (uint16_t) ((best0 & 0xffu) | ((best1 & 0xffu) << 8));
    }
    if constexpr (PACK) if (write_hist) qtf2_hist_store<2 * NP * KOUT>(hist_row, thread, word);
}

template <int NT, int KOUT>
__device__ __forceinline__ void qtf2_fill(half2 (&vals)[(65536 >> KOUT) / (2 * NT)][1 << KOUT], const int thread)
{
    constexpr int Kr = 16 - KOUT;
    #pragma unroll
    for (int j = 0; j < (65536 >> KOUT) / (2 * NT); ++j)
    {
        const int out = 2 * (thread + j * NT);
        #pragma unroll
        for (int k = 0; k < (1 << KOUT); ++k)
        {
            const uint32_t p0 = (uint32_t) ((k << Kr) | out) * 0x83DCD12Du;
            vals[j][k] = decode_mul1_product_2(p0, p0 + 0x83DCD12Du);
        }
    }
}

// a wave-uniform value made opaque in a scalar register (HIP: else the compiler folds its multiply into a 64-bit VALU
// multiply-add per use, a quarter-rate instruction, instead of one scalar multiply)
__device__ __forceinline__ uint32_t qtf2_uniform(uint32_t x)
{
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    asm volatile("" : "+s"(x));
#endif
    return x;
}

// v10 (#142): where qtf2_forward0 stores the 15-bit node n (costs and history bytes): bits 0-1 stay, bits 9-14 -> 2-7,
// bits 2-8 -> 8-14
__device__ __forceinline__ uint32_t qtf2_adr0(uint32_t n)
{
    return (n & 3u) | (((n >> 9) & 63u) << 2) | (((n >> 2) & 127u) << 8);
}

// v10 (#142): one forward pass at KA = 0. A sub-1 state only moves at a branching position, so each thread keeps the
// costs of the states it owns in registers between them: s = tb + mo(m) + r, tb = 8 wave + 1024 lane (thread = 32 wave +
// lane), mo(m) = 128 (m & 7) + 32768 (m >> 3), m < 16, r < 8; pairs (r, r + 1) in acc[4 m + r / 2]. Bit 15 is in m, so
// the two candidates (k = 0 / 1) of the out-edge o = s & 0x7fff of a (0, 1) min step are the thread's own (m, m + 8).
// The lanes of a wave sit on state bits 10-14: the codebook table reads of one instruction then hit distinct LDS banks
// (lanes 8 states apart conflicted ~2x more, a measured 2x slowdown). The 15-bit costs and the history rows are stored
// at qtf2_adr0(node), a bit permutation that makes both the writer's and the reader's accesses 16 / 8-byte vectors with
// the lanes contiguous (the traceback reads the history through it). Per step (D(i), D(i + 1)):
//   (0, 0): acc = fma(d, d, acc) (registers only)
//   (1, 0) fan-out: acc = fma(d, d, costs[s >> 1]) (global read, 8 bytes a thread per m)
//   (0, 1) min: candidates fma(d, d, acc), strict-less pick (qtf2_step's KIN = 0 scan), 15-bit costs (16 bytes a thread
//          per m pair) and history bytes (8) out
//   (1, 1): qtf2_step's KIN = KOUT = 1 step (packed minimum), global in and out (same vector widths)
// qtf2_step's arithmetic per state (half FMA chain in position order, salted / plain reads, first-step masks, the picks)
// is kept, so the output is identical. Barriers only after a global write (ping-pong buffers: a buffer is rewritten two
// writes later, after a barrier that follows every read of it). On return *pa holds the last position's 65536 costs
// (argmin_cost reads them).
// Codebook reads: decode_mul1_product_2's value depends only on the byte sum of the product (0..1020), so they read
// tab[sum] (1024 halves in LDS, filled by the kernel with decode_mul1_product_2 itself: every codebook variant, the same
// values; the 65536 table and the procedural reads were slower here). Products are formed by additions only (no
// 32-bit multiply per value).
template <int NT, uint32_t MASK, int L>
__device__ __forceinline__ void qtf2_forward0
(
    const float* __restrict__ input_tile,
    half** pa,                        // in / out: the last written cost buffer
    half** pb,                        // the other one
    uint8_t* __restrict__ hist,
    const int roll,
    const int pre_state,
    const int thread
)
{
    constexpr int NM = 16;                                    // state groups of 8 a thread
    // step kinds this MASK has (D(i), D(i + 1)) = (0, 0) / (1, 1): the other kernels do not compile them (registers)
    constexpr uint32_t ROT = ((MASK >> 1) | (MASK << 15)) & 0xffffu;  // bit i = D(i + 1)
    constexpr bool HAS00 = (~MASK & ~ROT & 0xffffu) != 0, HAS11 = (MASK & ROT) != 0;
    constexpr uint32_t C = 0x83DCD12Du;
    static_assert(NT == 512 && (MASK & 1u) == 0 && (MASK >> 15) == 1, "KA = 0: 512 threads, position 0 plain, 15 branching");
    half* a = *pa;
    half* b = *pb;
    half2 acc[4 * NM];
    // the byte-sum table: tab[b] = the value of a product with byte sum b (static LDS: its address folds into the
    // loads' offsets). Filled at every pass (2 values a thread)
    __shared__ half tab[1024];
    for (int x = thread; x < 1024; x += NT)
    {
        const uint32_t s = x > 1020 ? 1020 : x;
        const uint32_t p = (s > 255 ? 255 : s) | ((s > 510 ? 255 : s > 255 ? s - 255 : 0) << 8)
                         | ((s > 765 ? 255 : s > 510 ? s - 510 : 0) << 16) | ((s > 765 ? s - 765 : 0) << 24);
        tab[x] = __low2half(decode_mul1_product_2(p, p));
    }
    // the tile's weights as halves, read once (a global load at every step was exposed latency: the waves of a block
    // reach each step together)
    __shared__ half wsh[L];
    for (int x = thread; x < L; x += NT) wsh[x] = __float2half_rn(__ldg(input_tile + x));
    __syncthreads();
    // s0 = tb + mo(m) + 2 j (j < 4). tb is made opaque at every step: else the compiler hoists the per-state addresses
    // and products out of the position loop (64-128 VGPRs) and spills
    uint32_t tb = 8 * (thread >> 5) + 1024 * (thread & 31);
    auto mo = [](int m) { return 128 * (m & 7) + 32768 * (m >> 3); };

    // codebook pair of products (p0, p1): the byte sum x 2 is the table's byte offset
    auto decp = [&](uint32_t p0, uint32_t p1) -> half2
    {
        const uint8_t* t8 = reinterpret_cast<const uint8_t*>(tab);
#ifdef QTF2_F0NOTAB                           // timing-only build (output invalid): no table read
        return __halves2half2(__ushort_as_half((uint16_t) __dp4a(p0, 0x02020202u, 0u)), __ushort_as_half((uint16_t) __dp4a(p1, 0x02020202u, 0u)));
#endif
        return __halves2half2(*reinterpret_cast<const half*>(t8 + __dp4a(p0, 0x02020202u, 0u)),
                              *reinterpret_cast<const half*>(t8 + __dp4a(p1, 0x02020202u, 0u)));
    };
    // pair (s0, s0 + 1), s0 = tb + mo(m) + 2 j, at a 0-bit position (qtf2_step's KIN = 0 read, at s ^ salt): the bit
    // fields are disjoint, so s0 ^ salt = (tb ^ (salt & 0x7c78)) + (2 j ^ (salt & 7)) + (mo(m) ^ (salt & 0x8380)) and
    // p0 = xj[j] + um(m) (xj: 4 per step a thread, um: uniform, 16 per step); (s0 + 1) ^ salt = (s0 ^ salt) ^ 1:
    // p1 = p0 + dc, dc = +-C (by salt's bit 0)
    auto dec0 = [&](const uint32_t (&xj)[4], int m, int j, uint32_t salt, uint32_t dc) -> half2
    {
        const uint32_t p0 = xj[j] + qtf2_uniform(((uint32_t) mo(m) ^ (salt & 0x8380u)) * C);
        return decp(p0, p0 + dc);
    };
    auto xjs = [&](uint32_t (&xj)[4], uint32_t salt)
    {
        const uint32_t xsc = (tb ^ (salt & 0x7c78u)) * C;
        #pragma unroll
        for (int j = 0; j < 4; ++j) xj[j] = xsc + ((uint32_t) (2 * j) ^ (salt & 7u)) * C;
    };
    // first step: d^2, other states than pre_state at infinity (qtf2_step's KIN = 0 first branch)
    auto first_err = [&](uint32_t s0, half2 dh2) -> half2
    {
        const half2 e = __hmul2(dh2, dh2);
        if (pre_state < 0) return e;
        return __halves2half2((int) s0 == pre_state ? __low2half(e) : H_INF, (int) s0 + 1 == pre_state ? __high2half(e) : H_INF);
    };
    // stored addresses (qtf2_adr0): out-node o = tb + mo(m) + 2 j (+ h), m < 8, at wb + 8192 (m & 3) + 256 (j >> 1) +
    // 4 (m >> 2) + 2 (j & 1) (+ h): one 16-byte group holds (m, m + 4) x j pair; in-node s0 >> 1 at rb + rm(m) + j
    const uint32_t lane = thread & 31, wave = thread >> 5;
    const uint32_t wb = 8 * lane + 512 * wave, rb = 4 * lane + 256 * wave;
    auto rm = [](int m) { return 4096 * (m & 7) + 128 * (m >> 3); };
    // (0, 1): min over k (k = 0: m, k = 1: m + 8), 15-bit costs o = tb + mo(m) + (0..7) to a, history bytes
    auto min_step = [&](auto first_c, const half2 w2, const uint32_t salt, uint8_t* hist_row, const bool write_hist)
    {
        constexpr bool first = decltype(first_c)::value;
        half* t = a; a = b; b = t;
        uint32_t xj[4];
        xjs(xj, salt);
        const uint32_t dc = (salt & 1u) ? 0u - C : C;
        #pragma unroll
        for (int g = 0; g < NM / 2; ++g)
        {
            const int mq = g >> 1, jh = g & 1;
            uint32_t cw[4], kw[2] = {0u, 0u};
            #pragma unroll
            for (int x = 0; x < 4; ++x)
            {
                const int m = mq + 4 * (x >> 1), j = 2 * jh + (x & 1);
                const half2 dh0 = __hsub2(dec0(xj, m, j, salt, dc), w2);
                const half2 dh1 = __hsub2(dec0(xj, m + NM / 2, j, salt, dc), w2);
                const uint32_t s0 = tb + mo(m) + 2 * j;
                const half2 e0 = first ? first_err(s0, dh0) : __hfma2(dh0, dh0, acc[4 * m + j]);
                const half2 e1 = first ? first_err(s0 + 32768, dh1) : __hfma2(dh1, dh1, acc[4 * (m + NM / 2) + j]);
                const bool lo = __hlt(__low2half(e1), __low2half(e0)), hi = __hlt(__high2half(e1), __high2half(e0));
                cw[x] = QTF2_H2U(__halves2half2(lo ? __low2half(e1) : __low2half(e0), hi ? __high2half(e1) : __high2half(e0)));
                kw[x >> 1] |= ((lo ? 1u : 0u) | (hi ? 0x100u : 0u)) << (16 * (x & 1));
            }
#ifdef QTF2_F0NOGLOB                           // timing-only build (output invalid): no global costs / barriers
            acc[g] = QTF2_U2H(cw[0] ^ cw[1] ^ cw[2] ^ cw[3] ^ kw[0] ^ kw[1]);
        }
#else
            *reinterpret_cast<uint4*>(a + wb + 8192 * mq + 256 * jh) = make_uint4(cw[0], cw[1], cw[2], cw[3]);
            if (write_hist) *reinterpret_cast<uint2*>(hist_row + wb + 8192 * mq + 256 * jh) = make_uint2(kw[0], kw[1]);
        }
        __syncthreads();
#endif
    };

    for (int i = 0; i < L; ++i)
    {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        asm volatile("" : "+v"(tb));
#else
        asm volatile("" : "+r"(tb));
#endif
        int ri = i + roll; if (ri >= L) ri -= L;
#ifdef QTF2_NOHIST
        const bool write_hist = false;                        // timing-only build (output invalid)
#else
        const bool write_hist = pre_state >= 0 || ri < L / 2;
#endif
        uint8_t* hist_row = hist + (size_t) 65536 * ri;
        const half2 w2 = __half2half2(wsh[ri]);
        const int din = (MASK >> (ri & 15)) & 1, dout = (MASK >> ((ri + 1) & 15)) & 1;
        const uint32_t salt = qtf2_salt(ri);
        const uint32_t dc = (salt & 1u) ? 0u - C : C;
        uint32_t xj[4];
        if (din == 0) xjs(xj, salt);
        if (i == 0)                                           // ri = roll: D(ri) = 0, D(ri + 1) = MASK bit 1
        {
            if constexpr ((MASK >> 1) & 1u) min_step(qtf2_bool<true>{}, w2, salt, hist_row, write_hist);
            else
            {
                #pragma unroll
                for (int m = 0; m < NM; ++m)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[4 * m + j] = first_err(tb + mo(m) + 2 * j, __hsub2(dec0(xj, m, j, salt, dc), w2));
            }
            continue;
        }
        switch (din * 2 + dout)
        {
            case 0:
                if constexpr (HAS00)
                #pragma unroll
                for (int m = 0; m < NM; ++m)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        const half2 dh2 = __hsub2(dec0(xj, m, j, salt, dc), w2);
                        acc[4 * m + j] = __hfma2(dh2, dh2, acc[4 * m + j]);
                    }
                break;
            case 1: min_step(qtf2_bool<false>{}, w2, salt, hist_row, write_hist); break;
            case 2:
            {
                // in = s0 >> 1 = tb / 2 + mo(m) / 2 + j: four costs, one 8-byte load a group
                const uint32_t pb0 = tb * C;
                #pragma unroll
                for (int m = 0; m < NM; ++m)
                {
#ifdef QTF2_F0NOGLOB
                    const uint2 cin = make_uint2(QTF2_H2U(acc[4 * m]), QTF2_H2U(acc[4 * m + 1]));
#else
                    const uint2 cin = *reinterpret_cast<const uint2*>(a + rb + rm(m));
#endif
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        const uint32_t p0 = pb0 + (uint32_t) (mo(m) + 2 * j) * C;
                        const half2 dh2 = __hsub2(decp(p0, p0 + C), w2);
                        const uint32_t cw = (j < 2 ? cin.x : cin.y) >> (16 * (j & 1));
                        acc[4 * m + j] = __hfma2(dh2, dh2, __half2half2(__ushort_as_half((uint16_t) cw)));
                    }
                }
                break;
            }
            default:
            if constexpr (HAS11)
            {
                // (1, 1): qtf2_step<NT, 1, 1> (out o = tb + mo(m) + 2 j, m < 8; k = 0 / 1: state (k << 15) | o, in =
                // state >> 1), with its packed minimum: k = 1 wins a half only if its cost bits are strictly lower
                half* t = a; a = b; b = t;
                const uint32_t pb0 = tb * C;
                #pragma unroll
                for (int g = 0; g < NM / 2; ++g)
                {
                    const int mq = g >> 1, jh = g & 1;
                    uint2 c0[2], c1[2];                       // in-costs of m = mq, mq + 4 (k = 0 / 1)
                    #pragma unroll
                    for (int y = 0; y < 2; ++y)
                    {
#ifdef QTF2_F0NOGLOB
                        c0[y] = make_uint2(QTF2_H2U(acc[g + 8 * y]), QTF2_H2U(acc[g + 8 * y + 1]));
                        c1[y] = make_uint2(QTF2_H2U(acc[g + 8 * y + 2]), QTF2_H2U(acc[g + 8 * y + 3]));
#else
                        c0[y] = *reinterpret_cast<const uint2*>(b + rb + rm(mq + 4 * y));
                        c1[y] = *reinterpret_cast<const uint2*>(b + rb + rm(mq + 4 * y) + 128);
#endif
                    }
                    uint32_t cw[4], kw[2] = {0u, 0u};
                    #pragma unroll
                    for (int x = 0; x < 4; ++x)
                    {
                        const int m = mq + 4 * (x >> 1), j = 2 * jh + (x & 1);
                        const uint32_t p0 = pb0 + (uint32_t) (mo(m) + 2 * j) * C, q0 = p0 + (C << 15);
                        const half2 dh0 = __hsub2(decp(p0, p0 + C), w2), dh1 = __hsub2(decp(q0, q0 + C), w2);
                        const uint2 v0 = c0[x >> 1], v1 = c1[x >> 1];
                        const half i0 = __ushort_as_half((uint16_t) ((j < 2 ? v0.x : v0.y) >> (16 * (j & 1))));
                        const half i1 = __ushort_as_half((uint16_t) ((j < 2 ? v1.x : v1.y) >> (16 * (j & 1))));
                        const uint32_t e0 = QTF2_H2U(__hfma2(dh0, dh0, __half2half2(i0)));
                        const uint32_t e1 = QTF2_H2U(__hfma2(dh1, dh1, __half2half2(i1)));
                        const uint32_t best0 = min(e0 << 16, (e1 << 16) | 1u), best1 = min(e0 & 0xffff0000u, (e1 & 0xffff0000u) | 1u);
                        cw[x] = (best0 >> 16) | (best1 & 0xffff0000u);
                        kw[x >> 1] |= ((best0 & 1u) | ((best1 & 1u) << 8)) << (16 * (x & 1));
                    }
#ifdef QTF2_F0NOGLOB
                    acc[g] = QTF2_U2H(cw[0] ^ cw[1] ^ cw[2] ^ cw[3] ^ kw[0] ^ kw[1]);
                }
#else
                    *reinterpret_cast<uint4*>(a + wb + 8192 * mq + 256 * jh) = make_uint4(cw[0], cw[1], cw[2], cw[3]);
                    if (write_hist) *reinterpret_cast<uint2*>(hist_row + wb + 8192 * mq + 256 * jh) = make_uint2(kw[0], kw[1]);
                }
                __syncthreads();
#endif
                break;
            }
        }
    }
    // the last position (15 mod 16) is a fan-out: its 65536 costs go out for argmin_cost
    { half* t = a; a = b; b = t; }
    #pragma unroll
    for (int m = 0; m < NM; ++m)
        *reinterpret_cast<uint4*>(a + tb + mo(m)) = make_uint4(QTF2_H2U(acc[4 * m]), QTF2_H2U(acc[4 * m + 1]),
                                                                  QTF2_H2U(acc[4 * m + 2]), QTF2_H2U(acc[4 * m + 3]));
    __syncthreads();
    *pa = a; *pb = b;
}

template <int KA, uint32_t MASK, int L = 256>
__global__ __launch_bounds__(Qtf2Geom<KA>::nt, Qtf2Geom<KA>::lds ? 1 : 2)
void qtf2_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr,
    const half2* __restrict__ lut
)
{
    static_assert(L == 256, "the 16-position pattern tiles both halves of the 256 ring");
    static_assert((MASK & 1u) == 0 || KA > 0, "canonical masks keep bit 0 clear");
    using G = Qtf2Geom<KA>;
    constexpr int NT = G::nt;
    constexpr int NW = NT / 32;
    constexpr int KB = KA + 1;
    constexpr int edges_max = G::edges_max;
    constexpr int d0 = KA + (MASK & 1);                       // D(0): out-width of the last position of each pass
    constexpr int edges_last = 65536 >> d0;
    constexpr int HB = edges_last > 32768 ? 6 : 5;            // bits of e >> 10 in the argmin rank

    extern __shared__ uint8_t shbuf[];
    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    uint8_t* hist = reinterpret_cast<uint8_t*>(temp_edges_ptr) + (size_t) L * edges_max * tile_idx;

    half* temp_costs;
    if constexpr (G::lds) temp_costs = reinterpret_cast<half*>(shbuf);
    else temp_costs = temp_costs_ptr + (size_t) 2 * edges_max * tile_idx;
    half* temp_costs_inc = temp_costs + (G::single ? 0 : edges_max);
    int* sh_idx = reinterpret_cast<int*>(shbuf);              // aliases the costs when they are in LDS (dead then)
    constexpr int NPA = G::cached ? (65536 >> KA) / (2 * NT) : 1;
    constexpr int NPB = G::cached ? (65536 >> KB) / (2 * NT) : 1;
    constexpr int NPS = G::single ? NPA : 1;
    half2 valsA[G::cached ? NPA : 1][G::cached ? (1 << KA) : 1];
    half2 valsB[G::cached ? NPB : 1][G::cached ? (1 << KB) : 1];
    half2 stage[NPS];
    if constexpr (G::cached) { qtf2_fill<NT, KA>(valsA, thread); qtf2_fill<NT, KB>(valsB, thread); }

    auto ring = [&](int i, int roll) { int ri = i + roll; if (ri >= L) ri -= L; return ri; };
    auto D = [&](int i) { return KA + (int) ((MASK >> (i & 15)) & 1u); };

    // v7: step 0 is peeled with `first` a compile-time constant. With the 4-way width switch the compiler did not
    // unswitch it, so every candidate of every step kept an `if (first)` branch (55 KB of code, ~1,100 branches vs
    // 15 KB / 14 for MASK 0) -- the mixed-rate kernels ran ~35% slower than integer K. Same arithmetic, same output.
    auto forward = [&](int roll, int pre_state) QTF2_INLINE
    {
#ifndef QTF2_NOFUSE
        if constexpr (KA == 0)                                // v10: 0-bit steps fused (qtf2_forward0)
        {
            qtf2_forward0<NT, MASK, L>(input_tile, &temp_costs, &temp_costs_inc, hist, roll, pre_state, thread);
            return;
        }
#endif
        auto step = [&](auto first_c, int i) QTF2_INLINE
        {
            constexpr bool first = decltype(first_c)::value;
            const int ri = ring(i, roll);
            half* t = temp_costs; temp_costs = temp_costs_inc; temp_costs_inc = t;
#ifdef QTF2_NOHIST
            const bool write_hist = false;                    // timing-only build (output invalid)
#else
            const bool write_hist = pre_state >= 0 || ri < L / 2;
#endif
            uint8_t* hist_row = hist + (size_t) G::template row_bytes<MASK> * ri;
            const half w = __float2half_rn(__ldg(input_tile + ri));
            const int din = (MASK >> (ri & 15)) & 1, dout = (MASK >> ((ri + 1) & 15)) & 1;
            const uint32_t salt = qtf2_salt(ri);
            if constexpr (G::cached)
            {
                // KA 1: one buffer -- read it (costs_in), stage, barrier, write; else ping-pong as below
                half* cin = G::single ? temp_costs : temp_costs_inc;
                half* cout_ = G::single ? temp_costs : temp_costs;
                switch (din * 2 + dout)
                {
                    case 0: qtf2_step_cached<NT, KA, KA, G::single, NPS, G::template pack<MASK>>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsA, stage); break;
                    case 1: qtf2_step_cached<NT, KA, KB, G::single, NPS, G::template pack<MASK>>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsB, stage); break;
                    case 2: qtf2_step_cached<NT, KB, KA, G::single, NPS, G::template pack<MASK>>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsA, stage); break;
                    default: qtf2_step_cached<NT, KB, KB, G::single, NPS, G::template pack<MASK>>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsB, stage); break;
                }
                if constexpr (G::single)
                {
                    qtf2_lds_barrier();
                    const int np = (65536 >> (KA + dout)) / (2 * NT);
                    #pragma unroll
                    for (int j = 0; j < NPS; ++j)
                        if (j < np) reinterpret_cast<half2*>(temp_costs)[thread + j * NT] = stage[j];
                }
                qtf2_lds_barrier();
                return;
            }
            switch (din * 2 + dout)
            {
                case 0: qtf2_step<NT, KA, KA, G::template pack<MASK>>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
                case 1: qtf2_step<NT, KA, KB, G::template pack<MASK>>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
                case 2: qtf2_step<NT, KB, KA, G::template pack<MASK>>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
                default: qtf2_step<NT, KB, KB, G::template pack<MASK>>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
            }
            if constexpr (G::lds) qtf2_lds_barrier(); else __syncthreads();   // KA 0: costs in global
        };
        step(qtf2_bool<true>{}, 0);
        for (int i = 1; i < L; ++i) step(qtf2_bool<false>{}, i);
    };

    auto argmin_cost = [&]()
    {
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges_last; e += NT)
        {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << (5 + HB)) | ((__brev(v & 31) >> 27) << HB) | (e >> 10);
            unsigned key = ((uint32_t) __half_as_ushort(temp_costs[e]) << 16) | rank;
            best = min(best, key);
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        __syncthreads();                                      // costs read before sh_idx (may alias them) is written
        if ((thread & 31) == 0) ((uint32_t*) sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*) sh_idx)[thread] : 0x7c00ffffu;
            #pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        }
        __syncthreads();
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> (5 + HB)) >> 27) << 5) | (__brev((rank >> HB) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int) (((rank & ((1u << HB) - 1)) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge)
    {
#ifdef QTF2_NOBACK
        if (thread == 0) { sh_idx[0] = edge ^ (int) hist[edge]; }   // timing-only build: no traceback chain
        if (false)
#else
        if (thread == 0)
#endif
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                const int kin = D(ri), kout = D(ri + 1);
                int k = 0;
                if (kout)
                {
                    if constexpr (G::template pack<MASK>)
                    {
                        const uint8_t* row = hist + (size_t) G::template row_bytes<MASK> * ri;
                        const int t = (edge >> 1) & (NT - 1), sh = (2 * ((edge >> 1) / NT) + (edge & 1)) * kout;
                        // the store's width follows the bits used at this out-width (qtf2_hist_store)
                        const int np2 = ((65536 >> kout) + 2 * NT - 1) / (2 * NT) * 2;
                        const uint32_t wd = np2 * kout <= 16 ? (uint32_t) reinterpret_cast<const uint16_t*>(row)[t]
                                                             : reinterpret_cast<const uint32_t*>(row)[t];
                        k = (int) ((wd >> sh) & ((1u << kout) - 1));
                    }
#ifndef QTF2_NOFUSE
                    else if constexpr (KA == 0) k = (int) hist[(size_t) edges_max * ri + qtf2_adr0(edge)];   // v10 layout
#endif
                    else k = (int) hist[(size_t) edges_max * ri + edge];
                }
                const uint32_t state = kout ? (((uint32_t) k << (16 - kout)) | (uint32_t) edge) : (uint32_t) edge;
                edge = (int) (state >> kin);
                if (write)
                {
                    output_indices[ri] = (uint16_t) state;
                    output_tile[ri] = __half2float(qtf2_value(kin ? state : state ^ qtf2_salt(ri)));
                }
                else if (ri == 0) break;
            }
            sh_idx[0] = edge;
        }
        __syncthreads();
        const int e = sh_idx[0];
        __syncthreads();                                      // before the next forward overwrites (aliased) costs
        return e;
    };

    forward(L / 2, -1);
    int end_state = backward(L / 2, false, argmin_cost());
    forward(0, end_state);
    backward(0, true, end_state);
}
