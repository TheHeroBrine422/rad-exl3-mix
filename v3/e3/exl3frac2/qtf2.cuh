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
//   - v5: the per-step barrier (costs in LDS: KA >= 1) waits for LDS only. __syncthreads() also waits for every
//     outstanding global store (s_wait_storecnt), so each of the 512 steps stalled on its history store; the
//     traceback reads the history only after argmin_cost's full barrier. KA 0 (costs in global) keeps the full one.
//     Test builds: QTF2_FULLSYNC / QTF2_BYTEHIST restore the v4 barrier / byte history.
//   - v6: the uncached path (KA >= 5) takes the cached path's packed minimum instead of the strict-less scan with
//     per-candidate branches (same pick, ties to the lowest k). QTF2_SCANMIN restores the scan.

#include <cuda_fp16.h>
#include <cstdint>
#include "util.h"              // exllamav3_ext/: half_uint16 etc., before codebook.cuh (as the fork's kernels do)
#include "util.cuh"
#include "codebook.cuh"

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

// barrier for the forward steps when the costs live in LDS: LDS ordering only (see v5 above)
__device__ __forceinline__ void qtf2_lds_barrier()
{
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__) && !defined(QTF2_FULLSYNC)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

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
            const uint32_t e0 = __builtin_bit_cast(uint32_t, min_err2);
            uint32_t best0 = e0 << 16, best1 = e0 & 0xffff0000u;
            #pragma unroll
            for (int k = 1; k < max_q; ++k)
            {
                const int state0 = (k << Kr) | out_edge_idx;
                constexpr uint32_t product_step = 0x83DCD12Du << Kr;
                product0 += product_step;
                product1 += product_step;
                const uint32_t eb = __builtin_bit_cast(uint32_t, pair_err(state0, state0 >> KIN, product0, product1));
                best0 = min(best0, (eb << 16) | (uint32_t) k);
                best1 = min(best1, (eb & 0xffff0000u) | (uint32_t) k);
            }
            min_err2 = __builtin_bit_cast(half2, (best0 >> 16) | (best1 & 0xffff0000u));
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
    // v5 packed history rows of 4 nt bytes (KA 1-5: <= the byte layout's edges_max, so the scratch still fits)
#ifdef QTF2_BYTEHIST
    static constexpr bool pack = false;                       // the pre-v5 byte history (qtf2v5check.py compares)
#else
    static constexpr bool pack = KA >= 1 && 4 * nt <= edges_max;
#endif
    static constexpr int row_bytes = pack ? 4 * nt : edges_max;
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
            const uint32_t eb = __builtin_bit_cast(uint32_t, err2);
            best0 = min(best0, (eb << 16) | (uint32_t) k);
            best1 = min(best1, (eb & 0xffff0000u) | (uint32_t) k);
        }
        const half2 min_err2 = __builtin_bit_cast(half2, (best0 >> 16) | (best1 & 0xffff0000u));
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

    auto forward = [&](int roll, int pre_state)
    {
        for (int i = 0; i < L; ++i)
        {
            const int ri = ring(i, roll);
            half* t = temp_costs; temp_costs = temp_costs_inc; temp_costs_inc = t;
            const bool first = i == 0;
#ifdef QTF2_NOHIST
            const bool write_hist = false;                    // timing-only build (output invalid)
#else
            const bool write_hist = pre_state >= 0 || ri < L / 2;
#endif
            uint8_t* hist_row = hist + (size_t) G::row_bytes * ri;
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
                    case 0: qtf2_step_cached<NT, KA, KA, G::single, NPS, G::pack>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsA, stage); break;
                    case 1: qtf2_step_cached<NT, KA, KB, G::single, NPS, G::pack>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsB, stage); break;
                    case 2: qtf2_step_cached<NT, KB, KA, G::single, NPS, G::pack>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsA, stage); break;
                    default: qtf2_step_cached<NT, KB, KB, G::single, NPS, G::pack>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsB, stage); break;
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
                continue;
            }
            switch (din * 2 + dout)
            {
                case 0: qtf2_step<NT, KA, KA, G::pack>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
                case 1: qtf2_step<NT, KA, KB, G::pack>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
                case 2: qtf2_step<NT, KB, KA, G::pack>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
                default: qtf2_step<NT, KB, KB, G::pack>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread, (const half*) lut); break;
            }
            if constexpr (G::lds) qtf2_lds_barrier(); else __syncthreads();   // KA 0: costs in global
        }
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
                    if constexpr (G::pack)
                    {
                        const uint8_t* row = hist + (size_t) G::row_bytes * ri;
                        const int t = (edge >> 1) & (NT - 1), sh = (2 * ((edge >> 1) / NT) + (edge & 1)) * kout;
                        // the store's width follows the bits used at this out-width (qtf2_hist_store)
                        const int np2 = ((65536 >> kout) + 2 * NT - 1) / (2 * NT) * 2;
                        const uint32_t wd = np2 * kout <= 16 ? (uint32_t) reinterpret_cast<const uint16_t*>(row)[t]
                                                             : reinterpret_cast<const uint32_t*>(row)[t];
                        k = (int) ((wd >> sh) & ((1u << kout) - 1));
                    }
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
