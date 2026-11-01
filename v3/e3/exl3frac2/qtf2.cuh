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

#include <cuda_fp16.h>
#include <cstdint>
#include "util.h"              // exllamav3_ext/: half_uint16 etc., before codebook.cuh (as the fork's kernels do)
#include "util.cuh"
#include "codebook.cuh"

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

__device__ __forceinline__ uint32_t qtf2_salt(int i) { return (0x9E37u * (uint32_t) ((i & 15) + 1)) & 0xFFFFu; }

// mul1 value of a 16-bit state (decode_3inst<2> = what the old kernel's traceback writes)
__device__ __forceinline__ half qtf2_value(uint32_t state) { return decode_3inst<2>(state); }

template <int NT, int KIN, int KOUT>
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
    const int thread
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
        const half2 dh2 = __hsub2(decode_mul1_product_2(p0, p1), w2);
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

    for (int out_edge_idx = 2 * thread; out_edge_idx < edges_out; out_edge_idx += 2 * NT)
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
            if (write_hist)
            {
                hist_row[out_edge_idx] = (uint8_t) min_k0;
                hist_row[out_edge_idx + 1] = (uint8_t) min_k1;
            }
    }
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
    static constexpr int shmem = !lds ? 32 * (int) sizeof(int) : (single ? 1 : 2) * edges_max * (int) sizeof(half);
};

// Cached step (KIN, KOUT >= 1): identical arithmetic and candidate order to qtf2_step, values from registers.
// vals[j][k] = codebook pair of states (k << (16 - KOUT)) | out_j, out_j = 2 (thread + j NT). STAGE: the new costs go to
// stage[] (written to the single buffer by the caller after a barrier) instead of costs_out.
template <int NT, int KIN, int KOUT, bool STAGE, int NPS>
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
    const half2 w2 = __half2half2(w);
    const half2 inf2 = __half2half2(H_INF);
    #pragma unroll
    for (int j = 0; j < NP; ++j)
    {
        const int out = 2 * (thread + j * NT);
        half2 min_err2;
        int min_k0 = 0, min_k1 = 0;
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
            if (k == 0) { min_err2 = err2; continue; }
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
        if constexpr (STAGE) stage[j] = min_err2;
        else reinterpret_cast<half2*>(costs_out)[out >> 1] = min_err2;
        if (write_hist)
        {
            hist_row[out] = (uint8_t) min_k0;
            hist_row[out + 1] = (uint8_t) min_k1;
        }
    }
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
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            uint8_t* hist_row = hist + (size_t) edges_max * ri;
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
                    case 0: qtf2_step_cached<NT, KA, KA, G::single>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsA, stage); break;
                    case 1: qtf2_step_cached<NT, KA, KB, G::single>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsB, stage); break;
                    case 2: qtf2_step_cached<NT, KB, KA, G::single>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsA, stage); break;
                    default: qtf2_step_cached<NT, KB, KB, G::single>(w, cin, cout_, hist_row, first, pre_state, write_hist, thread, valsB, stage); break;
                }
                if constexpr (G::single)
                {
                    __syncthreads();
                    const int np = (65536 >> (KA + dout)) / (2 * NT);
                    #pragma unroll
                    for (int j = 0; j < NPS; ++j)
                        if (j < np) reinterpret_cast<half2*>(temp_costs)[thread + j * NT] = stage[j];
                }
                __syncthreads();
                continue;
            }
            switch (din * 2 + dout)
            {
                case 0: qtf2_step<NT, KA, KA>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread); break;
                case 1: qtf2_step<NT, KA, KB>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread); break;
                case 2: qtf2_step<NT, KB, KA>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread); break;
                default: qtf2_step<NT, KB, KB>(w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, salt, thread); break;
            }
            __syncthreads();
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
        if (thread == 0)
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                const int kin = D(ri), kout = D(ri + 1);
                const int k = kout ? (int) hist[(size_t) edges_max * ri + edge] : 0;
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
