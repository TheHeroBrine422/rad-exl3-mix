// madgemv.hip -- #177 (lead, 2026-11-13): ceiling of an exl3i-layout decode GEMV (T 1, int8 activations a8g128)
// with IQR's mad code in its byte-sum form: value = bytesum((st * A + B) & 0x3F3F3F3F) - 126, and the decode folded
// into the dot: sudot4(h, x_c * 0x01010101) = x_c * bytesum(h). No WMMA, no unpacking.
//
// Layout (a proposal; the quantizer would encode in this scan): 16 x 16 tiles [N/16][K/16], each tile one tail-biting
// trellis path of 256 positions at KB bits a position (L16 state = the 16 bits ending at the position's last bit,
// wrapping inside the tile), position p -> row p / 16, column p % 16. Lane l of a wave owns positions 8l .. 8l + 7 of
// every tile of its tile row: row l / 2, columns (l % 2) * 8 + 0..7.
// Activation side (made once a token by the act-quant kernel, ~4x the int8 bytes): xb[c] = x_c * 0x01010101 (int32 per
// column), xs[g] (f32 per 128-column group), csum[c / 8] = sum of 8 columns' x (the -126 bias).
//
// usage: madgemv N K KB [iters] [KW]   (KW waves a block split K; default 1)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

static constexpr uint32_t A_ = 0xB5297Bu, B_ = 0x5A5A5A5Bu;

__host__ __device__ inline uint32_t mad_h(uint32_t st) { return (st * A_ + B_) & 0x3F3F3F3Fu; }

// bits of tile t (KB * 256 bits = KB * 8 dwords), positions are LSB-first in a little-endian bit stream
template <int KB, int P>
__device__ inline uint64_t lane_win(const uint32_t* __restrict__ tw, uint32_t li) {
    // bits [s - 16, s + P KB) of the tile's stream (wrapping), s = P KB li; inside two dwords after the shift (P KB <= 32)
    constexpr uint32_t NB = KB * 256u, NW = KB * 8u;
    const uint32_t b0 = (P * KB * li + NB - 16u) % NB, d0 = b0 / 32u, off = b0 % 32u;
    const uint32_t a = __builtin_nontemporal_load(tw + d0), b = __builtin_nontemporal_load(tw + (d0 + 1u) % NW);
    return ((uint64_t)a | ((uint64_t)b << 32)) >> off;
}

template <int KB, int KW>
__global__ __launch_bounds__(32 * KW * 4) void madgemv(const uint32_t* __restrict__ W, const int32_t* __restrict__ xb,
                                                     const float* __restrict__ xs, const int32_t* __restrict__ csum,
                                                     const float* __restrict__ rs, float* __restrict__ y, uint32_t N,
                                                     uint32_t K) {
    // P positions a lane (16 at KB 2: a lane a tile row, two tiles a wave step; 8 above: two lanes a row, one tile a step)
    constexpr uint32_t P = KB == 2 ? 16u : 8u, LT = 256u / P, TS = 32u / LT;
    const uint32_t lane = threadIdx.x & 31u, wv = threadIdx.x >> 5;
    const uint32_t trl = wv / KW, kw = wv % KW;
    const uint32_t tr = blockIdx.x * 4u + trl;
    const uint32_t KT = K / 16u, TW = KB * 8u;
    const uint32_t ts = lane / LT, li = lane % LT, col0 = (li * P) % 16u;
    __shared__ float red[4][KW][32];
    float accf = 0.f;
    if (tr < N / 16u) {
        const uint32_t* tw = W + (size_t)tr * KT * TW;
        for (uint32_t g = kw; g < KT / 8u; g += KW) {
            int32_t acc = 0, cs = 0;
#pragma unroll
            for (uint32_t tt = 0; tt < 8u; tt += TS) {
                const uint32_t kt = g * 8u + tt + ts;
                const uint64_t win = lane_win<KB, P>(tw + (size_t)kt * TW, li);
                const uint32_t c0 = kt * 16u + col0;
                int32_t xv[P];
#pragma unroll
                for (uint32_t q = 0; q < P / 4u; ++q) {
                    const int4 v = *(const int4*)(xb + c0 + 4u * q);
                    xv[4 * q] = v.x; xv[4 * q + 1] = v.y; xv[4 * q + 2] = v.z; xv[4 * q + 3] = v.w;
                }
#pragma unroll
                for (uint32_t q = 0; q < P / 8u; ++q) cs += csum[c0 / 8u + q];
#pragma unroll
                for (uint32_t j = 0; j < P; ++j) {
                    const uint32_t st = (uint32_t)(win >> (KB * (j + 1u))) & 0xFFFFu;
                    acc = __builtin_amdgcn_sudot4(false, (int)mad_h(st), true, xv[j], acc, false);
                }
            }
            accf += (float)(acc - 126 * cs) * xs[g];
        }
    }
    // lanes sharing a row: P 8 -> l, l ^ 1 ; P 16 -> l, l ^ 16
    accf += __shfl_xor(accf, P == 8u ? 1 : 16);
    red[trl][kw][lane] = accf;
    __syncthreads();
    const bool owner = P == 8u ? (lane & 1u) == 0u : lane < 16u;
    if (kw == 0 && tr < N / 16u && owner) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < KW; ++k) s += red[trl][k][lane];
        const uint32_t row = tr * 16u + (P == 8u ? lane / 2u : lane);
        y[row] = s * rs[row];
    }
}

// host reference of the same definition
static int bitat(const std::vector<uint32_t>& w, size_t tile_off, uint32_t nbits, int64_t b) {
    b %= nbits; if (b < 0) b += nbits;
    return (w[tile_off + b / 32] >> (b % 32)) & 1;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: madgemv N K KB [iters] [KW]\n"); return 1; }
    const uint32_t N = atoi(argv[1]), K = atoi(argv[2]), KB = atoi(argv[3]);
    const int iters = argc > 4 ? atoi(argv[4]) : 200, KW = argc > 5 ? atoi(argv[5]) : 1;
    const uint32_t KT = K / 16, TW = KB * 8;
    const size_t nw = (size_t)N / 16 * KT * TW;
    std::mt19937 rng(177);
    std::vector<uint32_t> W(nw); for (auto& v : W) v = rng();
    std::vector<int8_t> x(K); for (auto& v : x) v = (int8_t)((int)(rng() % 255) - 127);
    std::vector<int32_t> xb(K), cs(K / 8, 0); std::vector<float> xs(K / 128), rs(N), y(N), yr(N);
    for (uint32_t c = 0; c < K; ++c) { xb[c] = (int32_t)((uint32_t)(uint8_t)x[c] * 0x01010101u); cs[c / 8] += x[c]; }
    for (auto& v : xs) v = 0.01f + (rng() % 100) * 1e-4f;
    for (auto& v : rs) v = 0.5f + (rng() % 100) * 1e-3f;
    // reference
    for (uint32_t r = 0; r < N; ++r) {
        const uint32_t tr = r / 16, rr = r % 16; double s = 0;
        for (uint32_t g = 0; g < K / 128; ++g) {
            long acc = 0;
            for (uint32_t tt = 0; tt < 8; ++tt) {
                const uint32_t kt = g * 8 + tt; const size_t to = ((size_t)tr * KT + kt) * TW;
                for (uint32_t cc = 0; cc < 16; ++cc) {
                    const int64_t p = rr * 16 + cc, last = (p + 1) * KB - 1;
                    uint32_t st = 0; for (int i = 0; i < 16; ++i) st |= bitat(W, to, 256 * KB, last - 15 + i) << i;
                    const uint32_t h = mad_h(st);
                    const int v = (h & 255) + (h >> 8 & 255) + (h >> 16 & 255) + (h >> 24) - 126;
                    acc += (long)v * x[kt * 16 + cc];
                }
            }
            s += (double)acc * xs[g];
        }
        yr[r] = (float)(s * rs[r]);
    }
    uint32_t *dW; int32_t *dxb, *dcs; float *dxs, *drs, *dy;
    CK(hipMalloc(&dW, nw * 4)); CK(hipMalloc(&dxb, K * 4)); CK(hipMalloc(&dcs, K / 8 * 4)); CK(hipMalloc(&dxs, K / 128 * 4));
    CK(hipMalloc(&drs, N * 4)); CK(hipMalloc(&dy, N * 4));
    CK(hipMemcpy(dW, W.data(), nw * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dxb, xb.data(), K * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dcs, cs.data(), K / 8 * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dxs, xs.data(), K / 128 * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(drs, rs.data(), N * 4, hipMemcpyHostToDevice));
    auto launch = [&]() {
        dim3 g((N / 16 + 3) / 4), b(32 * KW * 4);
#define L(kb, kw) if (KB == kb && KW == kw) hipLaunchKernelGGL((madgemv<kb, kw>), g, b, 0, 0, dW, dxb, dxs, dcs, drs, dy, N, K)
        L(2, 1); L(3, 1); L(4, 1); L(2, 2); L(3, 2); L(4, 2); L(2, 4); L(3, 4); L(4, 4);
    };
    launch(); CK(hipDeviceSynchronize());
    CK(hipMemcpy(y.data(), dy, N * 4, hipMemcpyDeviceToHost));
    double me = 0, mr = 0; for (uint32_t r = 0; r < N; ++r) { me = fmax(me, fabs(y[r] - yr[r])); mr = fmax(mr, fabs(yr[r])); }
    // timing: rotate over enough copies of W to defeat the 64 MB L2 / infinity cache (time each launch on its own copy)
    const int NC = (int)fmax(1.0, ceil(256e6 / (nw * 4.0)));
    std::vector<uint32_t*> cp(NC); cp[0] = dW;
    for (int i = 1; i < NC; ++i) { CK(hipMalloc(&cp[i], nw * 4)); CK(hipMemcpy(cp[i], dW, nw * 4, hipMemcpyDeviceToDevice)); }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int i = 0; i < 20; ++i) { dW = cp[i % NC]; launch(); }
    CK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) { dW = cp[i % NC]; launch(); }
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    const double us = 1e3 * ms / iters, bytes = nw * 4.0;
    printf("N %u K %u KB %u KW %d: %.2f us, %.0f GB/s (weights %.1f MB), max err %.3g of %.3g\n", N, K, KB, KW, us,
           bytes / us / 1e3, bytes / 1e6, me, mr);
    return 0;
}
