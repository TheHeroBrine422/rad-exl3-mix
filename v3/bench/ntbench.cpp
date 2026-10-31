// ntbench.cpp -- the exl3r4d dense row outside radiance: dlopen a built exl3_r4d.so and time exl3r4d_launch_nt
// at one shape. usage: ntbench PLUGIN.so M N K BITS [iters] [stacked 0|1]
// Weights are random words (any word decodes to an f16 of the mul1 codebook): timing only.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "exl3r4d_dev.h"
typedef int (*LaunchNt)(const Exl3r4dGemm*, int, int, void*);
#define CK(x) do { if ((x) != hipSuccess) { std::printf("fail %s\n", #x); return 1; } } while (0)
int main(int argc, char** argv) {
    void* so = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!so) { std::printf("dlopen: %s\n", dlerror()); return 1; }
    auto launch = (LaunchNt)dlsym(so, "exl3r4d_launch_nt");
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), bits = atoi(argv[5]);
    const int it = argc > 6 ? atoi(argv[6]) : 20, stacked = argc > 7 ? atoi(argv[7]) : 0;
    const size_t cb = (size_t)N * K * bits / 8;
    void *codes, *suh, *suh1, *svh, *x, *y, *xr;
    CK(hipMalloc(&codes, cb)); CK(hipMalloc(&suh, K * 2)); CK(hipMalloc(&suh1, K * 2)); CK(hipMalloc(&svh, N * 2));
    CK(hipMalloc(&x, (size_t)M * K * 2)); CK(hipMalloc(&y, (size_t)M * N * 2)); CK(hipMalloc(&xr, (size_t)M * 2 * K * 2));
    std::vector<unsigned> h(cb / 4);
    unsigned s = 12345;
    for (auto& v : h) { s = s * 1664525u + 1013904223u; v = s; }
    CK(hipMemcpy(codes, h.data(), cb, hipMemcpyHostToDevice));
    std::vector<unsigned short> one(std::max(N, K) + 1, 0x3c00);   // f16 1.0
    CK(hipMemcpy(suh, one.data(), K * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(suh1, one.data(), K * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(svh, one.data(), N * 2, hipMemcpyHostToDevice));
    std::vector<unsigned short> xh((size_t)M * K);
    for (auto& v : xh) { s = s * 1664525u + 1013904223u; v = 0x3000 | (s >> 22); }   // small positive f16
    CK(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    Exl3r4dGemm d; std::memset(&d, 0, sizeof d);
    d.cls[0].codes = codes; d.cls[0].suh = suh; d.cls[0].suh1 = stacked ? suh1 : nullptr; d.cls[0].svh = svh;
    d.cls[0].codes_s0 = d.cls[0].suh_s0 = d.cls[0].svh_s0 = d.cls[0].suh1_s0 = 1;
    d.cls[0].rows = N; d.cls[0].cols = K; d.cls[1] = d.cls[0];
    d.x = x; d.y = y; d.xr = xr; d.xr_pitch = 2 * K; d.x_pitch = K; d.y_pitch = N; d.T = M; d.n_tok = M;
    d.n_expert = 1; d.top_k = 1; d.n_class = 2; d.y_cols = N; d.x_bf16 = 0; d.sk = 1; d.chunk = 16; d.nslot = 1;
    if (!getenv("NO_T2")) {   // the two-pass prefill form's decoded-weight chunk (the op's scratch past the region)
        void* raw; CK(hipMalloc(&raw, (size_t)EXL3R4D_T2_CHUNK_BYTES)); d.raw = (float*)raw;
    }
    hipStream_t st; CK(hipStreamCreate(&st));
    int rc = launch(&d, bits, 2, st);
    if (rc) { std::printf("launch rc %d\n", rc); return 1; }
    CK(hipStreamSynchronize(st));
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a, st);
    for (int i = 0; i < it; ++i) launch(&d, bits, 2, st);
    hipEventRecord(b, st); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b); ms /= it;
    std::vector<unsigned short> yh(16);
    CK(hipMemcpy(yh.data(), y, 32, hipMemcpyDeviceToHost));
    unsigned long long ck = 0;
    { std::vector<unsigned short> all((size_t)M * N); CK(hipMemcpy(all.data(), y, all.size() * 2, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < all.size(); ++i) ck = ck * 1099511628211ull ^ all[i]; }
    if (const char* o = getenv("OUT")) {   // the whole y (bf16) for a numeric comparison
        std::vector<unsigned short> all((size_t)M * N); CK(hipMemcpy(all.data(), y, all.size() * 2, hipMemcpyDeviceToHost));
        FILE* f = std::fopen(o, "wb"); std::fwrite(all.data(), 2, all.size(), f); std::fclose(f);
    }
    std::printf("M=%d N=%d K=%d k%d: %.3f ms  %.1f TFLOPS  out-hash %016llx\n", M, N, K, bits, ms, 2.0 * M * N * K / ms / 1e9, ck);
    return 0;
}
