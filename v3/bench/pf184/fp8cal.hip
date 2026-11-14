// fp8cal.hip -- the board's FP8 reference GEMM (libr4d gemm_fp8a8_tiled: block E4M3 W8A8, the official FP8 entry's
// prefill row) and the int8 per-row-scale tile (gemm_i8a8_tiled) at the trunk shapes, M 2048, with RANDOM operands
// (r4d_selftest --perfw8 fills constant bytes, and at a power cap the WMMA's power tracks the data), each launch on
// its own copy of the weight (256 MB of copies: never in L2 / MALL), timed as a burst of back-to-back launches after a
// sustained warm-up (bench_twopass --burst's and r4d_selftest's uncold method). The calibrator for gemm_us: the board
// has FP8 at 3712 us a decoder layer (one rank of two). #184, w160.
// Usage: fp8cal LIBR4D.so [--m 2048] [--tp2] [--reps 5] [--burst 20] [--rows fp8,i8] [--shapes a,b]
// Build: hipcc -O2 --offload-arch=gfx1201 -I<install>/include/radiance fp8cal.hip -o fp8cal -ldl
#include <hip/hip_runtime.h>
#include "rad_plugin.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Shape { const char* name; int N, K; bool colsplit; };
static Shape SH[] = {
    { "gate_up", 34816, 5120, true }, { "down", 5120, 17408, false }, { "gdn_qkv", 10240, 5120, true },
    { "gdn_z", 6144, 5120, true }, { "attn_qg", 12288, 5120, true }, { "attn_k", 1024, 5120, true },
    { "attn_v", 1024, 5120, true }, { "attn_o", 5120, 6144, false }, { "ssm_out", 5120, 6144, false },
};

static uint64_t Z = 0x9E3779B97F4A7C15ull;
static uint64_t rnd() { Z ^= Z << 13; Z ^= Z >> 7; Z ^= Z << 17; return Z; }

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> s) {
    RadTensor t; std::memset(&t, 0, sizeof t);
    t.data = d; t.dtype = dt; t.rank = (uint32_t)s.size();
    int64_t acc = 1;
    for (size_t i = 0; i < s.size(); ++i) t.shape[i] = s[i];
    for (int i = (int)s.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= s[i]; }
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

static void* up(const std::vector<uint8_t>& h) { void* d; CK(hipMalloc(&d, h.size())); CK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice)); return d; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: fp8cal LIBR4D.so [--m 2048] [--tp2] [--reps 5] [--burst 20] [--rows fp8,i8] [--shapes a,b]\n"); return 2; }
    int M = 2048, reps = 5, burst = 20;
    bool tp2 = false;
    std::string rows = "fp8,i8", shapes;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return std::string(argv[++i]); };
        if (a == "--m") M = std::atoi(val().c_str());
        else if (a == "--tp2") tp2 = true;
        else if (a == "--reps") reps = std::atoi(val().c_str());
        else if (a == "--burst") burst = std::atoi(val().c_str());
        else if (a == "--rows") rows = val();
        else if (a == "--shapes") shapes = val();
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    auto has = [](const std::string& l, const char* x) { return l.empty() || ("," + l + ",").find(std::string(",") + x + ",") != std::string::npos; };
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    auto find = [&](const char* n) -> const RadKernelInfo* { for (int i = 0; i < cnt(); ++i) if (!std::strcmp(at(i)->name, n)) return at(i); return nullptr; };
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    std::printf("fp8cal: %s, M %d%s, burst %d x %d reps, random operands, weight copies over 256 MB\n", prop.gcnArchName, M,
                tp2 ? ", one rank of two" : "", burst, reps);
    double lay[2][2] = {{0, 0}, {0, 0}};   // [row][gdn, attn]
    for (Shape s : SH) {
        if (tp2) { if (s.colsplit) s.N /= 2; else s.K /= 2; }
        if (!has(shapes, s.name)) continue;
        const int64_t N = s.N, K = s.K, sk = K / 128, sn = N / 128;
        std::vector<uint8_t> hq((size_t)M * K), hw((size_t)N * K);
        for (auto& v : hq) { v = (uint8_t)rnd(); if ((v & 0x7Fu) == 0x7Fu) v &= 0xFEu; }   // no E4M3 NaN; random int8 too
        for (auto& v : hw) { v = (uint8_t)rnd(); if ((v & 0x7Fu) == 0x7Fu) v &= 0xFEu; }
        std::vector<uint8_t> hqs((size_t)M * sk * 4), hfs((size_t)sn * sk * 2), hrs((size_t)N * sk * 2);
        for (size_t i = 0; i < (size_t)M * sk; ++i) { float f = std::ldexp(1.0f + (rnd() % 256) / 256.0f, -7); std::memcpy(&hqs[i * 4], &f, 4); }
        for (size_t i = 0; i < (size_t)sn * sk; ++i) { uint16_t b = bf16(3e-4f * (1.0f + (rnd() % 256) / 256.0f)); std::memcpy(&hfs[i * 2], &b, 2); }
        for (size_t i = 0; i < (size_t)N * sk; ++i) { uint16_t b = bf16(3e-4f * (1.0f + (rnd() % 256) / 256.0f)); std::memcpy(&hrs[i * 2], &b, 2); }
        void *dq = up(hq), *dqs = up(hqs), *dfs = up(hfs), *drs = up(hrs), *dy;
        CK(hipMalloc(&dy, (size_t)M * N * 2));
        const size_t wb = hw.size();
        const int copies = (int)std::max<size_t>(2, (256ull << 20) / wb + 1);
        void* dw; CK(hipMalloc(&dw, wb * copies));
        for (int c = 0; c < copies; ++c) CK(hipMemcpy((char*)dw + (size_t)c * wb, hw.data(), wb, hipMemcpyHostToDevice));
        for (int r = 0; r < 2; ++r) {
            const char* row = r == 0 ? "gemm_fp8a8_tiled" : "gemm_i8a8_tiled";
            if (!has(rows, r == 0 ? "fp8" : "i8")) continue;
            const RadKernelInfo* k = find(row);
            if (!k) { std::printf("%s absent\n", row); continue; }
            std::vector<RadTensor> ts = r == 0
                ? std::vector<RadTensor>{ T(dq, RAD_F8E4M3, {M, K}), T(dqs, RAD_F32, {M, sk}), T(dw, RAD_F8E4M3, {N, K}), T(dfs, RAD_BF16, {sn, sk}),
                                          T(dy, RAD_BF16, {M, N}), T(nullptr, RAD_F32, {1}), T(nullptr, RAD_U8, {1}) }
                : std::vector<RadTensor>{ T(dq, RAD_I8, {M, K}), T(dqs, RAD_F32, {M, sk}), T(dw, RAD_I8, {N, K}), T(drs, RAD_BF16, {N, sk}),
                                          T(dy, RAD_BF16, {M, N}), T(nullptr, RAD_F32, {1}), T(nullptr, RAD_U8, {1}) };
            ts[5].data = nullptr; ts[5].rank = 0; ts[6].data = nullptr; ts[6].rank = 0;
            std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", r == 0 ? "fp8a8" : "i8a8") };
            RadArgs a; std::memset(&a, 0, sizeof a);
            a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
            void* sc = nullptr;
            if (k->scratch) { const int64_t need = k->scratch(&a); if (need > 0) { CK(hipMalloc(&sc, need)); a.scratch = sc; a.scratch_bytes = need; } }
            if (k->launch(&a, nullptr) != RAD_OK || hipDeviceSynchronize() != hipSuccess) { std::printf("%-8s %s refused\n", s.name, row); continue; }
            int ci = 0;
            auto go = [&]() { ts[2].data = (char*)dw + (size_t)(ci++ % copies) * wb; k->launch(&a, nullptr); };
            for (int i = 0; i < 4 * burst; ++i) go();   // warm-up: the clocks settle under load
            hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
            std::vector<double> t;
            for (int rr = 0; rr < reps; ++rr) {
                CK(hipEventRecord(e0, 0));
                for (int i = 0; i < burst; ++i) go();
                CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); t.push_back(ms * 1000.0 / burst);
            }
            std::sort(t.begin(), t.end());
            const double us = t[t.size() / 2];
            std::printf("%-8s %6lld %6lld %5d  %-17s %9.1f us  %6.1f TOPS\n", s.name, (long long)N, (long long)K, M, row, us, 2.0 * M * N * K / us / 1e6);
            const bool g = std::strcmp(s.name, "attn_qg") && std::strcmp(s.name, "attn_k") && std::strcmp(s.name, "attn_v") && std::strcmp(s.name, "attn_o");
            const bool at_ = std::strcmp(s.name, "gdn_qkv") && std::strcmp(s.name, "gdn_z") && std::strcmp(s.name, "ssm_out");
            if (g) lay[r][0] += us;
            if (at_) lay[r][1] += us;
            if (sc) CK(hipFree(sc));
        }
        CK(hipFree(dq)); CK(hipFree(dqs)); CK(hipFree(dfs)); CK(hipFree(drs)); CK(hipFree(dy)); CK(hipFree(dw));
    }
    for (int r = 0; r < 2; ++r)
        if (lay[r][0] > 0)
            std::printf("layer %-17s GDN %8.1f  attention %8.1f  mean (48:16) %8.1f us\n", r == 0 ? "gemm_fp8a8_tiled" : "gemm_i8a8_tiled",
                        lay[r][0], lay[r][1], (48 * lay[r][0] + 16 * lay[r][1]) / 64);
    return 0;
}
