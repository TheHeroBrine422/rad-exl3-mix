// exl3i_real.cpp -- import one weight from a real level directory through exl3quant.so's "exl3i" quantiser and write
// its decode (f32 [rows][cols]) plus W_hat statistics. Usage:
//   exl3i_real PLUGIN.so LEVEL_DIR CODEBOOK NAME ROWS COLS OUT.f32
// Prints the encoding's transform and plane sizes. Exit 1 on any plugin error.
#include "rad_quant.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 8) { std::fprintf(stderr, "usage: exl3i_real PLUGIN.so LEVEL_DIR CODEBOOK NAME ROWS COLS OUT.f32\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto nq = (int (*)())dlsym(h, "rad_quant_count");
    auto at = (const RadQuantizerInfo* (*)(int))dlsym(h, "rad_quant_at");
    const RadQuantizerInfo* Q = nullptr;
    for (int i = 0; i < nq(); ++i) if (!std::strcmp(at(i)->name, "exl3i")) Q = at(i);
    if (!Q) { std::fprintf(stderr, "no exl3i quantiser\n"); return 1; }
    RadParam p[2] = {};
    p[0].key = "src"; p[0].kind = RAD_P_STR; p[0].sval = argv[2];
    p[1].key = "codebook"; p[1].kind = RAD_P_STR; p[1].sval = argv[3];
    RadQuantWeight w{};
    w.name = argv[4]; w.rows = std::atoll(argv[5]); w.cols = std::atoll(argv[6]); w.rank = 2;
    RadEncoding e;
    int rc = Q->encoding(p, 2, &w, &e);
    if (rc != RAD_OK) { std::fprintf(stderr, "encoding: %d\n", rc); return 1; }
    std::printf("%s %s/%s planes", w.name, e.scheme, e.transform);
    std::vector<std::vector<char>> pl(e.n_planes);
    std::vector<void*> ptr(e.n_planes);
    for (int i = 0; i < e.n_planes; ++i) {
        pl[i].assign((size_t)rad_enc_plane_bytes(&e.plane[i], w.rows, w.cols), 0);
        ptr[i] = pl[i].data();
        std::printf(" %s:%zu", e.plane[i].role, pl[i].size());
    }
    std::printf("\n");
    if ((rc = Q->quantize(p, 2, &w, nullptr, 0, w.rows, ptr.data())) != RAD_OK) { std::fprintf(stderr, "quantize: %d\n", rc); return 1; }
    std::vector<const void*> cp(ptr.begin(), ptr.end());
    std::vector<float> out((size_t)w.rows * w.cols);
    if ((rc = Q->decode(&e, &w, cp.data(), 0, w.rows, out.data())) != RAD_OK) { std::fprintf(stderr, "decode: %d\n", rc); return 1; }
    FILE* f = std::fopen(argv[7], "wb");
    if (!f || std::fwrite(out.data(), 4, out.size(), f) != out.size()) { std::fprintf(stderr, "write\n"); return 1; }
    std::fclose(f);
    return 0;
}
