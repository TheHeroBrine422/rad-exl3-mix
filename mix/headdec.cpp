/* headdec -- decode one exl3 weight of a .rad container to a raw f16 file, row-major [rows][cols].
 *
 *   headdec CONTAINER.rad OUT.f16 [NAME]      (NAME default output.weight)
 *
 * The decode is the plugin's host decoder (exl3_decode_rows), as hproxy uses it: what exl3r4d serves, bit for bit.
 * For the vocab-tier head study (W_hat rows mixed across K, scored against the bf16 head on dumped head inputs). */
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rad_format.h"
#include "exl3_encoding.h"
extern "C" {
#include "exl3_decode.h"
}

static inline uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: headdec CONTAINER.rad OUT.f16 [NAME]\n"); return 2; }
    const std::string want = argc > 3 ? argv[3] : "output.weight";
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st; fstat(fd, &st);
    auto* base = (const uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    auto* h = (const RadFileHeader*)base;
    if (h->magic != RAD_MAGIC || h->version != RAD_FORMAT_VER) { fprintf(stderr, "not RAD2\n"); return 1; }
    auto* str = (const char*)base + h->str_off;
    auto* dir = (const RadFileEntry*)(base + h->dir_off);
    auto* pl = (const RadFilePlane*)(base + h->plane_off);
    auto* enc = (const RadEncoding*)(base + h->enc_off);
    for (uint64_t i = 0; i < h->dir_count; ++i) {
        const RadFileEntry& e = dir[i];
        if (want != str + e.name) continue;
        int bits, cb, stacked;
        if (exl3_encoding_read(&enc[e.enc], &bits, &cb, &stacked) != RAD_OK) { fprintf(stderr, "%s: not exl3\n", want.c_str()); return 1; }
        Exl3View v{}; v.rows = e.shape[0]; v.cols = e.shape[1]; v.bits = bits; v.codebook = cb;
        const RadEncoding& en = enc[e.enc];
        for (uint32_t p = 0; p < e.n_planes; ++p) {
            const uint8_t* d = base + pl[e.plane_first + p].offset;
            const char* r = en.plane[p].role;
            if (!strcmp(r, "codes")) { v.codes = d; v.codes_row_bytes = (int64_t)(pl[e.plane_first + p].bytes / v.rows); }
            else if (!strcmp(r, "suh")) v.suh = (const uint16_t*)d;
            else if (!strcmp(r, "suh.1")) v.suh1 = (const uint16_t*)d;
            else if (!strcmp(r, "svh")) v.svh = (const uint16_t*)d;
        }
        const int64_t N = v.rows, K = v.cols, RB = 8192;
        FILE* out = fopen(argv[2], "wb");
        if (!out) { perror(argv[2]); return 1; }
        std::vector<float> wq((size_t)RB * K); std::vector<uint16_t> hb((size_t)RB * K);
        double t0 = omp_get_wtime(); int bad = 0;
        for (int64_t r0 = 0; r0 < N; r0 += RB) {
            const int64_t rb = std::min(RB, N - r0);
            #pragma omp parallel for schedule(dynamic, 1) reduction(+:bad)
            for (int64_t b = 0; b < rb; b += 128) bad += exl3_decode_rows(&v, r0 + b, 128, wq.data() + (size_t)b * K) != EXL3_OK;
            #pragma omp parallel for
            for (size_t j = 0; j < (size_t)rb * K; ++j) hb[j] = f2h(wq[j]);
            fwrite(hb.data(), 2, (size_t)rb * K, out);
        }
        fclose(out);
        if (bad) { fprintf(stderr, "decode failed\n"); return 1; }
        fprintf(stderr, "%s %s: K%d %lld x %lld, %.1fs\n", argv[1], want.c_str(), bits, (long long)N, (long long)K, omp_get_wtime() - t0);
        return 0;
    }
    fprintf(stderr, "%s: no %s\n", argv[1], want.c_str());
    return 1;
}
