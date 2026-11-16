// dq197.cpp -- #197 (w185): dequantise one 2-D weight of a KL-IQR .rad container to bf16, through the core's own
// decoder (core/format/encoding.h enc_decode_rows) with the quantiser plugin's decoder registered (the plugin is
// dlopen'ed as radiance_home/quantizers/libquant.so and its rad_quant_at(i)->decode goes to enc_add_decoder, the same
// registration the core's plugin loader does). Nothing else of the radiance engine runs: no device, no GPU.
//
//   dq197 PLUGIN.so IN.rad NAME OUT.bf16 [--raw OUT.bin]
//     OUT.bf16  row-major [rows, cols] bf16 as raw u16 (round to nearest even), rows in the tensor's own order
//     --raw     the tensor's plane 0 bytes as stored (for i32 index tensors such as output.kB.ids);
//               with OUT.bf16 = '-' only the raw bytes are written
//
// Build: see data/hemb197/README.md (same flags as the core's own objects, linked as rad-info is).
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "format/encoding.h"
#include "format/radfile.h"

using namespace rad;

static uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: dq197 PLUGIN.so IN.rad NAME OUT.bf16 [--raw OUT.bin] [--check]\n");
        return 2;
    }
    const char* plugin = argv[1];
    const char* in = argv[2];
    const char* name = argv[3];
    const char* out = argv[4];
    const char* raw = nullptr;
    for (int i = 5; i < argc; i++) {
        if (!std::strcmp(argv[i], "--raw") && i + 1 < argc) raw = argv[++i];
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    void* h = dlopen(plugin, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", plugin, dlerror()); return 1; }
    auto qcount = (int (*)(void))dlsym(h, "rad_quant_count");
    auto qat = (const RadQuantizerInfo* (*)(int))dlsym(h, "rad_quant_at");
    if (!qcount || !qat) { std::fprintf(stderr, "plugin has no rad_quant_count / rad_quant_at\n"); return 1; }
    int nd = 0;
    for (int i = 0; i < qcount(); i++) {
        const RadQuantizerInfo* q = qat(i);
        if (q && q->decode) { enc_add_decoder(q->decode); nd++; }
    }
    std::fprintf(stderr, "dq197: %d quantiser decoder(s) registered from %s\n", nd, plugin);

    RadFile f;
    if (f.open(in) != 0) { std::fprintf(stderr, "open %s failed\n", in); return 1; }
    const RadFileEntry* e = f.find(name);
    if (!e) { std::fprintf(stderr, "no weight %s in %s\n", name, in); return 1; }
    const RadEncoding& enc = f.encoding(*e);
    const void* planes[RAD_ENC_MAX_PLANES] = {};
    for (uint32_t k = 0; k < e->n_planes; k++) planes[k] = f.data(f.plane(*e, (int)k));

    if (raw) {
        const RadFilePlane& p = f.plane(*e, 0);
        FILE* fo = std::fopen(raw, "wb");
        if (!fo) { std::perror(raw); return 1; }
        size_t w = std::fwrite(f.data(p), 1, (size_t)p.bytes, fo);
        std::fclose(fo);
        std::fprintf(stderr, "raw plane 0 of %s: %lld bytes -> %s (wrote %zu)\n", name, (long long)p.bytes, raw, w);
        if (!out || !std::strcmp(out, "-")) return 0;
    }

    if (e->rank != 2) { std::fprintf(stderr, "%s: rank %u, need 2\n", name, e->rank); return 1; }
    const int64_t rows = e->shape[0], cols = e->shape[1];
    std::string enc_str = enc_name(enc);
    std::fprintf(stderr, "%s: [%lld, %lld] encoding %s\n", name, (long long)rows, (long long)cols, enc_str.c_str());

    EncodedView v;
    enc_view(enc, rows, cols, planes, &v);
    FILE* fo = std::fopen(out, "wb");
    if (!fo) { std::perror(out); return 1; }
    const int64_t chunk = 1024;
    std::vector<float> buf((size_t)chunk * cols);
    std::vector<uint16_t> ob((size_t)chunk * cols);
    double sumsq = 0.0, sumabs = 0.0;
    int64_t n_nonfinite = 0;
    for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
        const int64_t n = std::min(chunk, rows - r0);
        std::string why;
        int rc = enc_decode_rows(v, r0, n, buf.data(), &why);
        if (rc != 0) {
            std::fprintf(stderr, "enc_decode_rows rc %d at row %lld: %s\n", rc, (long long)r0, why.c_str());
            return 1;
        }
        for (int64_t i = 0; i < n * cols; i++) {
            float x = buf[(size_t)i];
            if (!std::isfinite(x)) n_nonfinite++;
            sumsq += (double)x * x;
            sumabs += std::fabs((double)x);
            ob[(size_t)i] = f2bf(x);
        }
        if (std::fwrite(ob.data(), 2, (size_t)(n * cols), fo) != (size_t)(n * cols)) { std::fprintf(stderr, "short write\n"); return 1; }
    }
    std::fclose(fo);
    std::fprintf(stderr, "wrote %s: %lld x %lld bf16; rms %.6g mean|x| %.6g nonfinite %lld\n", out, (long long)rows,
                 (long long)cols, std::sqrt(sumsq / ((double)rows * cols)), sumabs / ((double)rows * cols),
                 (long long)n_nonfinite);
    return n_nonfinite ? 1 : 0;
}
