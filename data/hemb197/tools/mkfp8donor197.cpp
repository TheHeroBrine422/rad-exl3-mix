// mkfp8donor197.cpp -- #197 lead note (fp8 arm f): a one-weight .rad donor whose weight is quantised by libquant's `rtn`
// quantiser exactly as rad-convert would for the recipe line
//     NAME rtn:codes=fp8_e4m3,block=*x*,scale=f32,rule=absmax
// (E4M3 codes with ONE f32 scale for the whole table: the table form embed_lookup_q reads). The quantiser is called
// through its plugin ABI (dlopen libquant.so, rad_quant_at, RadQuantizerInfo.encoding / .quantize), the planes are
// written with the core's RadWriter.
//   mkfp8donor197 PLUGIN.so IN.bf16 ROWS COLS OUT.rad NAME
#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "format/encoding.h"
#include "format/radfile.h"

using namespace rad;

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    if (argc != 7) { std::fprintf(stderr, "usage: mkfp8donor197 PLUGIN.so IN.bf16 ROWS COLS OUT.rad NAME\n"); return 2; }
    const char* plugin = argv[1]; const char* in = argv[2];
    const int64_t rows = std::atoll(argv[3]), cols = std::atoll(argv[4]);
    const char* out = argv[5]; const char* name = argv[6];

    void* h = dlopen(plugin, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto qcount = (int (*)(void))dlsym(h, "rad_quant_count");
    auto qat = (const RadQuantizerInfo* (*)(int))dlsym(h, "rad_quant_at");
    const RadQuantizerInfo* rtn = nullptr;
    for (int i = 0; i < qcount(); i++) { const RadQuantizerInfo* q = qat(i); if (q && !std::strcmp(q->name, "rtn")) rtn = q; }
    if (!rtn) { std::fprintf(stderr, "no rtn quantiser\n"); return 1; }

    RadParam opts[] = { RAD_STR("codes", "fp8_e4m3"), RAD_STR("block", "*x*"), RAD_STR("scale", "f32"), RAD_STR("rule", "absmax") };
    const int n_o = 4;
    RadQuantWeight w{};
    w.name = name; w.rank = 2; w.shape[0] = rows; w.shape[1] = cols; w.rows = rows; w.cols = cols; w.layer = -1; w.expert = -1; w.importance = nullptr;
    RadEncoding enc{};
    int rc = rtn->encoding(opts, n_o, &w, &enc);
    if (rc != 0) { std::fprintf(stderr, "rtn encoding rc %d\n", rc); return 1; }
    std::fprintf(stderr, "encoding: %s (%u planes)\n", enc_name(enc).c_str(), (unsigned)enc.n_planes);

    // source: the whole table in f32 (the scale is table-wide, so one call)
    std::vector<float> src((size_t)rows * cols);
    {
        FILE* f = std::fopen(in, "rb");
        if (!f) { std::perror(in); return 1; }
        std::vector<uint16_t> row((size_t)cols);
        for (int64_t r = 0; r < rows; r++) {
            if (std::fread(row.data(), 2, (size_t)cols, f) != (size_t)cols) { std::fprintf(stderr, "short read\n"); return 1; }
            for (int64_t c = 0; c < cols; c++) src[(size_t)(r * cols + c)] = bf2f(row[(size_t)c]);
        }
        std::fclose(f);
    }

    RadWriter wr;
    std::string tmp = std::string(out) + ".part";
    if (wr.begin(tmp.c_str()) != 0) { std::fprintf(stderr, "begin failed\n"); return 1; }
    wr.set_arch("qwen35"); wr.set_model_name("Qwen/Qwen3.8-27B"); wr.set_quant("w185-fp8-donor");
    wr.set_created_by("w185 #197 mkfp8donor197 (rtn fp8_e4m3, one f32 scale)");
    wr.meta_i("n_vocab", rows); wr.meta_i("n_embd", cols);
    RadWriter::Weight wt;
    wt.name = name; wt.quantizer = "rtn"; wt.options = "block=*x*,codes=fp8_e4m3,rule=absmax,scale=f32";
    wt.enc = enc; wt.rank = 2; wt.shape[0] = rows; wt.shape[1] = cols;
    int64_t idx = wr.add_weight(wt);
    if (idx < 0) { std::fprintf(stderr, "add_weight: %lld\n", (long long)idx); wr.abort(); return 1; }
    std::vector<std::vector<uint8_t>> pl(enc.n_planes);
    std::vector<void*> ptr(enc.n_planes);
    for (uint32_t k = 0; k < enc.n_planes; k++) {
        int64_t nb = wr.plane_bytes(idx, (int)k);
        if (nb <= 0) { std::fprintf(stderr, "plane %u bytes %lld\n", k, (long long)nb); wr.abort(); return 1; }
        pl[k].assign((size_t)nb, 0); ptr[k] = pl[k].data();
    }
    rc = rtn->quantize(opts, n_o, &w, src.data(), 0, rows, ptr.data());
    if (rc != 0) { std::fprintf(stderr, "rtn quantize rc %d\n", rc); wr.abort(); return 1; }
    for (uint32_t k = 0; k < enc.n_planes; k++)
        if (wr.write_plane(idx, (int)k, 0, pl[k].data(), (int64_t)pl[k].size()) != 0) { std::fprintf(stderr, "write_plane %u\n", k); wr.abort(); return 1; }
    if (wr.finish() != 0) { std::fprintf(stderr, "finish failed\n"); return 1; }
    if (std::rename(tmp.c_str(), out) != 0) { std::perror("rename"); return 1; }
    for (uint32_t k = 0; k < enc.n_planes; k++) std::fprintf(stderr, "plane %u: %zu bytes\n", k, pl[k].size());
    std::fprintf(stderr, "wrote %s\n", out);
    return 0;
}
