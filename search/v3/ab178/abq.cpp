/* abq -- one weight through a libquant quantiser outside rad-convert (w142, #178): the radiance libquant plugin is
 * dlopen'ed and its quantiser called exactly as rad-convert calls it (RadQuantWeight + the recipe's key=value options),
 * for a quantiser whose encoding is plain bf16 (gptqsim: GPTQ / RTN on any grid, decoded and stored bf16).
 *
 *   abq LIBQUANT.so QUANTISER NAME ROWS COLS IN.f32 OUT.bf16 [key=value ...]
 *
 * NAME is the weight's logical name (blk.N.ssm_a.weight): libquant reads calib=DIR/gram.r<rank>.<stem>.bin by it.
 * IN.f32: ROWS x COLS row-major f32; OUT.bf16: the plane, ROWS x COLS bf16. Option kinds come from the quantiser's own
 * option table, so the values parse as rad-convert's would. Build: g++ -O2 -std=c++17 -I RAD/include/radiance abq.cpp -ldl */
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "rad_quant.h"

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "abq LIBQUANT.so QUANTISER NAME ROWS COLS IN.f32 OUT.bf16 [key=value ...]\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto count = (int (*)())dlsym(h, "rad_quant_count");
    auto at = (const RadQuantizerInfo* (*)(int))dlsym(h, "rad_quant_at");
    if (!count || !at) { fprintf(stderr, "not a quantiser plugin\n"); return 1; }
    const RadQuantizerInfo* q = nullptr;
    for (int i = 0; i < count(); ++i) if (!strcmp(at(i)->name, argv[2])) q = at(i);
    if (!q) { fprintf(stderr, "no quantiser %s\n", argv[2]); return 1; }
    const int64_t N = atoll(argv[4]), K = atoll(argv[5]);

    std::vector<std::string> keys, vals;
    for (int i = 8; i < argc; ++i) {
        const char* e = strchr(argv[i], '=');
        if (!e) { fprintf(stderr, "option %s: not key=value\n", argv[i]); return 2; }
        keys.emplace_back(argv[i], e - argv[i]); vals.emplace_back(e + 1);
    }
    std::vector<RadParam> o(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const RadQuantOption* opt = nullptr;
        for (int j = 0; j < q->n_options; ++j) if (keys[i] == q->options[j].key) opt = &q->options[j];
        if (!opt) { fprintf(stderr, "%s takes no option %s\n", q->name, keys[i].c_str()); return 2; }
        RadParam p{}; p.key = keys[i].c_str(); p.kind = opt->type;
        if (opt->type == RAD_P_INT) p.ival = atoll(vals[i].c_str());
        else if (opt->type == RAD_P_F64) p.dval = atof(vals[i].c_str());
        else p.sval = vals[i].c_str();
        o[i] = p;
    }
    RadQuantWeight w{}; w.name = argv[3]; w.rank = 2; w.shape[0] = N; w.shape[1] = K; w.rows = N; w.cols = K;
    w.layer = -1; w.expert = -1; w.importance = nullptr;
    RadEncoding enc{};
    int st = q->encoding(o.data(), (int)o.size(), &w, &enc);
    if (st != RAD_OK) { fprintf(stderr, "encoding: %d\n", st); return 1; }
    if (strcmp(enc.scheme, "plain") || enc.n_planes != 1 || enc.plane[0].dtype != RAD_BF16) {
        fprintf(stderr, "encoding %s (%d planes): abq writes plain bf16 only\n", enc.scheme, enc.n_planes); return 1;
    }
    std::vector<float> src((size_t)(N * K));
    FILE* f = fopen(argv[6], "rb");
    if (!f || fread(src.data(), 4, src.size(), f) != src.size()) { fprintf(stderr, "read %s\n", argv[6]); return 1; }
    fclose(f);
    std::vector<uint16_t> out((size_t)(N * K));
    void* planes[1] = { out.data() };
    st = q->quantize(o.data(), (int)o.size(), &w, src.data(), 0, N, planes);
    if (st != RAD_OK) { fprintf(stderr, "quantize: %d\n", st); return 1; }
    f = fopen(argv[7], "wb");
    if (!f || fwrite(out.data(), 2, out.size(), f) != out.size()) { fprintf(stderr, "write %s\n", argv[7]); return 1; }
    fclose(f);
    return 0;
}
