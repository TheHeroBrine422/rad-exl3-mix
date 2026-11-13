/* rowcheck144.cpp -- (w144, #174-02) embed_exl3i's host row (== the device row bit for bit, exl3dense_embed_exl3i_gpu_test)
 * on the REAL table: rowcheck144 REC.u32 SUH.f16 SVH.f16 TOK.i32 OUT.bf16 M_RATE CODEBOOK -- the record in the exl3i table
 * layout, suh, svh and token ids as raw little-endian files (rowcheck144.py writes them), every token's row into OUT.
 * Build: g++ -O2 -ffp-contract=off -fopenmp -I<plugin>/exl3dense -I<plugin>/exl3import/{common,include} -I<abi>
 *        rowcheck144.cpp <plugin>/exl3dense/embed_exl3i.cpp <plugin>/exl3import/common/exl3{,i}_decode.c */
#include "embed_exl3i.h"
#include "exl3i_encoding.h"
#include "rad_plugin.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

template <class T> std::vector<T> slurp(const char* p) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::perror(p); std::exit(2); }
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<T> v((size_t)n / sizeof(T));
    if (std::fread(v.data(), sizeof(T), v.size(), f) != v.size()) { std::perror(p); std::exit(2); }
    std::fclose(f);
    return v;
}

RadTensor tensor(void* data, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t acc = 1;
    for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = acc; acc *= t.shape[k]; }
    return t;
}

int main(int argc, char** argv) {
    if (argc != 8) { std::fprintf(stderr, "usage: rowcheck144 REC SUH SVH TOK OUT M CODEBOOK\n"); return 2; }
    auto rec = slurp<uint32_t>(argv[1]); auto suh = slurp<uint16_t>(argv[2]); auto svh = slurp<uint16_t>(argv[3]);
    auto tok = slurp<int32_t>(argv[4]);
    const int m = std::atoi(argv[6]);
    const int64_t ne = (int64_t)suh.size(), nv = (int64_t)svh.size(), M = (int64_t)tok.size();
    std::vector<uint16_t> x((size_t)(M * ne));
    const int64_t chunk = 64;
    int bad = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : bad)
    for (int64_t s = 0; s < M; s += chunk) {
        const int64_t n = s + chunk < M ? chunk : M - s;
        std::vector<RadParam> p = { RAD_INT("M", n), RAD_INT("n_embd", ne), RAD_INT("n_vocab", nv), RAD_INT("m", m),
                                    RAD_STR("codebook", argv[7]), RAD_INT("r16", 0) };
        std::vector<RadTensor> t = { tensor(tok.data() + s, RAD_I32, {n}), tensor(rec.data(), RAD_U32, {nv / 16, exl3i_row_words(ne, m)}),
                                     tensor(suh.data(), RAD_F16, {1, ne}), tensor(svh.data(), RAD_F16, {nv, 1}),
                                     tensor(x.data() + s * ne, RAD_BF16, {n, ne}) };
        RadArgs a{};
        a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
        bad += embed_exl3i_host_launch(&a, nullptr) != RAD_OK;
    }
    FILE* f = std::fopen(argv[5], "wb");
    std::fwrite(x.data(), 2, x.size(), f);
    std::fclose(f);
    std::printf("rowcheck144: %lld rows, %d failed launches\n", (long long)M, bad);
    return bad != 0;
}
