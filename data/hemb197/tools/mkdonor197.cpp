// mkdonor197.cpp -- #197 (w185): a one-weight .rad donor holding a plain bf16 2-D weight (output.weight or
// token_embd.weight), for radmix's donor lines, written with the core's RadWriter (the format writer the engine reads).
//
//   mkdonor197 OUT.rad NAME ROWS COLS IN.bf16
//     IN.bf16 is row-major [ROWS, COLS] bf16 raw (dq197 / bf16 checkpoint bytes); the plane is copied as is.
//
// Build: as dq197 (data/hemb197/README.md). No plugin needed: a plain weight is the core's own encoding.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "format/radfile.h"
#include "format/encoding.h"

using namespace rad;

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: mkdonor197 OUT.rad NAME ROWS COLS IN.bf16\n");
        return 2;
    }
    const char* out = argv[1];
    const char* name = argv[2];
    const int64_t rows = std::atoll(argv[3]), cols = std::atoll(argv[4]);
    const char* in = argv[5];
    const int64_t bytes = rows * cols * 2;

    FILE* fi = std::fopen(in, "rb");
    if (!fi) { std::perror(in); return 1; }
    std::fseek(fi, 0, SEEK_END);
    if (std::ftell(fi) != bytes) { std::fprintf(stderr, "%s: %ld bytes, want %lld\n", in, std::ftell(fi), (long long)bytes); return 1; }
    std::fseek(fi, 0, SEEK_SET);

    std::string tmp = std::string(out) + ".part";
    RadWriter w;
    if (w.begin(tmp.c_str()) != 0) { std::fprintf(stderr, "begin %s failed\n", tmp.c_str()); return 1; }
    w.set_arch("qwen35");
    w.set_model_name("Qwen/Qwen3.8-27B");
    w.set_quant("w185-donor");
    w.set_created_by("w185 #197 mkdonor197 (plain bf16 donor)");
    w.meta_i("n_vocab", rows);
    w.meta_i("n_embd", cols);

    RadWriter::Weight wt;
    wt.name = name;
    wt.quantizer = "";
    wt.options = "";
    wt.enc = rad_enc_plain(RAD_BF16);
    wt.rank = 2;
    wt.shape[0] = rows;
    wt.shape[1] = cols;
    int64_t idx = w.add_weight(wt);
    if (idx < 0) { std::fprintf(stderr, "add_weight %s: %lld\n", name, (long long)idx); w.abort(); return 1; }
    std::vector<char> buf(1 << 24);
    int64_t off = 0;
    while (off < bytes) {
        size_t k = std::fread(buf.data(), 1, std::min<int64_t>((int64_t)buf.size(), bytes - off), fi);
        if (k == 0) { std::fprintf(stderr, "short read\n"); w.abort(); return 1; }
        if (w.write_plane(idx, 0, off, buf.data(), (int64_t)k) != 0) { std::fprintf(stderr, "write_plane failed at %lld\n", (long long)off); w.abort(); return 1; }
        off += (int64_t)k;
    }
    std::fclose(fi);
    if (w.finish() != 0) { std::fprintf(stderr, "finish failed\n"); return 1; }
    if (std::rename(tmp.c_str(), out) != 0) { std::perror("rename"); return 1; }
    std::fprintf(stderr, "wrote %s: %s [%lld, %lld] plain bf16\n", out, name, (long long)rows, (long long)cols);
    return 0;
}
