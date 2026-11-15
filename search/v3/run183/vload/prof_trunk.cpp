// prof_trunk.cpp -- #185 w162 step 3: where the exl3i import's time goes, for one trunk weight and for a whole level.
//
// The import is the one radiance's exl3_quant plugin runs in rad-convert (plugin/exl3quant/exl3_quant.cpp, quantize_i and
// encoding_i), reduced to what a load-time import would do: plan (the source's header lookups and the encoding), read
// (the trellis bytes, pread through exl3::Source), reorder (the same per-tile memcpy into the record), and the F16 suh / svh.
// Nothing is written and no GPU is touched.
//
//   prof_trunk DIR one BASE            one weight, phase by phase, cold (page cache dropped) and warm
//   prof_trunk DIR all [THREADS]       every trunk weight of the level, single thread, then THREADS in parallel
//
// BASE is the checkpoint's name of the projection without the suffix, e.g. model.language_model.layers.3.mlp.gate_proj
// (a stacked gate_up is "model.language_model.layers.3.mlp.gate_proj+model.language_model.layers.3.mlp.up_proj").
//
// Build (ds8; the fork's sources copied as-is, see README):
//   g++ -O3 -march=native -std=c++17 -pthread -I FORK/abi -I FORK/plugin/exl3import/include -I FORK/plugin/exl3import/common \
//       -I FORK/plugin/exl3quant exl3_src.cpp prof_trunk.cpp -o prof_trunk
#include "exl3_src.h"
#include "exl3_decode.h"   // EXL3_MUL1_MARKER
#include "exl3i_encoding.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// one stacked or plain weight of the trunk: its parts (1 or 2 projections, gate then up), each with a trellis
struct Weight {
    std::string name;                  // the trellis base name(s), "+"-joined when stacked
    std::vector<std::string> parts;
    int64_t rows = 0, cols = 0;        // rows = total output rows (both parts), cols = input features
};

// the same reorder as exl3_quant.cpp's quantize_i: exllamav3's [kt][nt][m] int16 tile (tk, tn) at (tk nt + tn) tw words
// -> record table row tn, tile tk at tk tw words. Returns the bytes of the record it wrote.
static void reorder_part(const uint32_t* tr, int64_t kt, int64_t nt, int tw, int64_t rw, uint32_t* rec_part) {
    for (int64_t tn = 0; tn < nt; ++tn)
        for (int64_t tk = 0; tk < kt; ++tk)
            std::memcpy(rec_part + tn * rw + tk * tw, tr + (tk * nt + tn) * tw, (size_t)tw * 4);
}

// drop a file's pages from the page cache (cold reads)
static void drop_cache(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    ::fsync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}

// the level's shard files (the index's distinct values), for the cold passes
static std::vector<std::string> shard_files(const std::string& dir) {
    std::string text, err;
    exl3::Json ix;
    if (!exl3::read_file(dir + "/model.safetensors.index.json", &text, &err) ||
        !exl3::parse_json(text.data(), text.data() + text.size(), &ix, &err)) return {};
    std::map<std::string, bool> seen;
    for (const auto& kv : ix.get("weight_map")->obj) seen[kv.second.str] = true;
    std::vector<std::string> out;
    for (const auto& kv : seen) out.push_back(dir + "/" + kv.first);
    return out;
}

struct Phase { double plan = 0, read = 0, reorder = 0, scales = 0; double bytes = 0; double codes_bytes = 0; };

// import one weight: plan, read, reorder, scales (cold: the caller dropped the level's pages). Returns false with `err`.
static bool import_weight(exl3::Source* src, const Weight& w, Phase* ph, std::vector<uint32_t>* rec,
                          std::vector<uint8_t>* raw, std::vector<uint16_t>* scales, std::string* err) {
    const int n_part = (int)w.parts.size();
    const int64_t n = w.rows / n_part, k = w.cols;
    // ---- plan: the source's lookups for every part and the encoding (what encoding_i / front_i do per weight)
    auto t0 = clk::now();
    int m = -1;
    std::vector<exl3::Tensor> trellis(n_part);
    for (int i = 0; i < n_part; ++i) {
        const std::string& b = w.parts[i];
        exl3::Tensor s1, s2, bias, mk;
        if (src->find(b + ".trellis", &trellis[i], err) != 1) { *err = b + ".trellis: " + *err; return false; }
        if (src->find(b + ".suh", &s1, err) != 1 || src->find(b + ".svh", &s2, err) != 1) { *err = b + ": suh/svh"; return false; }
        if (src->find(b + ".bias", &bias, err) != 0) { *err = b + " has a .bias"; return false; }
        m = (int)trellis[i].shape[2];
        uint32_t mv = 0;
        if (src->find(b + ".mul1", &mk, err) != 1 || !src->read(mk, &mv, err) || mv != EXL3_MUL1_MARKER) { *err = b + ": mul1"; return false; }
    }
    RadEncoding enc;
    if (exl3i_encoding_make(m, EXL3I_CB_MAD, n_part == 2, w.rows, k, &enc) != RAD_OK) { *err = "encoding"; return false; }
    ph->plan += ms_since(t0);
    const int tw = m / 2;
    const int64_t kt = k / 16, nt = n / 16, rw = exl3i_row_words(k, m);
    rec->assign((size_t)(n_part * nt * rw), 0u);
    // ---- read + reorder per part
    for (int i = 0; i < n_part; ++i) {
        raw->resize((size_t)trellis[i].bytes());
        t0 = clk::now();
        if (!src->read(trellis[i], raw->data(), err)) return false;
        ph->read += ms_since(t0);
        ph->bytes += (double)trellis[i].bytes();
        t0 = clk::now();
        reorder_part(reinterpret_cast<const uint32_t*>(raw->data()), kt, nt, tw, rw, rec->data() + (size_t)i * nt * rw);
        ph->reorder += ms_since(t0);
    }
    // ---- suh / svh (F16): suh part i at i k, svh part i at n_part k + i n
    t0 = clk::now();
    scales->assign((size_t)(n_part * (k + n)), 0);
    for (int i = 0; i < n_part; ++i) {
        exl3::Tensor s1, s2;
        if (src->find(w.parts[i] + ".suh", &s1, err) != 1 || src->find(w.parts[i] + ".svh", &s2, err) != 1) return false;
        if (!src->read(s1, scales->data() + (size_t)i * k, err) || !src->read(s2, scales->data() + n_part * k + (size_t)i * n, err)) return false;
        ph->bytes += (double)(s1.bytes() + s2.bytes());
    }
    ph->scales += ms_since(t0);
    ph->codes_bytes += (double)(rec->size() * 4);
    return true;
}

// trunk weights of the level, from the index: every layer's projections, stacked gate+up as one weight
static std::vector<Weight> trunk_weights(const std::string& dir) {
    std::string text, err;
    exl3::Json ix;
    if (!exl3::read_file(dir + "/model.safetensors.index.json", &text, &err) ||
        !exl3::parse_json(text.data(), text.data() + text.size(), &ix, &err)) { std::fprintf(stderr, "index: %s\n", err.c_str()); std::exit(2); }
    const exl3::Json* wm = ix.get("weight_map");
    std::map<std::string, bool> bases;
    for (const auto& kv : wm->obj) {
        const std::string& s = kv.first;
        const std::string suf = ".trellis";
        if (s.size() > suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0 &&
            s.find("model.language_model.layers.") == 0)
            bases[s.substr(0, s.size() - suf.size())] = true;
    }
    std::vector<Weight> out;
    for (const auto& kv : bases) {
        const std::string& b = kv.first;
        if (b.find(".mlp.gate_proj") != std::string::npos) {
            std::string up = b;
            up.replace(up.find("gate_proj"), 9, "up_proj");
            if (!bases.count(up)) continue;
            Weight w;
            w.parts = {b, up};
            w.name = b + "+" + up;
            out.push_back(w);
        } else if (b.find(".mlp.up_proj") != std::string::npos) {
            continue;    // handled with its gate
        } else {
            Weight w;
            w.parts = {b};
            w.name = b;
            out.push_back(w);
        }
    }
    return out;
}

static void fill_shapes(exl3::Source* src, Weight* w) {
    exl3::Tensor t;
    std::string err;
    src->find(w->parts[0] + ".trellis", &t, &err);
    // trellis [k/16][n/16][m] -> cols = k, rows(part) = n
    w->cols = t.shape[0] * 16;
    w->rows = t.shape[1] * 16 * (int64_t)w->parts.size();
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: prof_trunk DIR one BASE | all [THREADS]\n"); return 1; }
    const std::string dir = argv[1], mode = argv[2];
    std::string err;
    auto t_open = clk::now();
    exl3::Source* src = exl3::Source::open(dir, &err);
    if (!src) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 2; }
    const double open_ms = ms_since(t_open);
    std::printf("open       %.3f ms (config + index)\n", open_ms);
    if (mode == "one") {
        if (argc < 4) return 1;
        Weight w;
        std::string b = argv[3];
        w.parts = {b};
        if (b.find('+') != std::string::npos) { w.parts = {b.substr(0, b.find('+')), b.substr(b.find('+') + 1)}; }
        w.name = b;
        fill_shapes(src, &w);
        std::printf("weight     %s  parts=%zu  rows=%lld cols=%lld\n", w.name.c_str(), w.parts.size(), (long long)w.rows, (long long)w.cols);
        for (int pass = 0; pass < 3; ++pass) {
            // pass 0 cold (the weight's shard pages dropped first), passes 1-2 warm
            if (pass == 0) for (const auto& f : shard_files(dir)) drop_cache(f);
            Phase ph;
            std::vector<uint32_t> rec;
            std::vector<uint8_t> raw;
            std::vector<uint16_t> sc;
            if (!import_weight(src, w, &ph, &rec, &raw, &sc, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
            std::printf("%-5s plan %.3f ms  read %.3f ms (%.0f MB, %.2f GB/s)  reorder %.3f ms  scales %.3f ms  record %.1f MB\n",
                        pass == 0 ? "cold" : "warm", ph.plan, ph.read, ph.bytes / 1e6, ph.bytes / 1e9 / (ph.read / 1e3), ph.reorder,
                        ph.scales, ph.codes_bytes / 1e6);
        }
        return 0;
    }
    // ---- all
    const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
    std::vector<Weight> ws = trunk_weights(dir);
    for (auto& w : ws) fill_shapes(src, &w);
    std::printf("weights    %zu (trunk: stacked gate_up counted once)\n", ws.size());
    const std::vector<std::string> files = shard_files(dir);
    for (int nt_run : {1, threads}) {
        // every pass starts cold: the level's shards are dropped from the page cache first
        for (const auto& f : files) drop_cache(f);
        std::atomic<size_t> next{0};
        std::vector<Phase> per((size_t)nt_run);
        std::atomic<int> fails{0};
        auto t0 = clk::now();
        std::vector<std::thread> pool;
        for (int th = 0; th < nt_run; ++th)
            pool.emplace_back([&, th] {
                std::vector<uint32_t> rec;
                std::vector<uint8_t> raw;
                std::vector<uint16_t> sc;
                std::string e;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= ws.size()) break;
                    if (!import_weight(src, ws[i], &per[(size_t)th], &rec, &raw, &sc, &e)) { std::fprintf(stderr, "%s\n", e.c_str()); fails++; }
                }
            });
        for (auto& t : pool) t.join();
        const double wall = ms_since(t0);
        Phase tot;
        for (const Phase& p : per) { tot.plan += p.plan; tot.read += p.read; tot.reorder += p.reorder; tot.scales += p.scales; tot.bytes += p.bytes; tot.codes_bytes += p.codes_bytes; }
        std::printf("threads %2d  wall %.0f ms (cold)  summed over threads: plan %.0f  read %.0f  reorder %.0f  scales %.0f ms; read %.2f GB; record %.2f GB; fails %d\n",
                    nt_run, wall, tot.plan, tot.read, tot.reorder, tot.scales, tot.bytes / 1e9, tot.codes_bytes / 1e9, fails.load());
        if (nt_run == threads) break;
    }
    return 0;
}
