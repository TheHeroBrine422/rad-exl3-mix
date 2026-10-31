/* hproxy -- the Hessian proxy table: for every exl3 weight of every container, the expected squared output error
 * on the challenge's calibration set,
 *
 *     err = tr(dW H dW^T) / rows,   dW = W_q - W,   H = sum over the calib set of x x^T (the weight's input)
 *
 * W_q is decoded with the plugin's own host decoder (exl3_decode_rows: what exl3r4d serves, bit for bit), W is
 * the bf16 checkpoint. H comes from the challenge dataset's hessians/calib (packed upper triangle, f32). The GEMM
 * runs on the card (hipBLAS sgemm), the decode on the host (OpenMP).
 *
 *   hproxy CKPT_DIR HESS_DIR OUT.jsonl TAG=CONTAINER.rad [TAG=CONTAINER.rad ...]
 *   env HP_FILTER=<substring>  only weights whose name contains it (testing)
 *
 * One JSON line per (weight, container): {"w","c","bits","bytes","err","ref","dw2","w2"}; ref = tr(W H W^T)/rows,
 * dw2 = ||dW||^2, w2 = ||W||^2 (dw2/w2 small = the mapping is right). Weights are visited grouped by Hessian, each
 * Hessian and bf16 weight loaded once; every container's version of the weight follows. */
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <omp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "rad_format.h"
#include "exl3_encoding.h"
extern "C" {
#include "exl3_decode.h"
}

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { hipblasStatus_t e_ = (x); if (e_ != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d hipblas %d\n", __FILE__, __LINE__, (int)e_); exit(1); } } while (0)

static const uint8_t* map_file(const std::string& p, size_t* n) {
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) { perror(p.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    if (n) *n = st.st_size;
    return (const uint8_t*)m;
}

/* ------------------------------------------------------------------ the .rad container */
struct Ent { int bits, cb, stacked; int64_t rows, cols; const uint8_t* codes; int64_t crb;
             const uint16_t *suh, *suh1, *svh; uint64_t bytes; };
struct Container {
    std::string tag; const uint8_t* base;
    std::map<std::string, Ent> w;
    void open(const std::string& t, const std::string& path) {
        tag = t; base = map_file(path, nullptr);
        auto* h = (const RadFileHeader*)base;
        if (h->magic != RAD_MAGIC || h->version != RAD_FORMAT_VER) { fprintf(stderr, "%s: not RAD2\n", path.c_str()); exit(1); }
        auto* str = (const char*)base + h->str_off;
        auto* dir = (const RadFileEntry*)(base + h->dir_off);
        auto* pl = (const RadFilePlane*)(base + h->plane_off);
        auto* enc = (const RadEncoding*)(base + h->enc_off);
        for (uint64_t i = 0; i < h->dir_count; ++i) {
            const RadFileEntry& e = dir[i];
            Ent x{};
            if (exl3_encoding_read(&enc[e.enc], &x.bits, &x.cb, &x.stacked) != RAD_OK) continue;
            x.rows = e.shape[0]; x.cols = e.shape[1]; x.bytes = e.bytes;
            const RadEncoding& en = enc[e.enc];
            for (uint32_t p = 0; p < e.n_planes; ++p) {
                const uint8_t* d = base + pl[e.plane_first + p].offset;
                const char* r = en.plane[p].role;
                if (!strcmp(r, "codes")) { x.codes = d; x.crb = (int64_t)(pl[e.plane_first + p].bytes / x.rows); }
                else if (!strcmp(r, "suh")) x.suh = (const uint16_t*)d;
                else if (!strcmp(r, "suh.1")) x.suh1 = (const uint16_t*)d;
                else if (!strcmp(r, "svh")) x.svh = (const uint16_t*)d;
            }
            w[str + e.name] = x;
        }
    }
};

/* ------------------------------------------------------------------ the bf16 checkpoint (safetensors) */
struct STensor { const uint16_t* p; int64_t rows, cols; };
static std::map<std::string, STensor> g_ck;
static void ck_open(const std::string& dir) {
    for (int s = 1; s < 100; ++s) {
        char fn[512]; bool found = false;
        for (int n = 1; n < 100 && !found; ++n) {
            snprintf(fn, sizeof fn, "%s/model-%05d-of-%05d.safetensors", dir.c_str(), s, n);
            found = access(fn, R_OK) == 0;
        }
        if (!found) break;
        const uint8_t* m = map_file(fn, nullptr);
        uint64_t hn; memcpy(&hn, m, 8);
        std::string hdr((const char*)m + 8, hn);
        const uint8_t* data = m + 8 + hn;
        /* {"name":{"dtype":"BF16","shape":[r,c],"data_offsets":[a,b]},...} -- flat, so a scan does */
        size_t pos = 0;
        while ((pos = hdr.find("\"dtype\"", pos)) != std::string::npos) {
            size_t q1 = hdr.rfind("\":{", pos), q0 = hdr.rfind('"', q1 - 1);
            std::string name = hdr.substr(q0 + 1, q1 - q0 - 1);
            size_t sh = hdr.find("\"shape\":[", pos) + 9, of = hdr.find("\"data_offsets\":[", pos) + 16;
            int64_t r = 0, c = 1; long long a = 0;
            sscanf(hdr.c_str() + sh, "%lld", (long long*)&r);
            size_t comma = hdr.find(',', sh), close = hdr.find(']', sh);
            if (comma < close) sscanf(hdr.c_str() + comma + 1, "%lld", (long long*)&c);
            sscanf(hdr.c_str() + of, "%lld", &a);
            if (hdr.compare(pos, 14, "\"dtype\":\"BF16\"") == 0)
                g_ck[name] = { (const uint16_t*)(data + a), r, c };
            pos = close;
        }
    }
    fprintf(stderr, "checkpoint: %zu bf16 tensors\n", g_ck.size());
}
static inline float bf(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; memcpy(&f, &x, 4); return f; }

/* container name -> checkpoint source(s) and Hessian file */
static bool route(const std::string& n, std::vector<std::string>* src, std::string* hess) {
    const std::string L = "model.language_model.layers.";
    if (n == "output.weight") { *src = {"lm_head.weight"}; *hess = "lm_head.input"; return true; }
    int l; char t[64];
    if (sscanf(n.c_str(), "blk.%d.%63[^.]", &l, t) != 2) return false;
    std::string p = L + std::to_string(l) + ".", s = t;
    if (s == "attn_qg")      { *src = {p + "self_attn.q_proj.weight"}; *hess = p + "input"; }
    else if (s == "attn_k")  { *src = {p + "self_attn.k_proj.weight"}; *hess = p + "input"; }
    else if (s == "attn_v")  { *src = {p + "self_attn.v_proj.weight"}; *hess = p + "input"; }
    else if (s == "attn_output") { *src = {p + "self_attn.o_proj.weight"}; *hess = p + "self_attn.o_proj.input"; }
    else if (s == "ssm_qkv") { *src = {p + "linear_attn.in_proj_qkv.weight"}; *hess = p + "input"; }
    else if (s == "ssm_z")   { *src = {p + "linear_attn.in_proj_z.weight"}; *hess = p + "input"; }
    else if (s == "ssm_out") { *src = {p + "linear_attn.out_proj.weight"}; *hess = p + "linear_attn.out_proj.input"; }
    else if (s == "ffn_gate_up") { *src = {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}; *hess = p + "post_attention.input"; }
    else if (s == "ffn_down") { *src = {p + "mlp.down_proj.weight"}; *hess = p + "mlp.down_proj.input"; }
    else return false;
    return true;
}

/* ------------------------------------------------------------------ the Hessian */
static float* load_hessian(const std::string& path, int64_t* K, double* rows) {
    size_t n; const uint8_t* m = map_file(path, &n);
    uint64_t hn; memcpy(&hn, m, 8);
    std::string hdr((const char*)m + 8, hn);
    long long k = 0, r = 0;
    sscanf(hdr.c_str() + hdr.find("\"k\":\"") + 5, "%lld", &k);
    sscanf(hdr.c_str() + hdr.find("\"rows\":\"") + 8, "%lld", &r);
    size_t of = hdr.find("\"data_offsets\":[") + 16; long long a = 0; sscanf(hdr.c_str() + of, "%lld", &a);
    const float* up = (const float*)(m + 8 + hn + a);
    float* H = (float*)malloc((size_t)k * k * 4);
    const float inv = 1.0f / (float)r;
    #pragma omp parallel for schedule(dynamic, 16)
    for (long long i = 0; i < k; ++i) {
        const float* row = up + (size_t)i * k - (size_t)i * (i - 1) / 2;   /* row i holds (i, i..k-1) */
        for (long long j = i; j < k; ++j) { float v = row[j - i] * inv; H[(size_t)i * k + j] = v; H[(size_t)j * k + i] = v; }
    }
    munmap((void*)m, n);
    *K = k; *rows = (double)r;
    return H;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "hproxy CKPT_DIR HESS_DIR OUT.jsonl TAG=CONTAINER...\n"); return 2; }
    const std::string ckdir = argv[1], hdir = argv[2];
    FILE* out = fopen(argv[3], "a");
    const char* filt = getenv("HP_FILTER");
    std::vector<Container> C(argc - 4);
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i]; size_t eq = a.find('=');
        C[i - 4].open(a.substr(0, eq), a.substr(eq + 1));
        fprintf(stderr, "%s: %zu exl3 weights\n", C[i - 4].tag.c_str(), C[i - 4].w.size());
    }
    ck_open(ckdir);

    /* group by Hessian */
    std::map<std::string, std::vector<std::string>> groups;
    for (auto& kv : C[0].w) {
        std::vector<std::string> src; std::string h;
        if (filt && kv.first.find(filt) == std::string::npos) continue;
        if (!route(kv.first, &src, &h)) { fprintf(stderr, "skip %s (no route)\n", kv.first.c_str()); continue; }
        groups[h].push_back(kv.first);
    }

    hipblasHandle_t hb; CB(hipblasCreate(&hb));
    const int64_t RB = 2048;                       /* rows a block */
    float *dH = nullptr, *dA = nullptr, *dC = nullptr; int64_t cap = 0;
    std::vector<float> wq, w;                      /* [RB][K] host: decoded, reference */

    for (auto& g : groups) {
        int64_t K; double hrows;
        double t0 = omp_get_wtime();
        float* H = load_hessian(hdir + "/" + g.first + ".safetensors", &K, &hrows);
        if (K > cap) {
            if (dH) { CK(hipFree(dH)); CK(hipFree(dA)); CK(hipFree(dC)); }
            CK(hipMalloc(&dH, (size_t)K * K * 4)); CK(hipMalloc(&dA, (size_t)K * RB * 4)); CK(hipMalloc(&dC, (size_t)K * RB * 4));
            cap = K;
        }
        CK(hipMemcpy(dH, H, (size_t)K * K * 4, hipMemcpyHostToDevice));
        free(H);
        wq.resize((size_t)RB * K); w.resize((size_t)RB * K);
        fprintf(stderr, "%s: K=%lld, %.1fs\n", g.first.c_str(), (long long)K, omp_get_wtime() - t0);

        /* sum over rows of (dW H) . dW, as (H dW^T) . dW^T column-major: A = dW^T [K x rb] */
        auto quad = [&](const float* rowsK, int64_t rb) -> double {
            CK(hipMemcpy(dA, rowsK, (size_t)rb * K * 4, hipMemcpyHostToDevice));
            const float one = 1.f, zero = 0.f; float dot = 0.f;
            CB(hipblasSgemm(hb, HIPBLAS_OP_N, HIPBLAS_OP_N, (int)K, (int)rb, (int)K, &one, dH, (int)K, dA, (int)K, &zero, dC, (int)K));
            CB(hipblasSdot(hb, (int)(K * rb), dC, 1, dA, 1, &dot));
            return (double)dot;
        };

        for (const std::string& name : g.second) {
            std::vector<std::string> src; std::string hs; route(name, &src, &hs);
            std::vector<STensor> st;
            for (auto& s : src) { auto it = g_ck.find(s); if (it == g_ck.end()) { fprintf(stderr, "no %s\n", s.c_str()); exit(1); } st.push_back(it->second); }
            const Ent& e0 = C[0].w.at(name);
            const int64_t N = e0.rows;
            if (e0.cols != K) { fprintf(stderr, "%s: cols %lld vs Hessian %lld\n", name.c_str(), (long long)e0.cols, (long long)K); exit(1); }
            auto ref_rows = [&](int64_t r0, int64_t rb, float* dst) {   /* the checkpoint's rows, stacked parts in order */
                #pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < rb; ++r) {
                    int64_t gr = r0 + r; size_t part = 0;
                    while (gr >= st[part].rows) { gr -= st[part].rows; ++part; }
                    const uint16_t* s = st[part].p + (size_t)gr * K;
                    float* d = dst + (size_t)r * K;
                    for (int64_t k = 0; k < K; ++k) d[k] = bf(s[k]);
                }
            };
            /* reference quadratic form, once a weight */
            double ref = 0, w2 = 0;
            for (int64_t r0 = 0; r0 < N; r0 += RB) {
                int64_t rb = std::min(RB, N - r0);
                ref_rows(r0, rb, w.data());
                ref += quad(w.data(), rb);
                for (size_t i = 0; i < (size_t)rb * K; ++i) w2 += (double)w[i] * w[i];
            }
            for (auto& c : C) {
                auto it = c.w.find(name);
                if (it == c.w.end()) continue;
                const Ent& e = it->second;
                Exl3View v{}; v.rows = e.rows; v.cols = e.cols; v.bits = e.bits; v.codebook = e.cb;
                v.codes = e.codes; v.codes_row_bytes = e.crb; v.suh = e.suh; v.suh1 = e.suh1; v.svh = e.svh;
                double err = 0, dw2 = 0; double t1 = omp_get_wtime(); int bad = 0;
                for (int64_t r0 = 0; r0 < N; r0 += RB) {
                    int64_t rb = std::min(RB, N - r0);
                    #pragma omp parallel for schedule(dynamic, 1) reduction(+:bad)
                    for (int64_t b = 0; b < rb; b += 128)
                        bad += exl3_decode_rows(&v, r0 + b, 128, wq.data() + (size_t)b * K) != EXL3_OK;
                    ref_rows(r0, rb, w.data());
                    double s2 = 0;
                    #pragma omp parallel for reduction(+:s2)
                    for (size_t i = 0; i < (size_t)rb * K; ++i) { float d = wq[i] - w[i]; wq[i] = d; s2 += (double)d * d; }
                    dw2 += s2;
                    err += quad(wq.data(), rb);
                }
                if (bad) { fprintf(stderr, "%s/%s: decode failed\n", c.tag.c_str(), name.c_str()); exit(1); }
                fprintf(out, "{\"w\": \"%s\", \"c\": \"%s\", \"bits\": %d, \"bytes\": %llu, \"err\": %.9g, \"ref\": %.9g, \"dw2\": %.9g, \"w2\": %.9g}\n",
                        name.c_str(), c.tag.c_str(), e.bits, (unsigned long long)e.bytes, err, ref, dw2, w2);
                fflush(out);
                fprintf(stderr, "  %-24s %-14s K%d err/ref %.3e dw2/w2 %.3e %.1fs\n", name.c_str(), c.tag.c_str(), e.bits,
                        err / ref, dw2 / w2, omp_get_wtime() - t1);
            }
        }
    }
    fclose(out);
    return 0;
}
