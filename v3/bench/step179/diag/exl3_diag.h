/* exl3_diag.h -- TIMING DIAGNOSTIC ONLY, never in a served build (w152, #179): QWEN35_EXL3_DIAG_SKIP="op,op,..." drops
 * those ops' issues (the outputs are garbage; GEMV time does not depend on the values), "nongemv" every op but the
 * dense / gemm / logits / embed / gather / attention ones. Included by qwen35_exl3.cpp after the radiance headers and
 * before the blocks, so RAD_OP / RAD_ISSUE at the plugin's own sites go through here. */
#ifndef EXL3_DIAG_H
#define EXL3_DIAG_H
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static std::vector<std::string>& diag_names() { static std::vector<std::string> v; return v; }
static inline rad_op diag_decl_op(RadBuilder* b, const char* name, const RadParam* p, int np, const rad_weight* w, int nw) {
    const rad_op h = rad_decl_op(b, name, p, np, w, nw);
    if (h) { if (diag_names().size() <= (size_t)h) diag_names().resize((size_t)h + 1); diag_names()[(size_t)h] = name; }
    return h;
}
static inline bool diag_skip(rad_op op) {
    static const char* s = std::getenv("QWEN35_EXL3_DIAG_SKIP");
    if (!s || !*s || (size_t)op >= diag_names().size()) return false;
    const std::string& n = diag_names()[(size_t)op];
    if (n.empty()) return false;
    if (std::strstr(s, "nongemv")) {
        static const char* keep[] = { "exl3i_dense", "exl3_dense", "gemm_nt", "logits_gemm", "embed_lookup", "embed_tq",
                                      "gather_rows", "attn_paged", "kv_store", "qk_norm_rope_gate_kv_store", "rope_table" };
        for (const char* k : keep) if (n == k) return false;
        return true;
    }
    const std::string L = std::string(",") + s + ",";
    return L.find("," + n + ",") != std::string::npos;
}
static inline int diag_issue(RadCtx* c, rad_op op, const RadOperand* o, int n_o, int64_t n) {
    return diag_skip(op) ? RAD_OK : rad_issue(c, op, o, n_o, n);
}
#define rad_decl_op diag_decl_op
#define rad_issue diag_issue
#endif
