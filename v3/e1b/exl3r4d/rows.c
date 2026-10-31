/* Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE. */
/* rows.c -- exl3r4d, the gfx1201 kernel library for the dense EXL3 linear (exl3_dense / gemm_exl3_nt): its identity,
 * its one row, the shape hook.
 *
 * Refused at selection, by constraint: dtype other than exl3k1 .. exl3k8, codebook other than mul1, K or N not a
 * multiple of 128; at declare, by the layout hook: an encoding the row does not decode or a plane of the wrong
 * extent; at launch (nt.c): the same again plus N % 256 with suh.1 and the alignment. */
#include "exl3r4d.h"

#include <string.h>

static void desc(RadOpdDesc* d, uint32_t dt, uint32_t rank, const int64_t* shape, uint32_t fill) {
    memset(d, 0, sizeof *d);
    d->dtype = dt;
    d->rank = rank;
    for (uint32_t r = 0; r < rank; ++r) d->shape[r] = shape[r];
    d->fill = fill;
    d->idx_const = -1;
}

/* v8 gemm_exl3_nt: the dense row's shape -- described as the reference row describes it (kernels/exl3ref/
 * rows.c's nt branch: plain tensors, not a table -- codes [N, K], suh / suh.1 [K], svh [N], no routing
 * operands), over every K the device decode instantiates (2..8: this row is not the decode band's R23
 * configuration). The engine's serving path hands the same planes to the layout hooks (nt.c), which take
 * them through relayout as the tool's; the cold-tool form is the description. */
static int nt_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    Exl3Geom g;
    if (exl3_geom_parse(p, n_p, EXL3_NT, 0, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (g.bits < 1 || g.bits > 8 || g.codebook != EXL3_CB_MUL1) return RAD_E_UNSUPPORTED;
    if (g.rows[0] % 128 || g.cols[0] % 128) return RAD_E_UNSUPPORTED;
    const Exl3Positions* at = exl3_positions(EXL3_NT);
    if (operand < 0 || operand >= at->n_opd || !out) return RAD_E_SHAPE;
    RadOpdDesc v[8];
    desc(&v[at->x], RAD_F16, 2, (int64_t[]){ g.M, g.x_cols }, RAD_FILL_NORMAL);
    for (int pl = 0; pl < EXL3_N_P; ++pl) {
        const int k = at->plane[0][pl];
        if (k < 0) continue;
        int64_t r, cl;
        exl3_plane_extents(g.rows[0], g.cols[0], pl, &r, &cl);
        if (pl == EXL3_P_CODES) {
            desc(&v[k], exl3_codes_dtype(g.bits), 2, (int64_t[]){ r, cl }, RAD_FILL_INDEX);
            v[k].idx_max = (int64_t)1 << g.bits;
        } else {
            desc(&v[k], RAD_F16, 1, (int64_t[]){ pl == EXL3_P_SVH ? r : cl }, RAD_FILL_NORMAL);
        }
    }
    /* y: f16, as exl3dense describes it (the row stores bf16, f16 or f32, whichever the engine hands it) */
    desc(&v[at->y], RAD_F16, 2, (int64_t[]){ g.M, g.y_cols }, RAD_FILL_NORMAL);
    for (int k = 0; k < at->n_opd; ++k)
        for (uint32_t r = 0; r < v[k].rank; ++r)
            if (v[k].shape[r] <= 0) return RAD_E_UNSUPPORTED;
    *out = v[operand];
    return RAD_OK;
}
static int nt_launch(const RadArgs* a, RadStream s) { return exl3r4d_nt_launch(a, s); }
static int64_t nt_scratch(const RadArgs* a) { return exl3r4d_nt_scratch(a); }
static int nt_layout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                     const RadTensor* pl, int n, RadLayout* out) {
    return exl3r4d_nt_layout(p, n_p, o, e, sel, pl, n, out);
}
static int nt_relayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                       const RadTensor* pl, int n, void* dst, int64_t b) {
    return exl3r4d_nt_relayout(p, n_p, o, e, sel, pl, n, dst, b);
}
static int nt_unrelayout(const RadParam* p, int n_p, int o, const RadEncoding* e, const int* sel,
                         const void* src, int64_t b, const RadTensor* pl, int n) {
    return exl3r4d_nt_unrelayout(p, n_p, o, e, sel, src, b, pl, n);
}

/* v8 gemm_exl3_nt (the dense linear): one weight, no routing -- its own dtype list (the decode band's R23
 * K 2..6 is the MoE ops' shipped configuration; the dense op instantiates every K the Plan's decode takes).
 * The first pass's band is the decode band's (M <= 16, split-K 1); N % 256 with the optional suh.1 is the
 * adapter's check (the seam at a 128-column block boundary), the constraints carry the op's own keys. */
#define EXL3R4D_NT_DTYPES "exl3k1 exl3k2 exl3k3 exl3k4 exl3k5 exl3k6 exl3k7 exl3k8"
/* Every M: a weight is stored once, in this row's relayout, so no band of the op can go to a row that reads it
 * as stored (exl3dense) -- the row covers the whole range, a 16-row chunk a grid.y workgroup. */
static const RadConstraint c_nt[] = {
    RAD_CGE("M", 1),
    RAD_CIN("dtype", EXL3R4D_NT_DTYPES),
    RAD_CIN("codebook", "mul1"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
};
#define N_NT (int)(sizeof c_nt / sizeof c_nt[0])

static const RadKernelInfo k_rows[] = {
    {
        .name = "exl3r4d_gemm_nt", .op = EXL3_OP_NT, .family = "gemm",
        .computes = "EXL3 dense linear (v8), one weight, no routing: a x suh (suh.1 for the weight's second half "
                    "when the optional suh.1 operand is present) and rotated, trellis decode in registers, f16 "
                    "WMMA, the output rotation, x svh, bf16 / f16 / f32 y (the GSCR form, the P1c rotation in "
                    "the workgroup; one 16-row tile a workgroup at M <= 16, EXL3R4D_NT_MT past it)",
        .shape = "any M (16 MT-row tiles over grid.y); K % 128, N % 128 (N % 256 with suh.1); split-K 1",
        .dtypes = "exl3k1 .. exl3k8, mul1; a f16 or bf16; y bf16, f16 or f32",
        .domain = RAD_DOMAIN_DEVICE, .priority = 100,
        .constraints = c_nt, .n_constraints = N_NT,
        .launch = nt_launch, .scratch = nt_scratch,
        .layout = nt_layout, .relayout = nt_relayout, .opd_shape = nt_shape, .unrelayout = nt_unrelayout,
    },
};

/* ------------------------------------------------------------------ the exports */

#ifndef EXL3R4D_BUILD_TARGET
#define EXL3R4D_BUILD_TARGET "gfx1201"
#endif

static const RadPluginInfo k_info = {
    RAD_PLUGIN_KERNEL, "exl3r4d", "0.1.0",
    "the dense EXL3 linear (gemm_exl3_nt) on gfx1201: trellis decode, f16 WMMA (decode band in registers, prefill two-pass)",
    EXL3R4D_BUILD_TARGET,
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &k_info; }

int                rad_kernel_schema_count(void) { return exl3_schema_count(); }
const RadOpSchema* rad_kernel_schema_at(int i) { return exl3_schema_at(i); }
int                rad_kernel_count(void) { return (int)(sizeof k_rows / sizeof k_rows[0]); }
const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < rad_kernel_count() ? &k_rows[i] : 0;
}

/* The row reads and writes its operands and the scratch slice it is handed and nothing else: no counter, no global,
 * nothing kept between launches; its dispatches (prep, decode, GEMM) are one op's, which the engine always orders. */
int rad_kernel_concurrent(int i) { return i >= 0 && i < rad_kernel_count(); }
