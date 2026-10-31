/* Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE. */
/* nt.c -- the gemm_exl3_nt (interface v8) host side, served as radiance's `exl3_dense` op: the row's operand
 * checks, argument block, the GSCR region's scratch, and the weight's layout hooks at K 1..8 (mul1). The device
 * side is exl3r4d.hip's v8 section (the GSCR form, split-K 1; MT 16-row tiles a workgroup past M = 16).
 *
 * EVERY M: a weight is stored once, in the layout its kernel asked for, so the op's bands cannot split between
 * this row (the trellis record) and one that reads the canonical codes (exl3dense). The row covers decode and
 * prefill alike.
 *
 * ONE WEIGHT, NO ROUTING: the block is filled as the down's over a single entry -- cls[0] the weight
 * (its record, one matrix [N, K]), cls[1] degenerate (the kernel reads class 0 only), n_expert / top_k
 * one, sorted / off null (the kernel reads neither), T the tokens, x / y the operands. The GSCR region is
 * one shared [T][2 * K] f16 (nt_pass's seam; without plane 3 both halves rotate identically and a workgroup
 * reads its own half). y is bf16, f16 or f32 (the logits). */
#include "exl3r4d.h"
#include "exl3r4d_dev.h"
#include "rad_types.h"
#include <string.h>

#ifndef EXL3R4D_NPW
#define EXL3R4D_NPW 2   /* column tiles a wave (launch.c's; the dense row's tuning arm is the same lever) */
#endif
/* the device side (exl3r4d.hip's v8 section) */
int exl3r4d_launch_nt(const Exl3r4dGemm* a, int bits, int npw, void* stream);

/* The flat width of a 2-D or 3-D operand whose rows are shape[0] (the trailing dims contiguous), or -1. */
static int64_t nt_flat_width(const RadTensor* t)
{
    if (t->rank < 2 || t->rank > 3) return -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 3 && t->stride[1] != t->shape[2]) return -1;
    const int64_t w = t->rank == 3 ? t->shape[1] * t->shape[2] : t->shape[1];
    return t->stride[0] >= w ? w : -1;
}

/* ------------------------------------------------------------------ the geometry + operand checks
 *
 * The same contract the reference row takes (kernels/exl3ref/ref.c's launch_nt), plus the device row's own
 * facts (a constraint cannot say them): K % 128 and N % 128 (the 128-point rotations), N % 256 when the
 * weight has plane 3 (the seam must be a 128-column block boundary; the reference's two-half split demands
 * the same), a f16 or bf16 `a`, a bf16, f16 or f32 `y`, mul1 (the device decode is the in-register Plan, no
 * LUT), and the alignment the loads and stores take (8 bytes a lane in, 16 bytes a lane out; the region's
 * rows 16-B aligned for the A fragment loads). */
static int nt_geom(const RadArgs* a, Exl3Geom* g, const RadTensor** x, const RadTensor** y,
                   const RadTensor** tc, const RadTensor** ts, const RadTensor** ts1, const RadTensor** tv)
{
    const Exl3Positions* at = exl3_positions(EXL3_NT);
    if (a->n_t < at->n_opd) return RAD_E_INVAL;
    *x = rad_arg_in(a, at->x);
    *y = rad_arg_in(a, at->y);
    *tc = at->plane[0][EXL3_P_CODES] >= 0 ? rad_arg_in(a, at->plane[0][EXL3_P_CODES]) : 0;
    *ts = at->plane[0][EXL3_P_SUH] >= 0 ? rad_arg_in(a, at->plane[0][EXL3_P_SUH]) : 0;
    *ts1 = at->plane[0][EXL3_P_SUH1] >= 0 ? rad_arg_in(a, at->plane[0][EXL3_P_SUH1]) : 0;
    *tv = at->plane[0][EXL3_P_SVH] >= 0 ? rad_arg_in(a, at->plane[0][EXL3_P_SVH]) : 0;
    if (!*x || !*y || !*tc || !*ts || !*tv) return RAD_E_INVAL;

    const int rc = exl3_geom_parse(a->p, a->n_p, EXL3_NT, 0, g);
    if (rc != RAD_OK) return rc;
    if (g->bits < 1 || g->bits > 8) return RAD_E_DTYPE;       /* the row's dtypes: exl3k1 .. exl3k8 */
    if (g->codebook != EXL3_CB_MUL1) return RAD_E_DTYPE;    /* the device decode is mul1 (no LUT) */
    if (g->rows[0] % 128 || g->cols[0] % 128) return RAD_E_SHAPE;
    if (*ts1 && g->rows[0] % 256) return RAD_E_SHAPE;       /* the seam at a block boundary */

    if ((*x)->dtype != RAD_BF16 && (*x)->dtype != RAD_F16) return RAD_E_DTYPE;
    if ((*y)->dtype != RAD_BF16 && (*y)->dtype != RAD_F32 && (*y)->dtype != RAD_F16) return RAD_E_DTYPE;
    /* token rows, flat: the engine keeps a buffer's own rank (a GDN output is [tok, n_head_v, head_v]), so
     * the rows are shape[0] and the width the trailing dims' product, contiguous (exl3dense's rule) */
    const int64_t xw = nt_flat_width(*x), yw = nt_flat_width(*y);
    if (xw < 0 || yw < 0) return RAD_E_STRIDE;
    const int64_t M = (*x)->shape[0];   /* the step's rows (exl3dense's Mrows) */
    if (M < 1) return RAD_E_SHAPE;
    if ((*y)->shape[0] < M) return RAD_E_SHAPE;
    if (yw != g->rows[0] || xw < g->cols[0]) return RAD_E_SHAPE;
    if (!rad_ptr_aligned((*x)->data, 8) || ((*x)->stride[0] * 2) % 8) return RAD_E_ALIGN;
    if (!rad_ptr_aligned((*y)->data, 16) || rad_dtype_bytes((*y)->dtype, (*y)->stride[0]) % 16) return RAD_E_ALIGN;
    return RAD_OK;
}

/* One weight plane as stored: codes in this library's record form (the layout hook's, below), the scales f16,
 * the extents the geometry implies. */
static int nt_plane(const RadTensor* t, int p, int bits, int64_t rows, int64_t cols)
{
    int64_t pr, pc;
    exl3_plane_extents(rows, cols, p, &pr, &pc);
    if (!t) return RAD_E_INVAL;
    (void)bits;
    if (p == EXL3_P_CODES) {   /* the stored form: the layout hook's record, [N, K] of the trellis dtype */
        if (t->dtype != EXL3R4D_DT_TRELLIS) return RAD_E_DTYPE;
        if (t->rank != 2 || t->shape[0] != rows || t->shape[1] != cols) return RAD_E_SHAPE;
        return RAD_OK;
    }
    if (t->dtype != RAD_F16) return RAD_E_DTYPE;
    /* a vector, stored [n, 1] or [1, n] (the layout hook checks the count alone, as here) */
    if (t->rank == 1) {
        if (t->shape[0] != pr * pc) return RAD_E_SHAPE;
    } else if (t->rank == 2) {
        if (t->shape[0] * t->shape[1] != pr * pc) return RAD_E_SHAPE;
        if (t->shape[1] > 1 && t->stride[1] != 1) return RAD_E_STRIDE;
    } else {
        return RAD_E_SHAPE;
    }
    if (p != EXL3_P_CODES && !rad_ptr_aligned(t->data, 2)) return RAD_E_ALIGN;
    return RAD_OK;
}

/* The argument block. The plane tables take the engine's stored form (the layout hooks' records, one entry
 * a weight) through launch.c's table fill, as the decode rows do -- `entries` one. */
static int nt_fill(const RadArgs* a, Exl3r4dGemm* d)
{
    const RadTensor *x, *y, *tc, *ts, *ts1, *tv;
    Exl3Geom g;
    int rc = nt_geom(a, &g, &x, &y, &tc, &ts, &ts1, &tv);
    if (rc != RAD_OK) return rc;
    if ((rc = nt_plane(tc, EXL3_P_CODES, g.bits, g.rows[0], g.cols[0])) != RAD_OK) return rc;
    if ((rc = nt_plane(ts, EXL3_P_SUH, g.bits, g.rows[0], g.cols[0])) != RAD_OK) return rc;
    if ((rc = nt_plane(tv, EXL3_P_SVH, g.bits, g.rows[0], g.cols[0])) != RAD_OK) return rc;
    if (ts1 && (rc = nt_plane(ts1, EXL3_P_SUH1, g.bits, g.rows[0], g.cols[0])) != RAD_OK) return rc;

    memset(d, 0, sizeof *d);
    const void* base[5];
    int64_t s0[5];
    const RadTensor* pl[4] = { tc, ts, ts1, tv };
    for (int q = 0; q < 4; ++q) {
        if (q == 2 && !pl[q]) {   /* plane 3 absent: the entry is null, the kernel's fallback's */
            base[2] = 0;
            s0[2] = 0;
            continue;
        }
        /* one weight, no table: the stored plane itself (table_fill reads the MoE form, entries along dim 0) */
        if (!rad_ptr_aligned(pl[q]->data, q == 0 ? 256 : 2)) return RAD_E_ALIGN;
        base[q] = pl[q]->data;
        s0[q] = 1;   /* a record stride, not 0 (entry_at's pointer-table form): entry 0 is the plane itself */
    }
    d->cls[0].codes = base[0];
    d->cls[0].suh = base[1];
    d->cls[0].suh1 = base[2];
    d->cls[0].svh = base[3];
    d->cls[0].codes_s0 = s0[0];
    d->cls[0].suh_s0 = s0[1];
    d->cls[0].suh1_s0 = s0[2];
    d->cls[0].svh_s0 = s0[3];
    d->cls[0].rows = (int32_t)g.rows[0];
    d->cls[0].cols = (int32_t)g.cols[0];
    d->cls[1] = d->cls[0];   /* degenerate for the kernel (it reads class 0 only); the block's shape */

    d->x = x->data;
    d->y = y->data;
    d->sorted = 0;
    d->off = 0;
    d->xr = a->scratch;     /* the GSCR region, [T][2 * K] f16 (the row's scratch) */
    {   /* T2: the decoded weight chunk past the region, when the scratch holds it (exl3r4d_nt_scratch) */
        const int64_t xr_bytes = ((int64_t)x->shape[0] * 2 * g.cols[0] * 2 + 255) / 256 * 256;
        d->raw = x->shape[0] >= EXL3R4D_T2_MIN && a->scratch_bytes >= xr_bytes + EXL3R4D_T2_CHUNK_BYTES
                     ? (float*)((char*)a->scratch + xr_bytes) : 0;
    }
    d->xr_pitch = 2 * g.cols[0];
    d->x_pitch = x->stride[0];
    d->y_pitch = y->stride[0];
    d->T = (int32_t)x->shape[0];
    d->n_tok = d->T;
    d->n_expert = 1;
    d->top_k = 1;
    d->n_class = 2;
    d->y_cols = (int32_t)g.rows[0];
    d->x_bf16 = x->dtype == RAD_BF16;
    d->y_dt = y->dtype == RAD_F32 ? 1 : y->dtype == RAD_F16 ? 2 : 0;
    d->sk = 1;              /* the first pass's (the down's fit logic is the tuning arm) */
    d->chunk = 16;
    d->nslot = 1;
    return RAD_OK;
}

int exl3r4d_nt_launch(const RadArgs* a, RadStream s)
{
    Exl3r4dGemm d;
    int rc = nt_fill(a, &d);
    if (rc != RAD_OK) return rc;
    if (!d.xr) return RAD_E_INVAL;   /* the region (the row's scratch hook sizes it) */
    Exl3Geom g;
    const RadTensor *x, *y, *tc, *ts, *ts1, *tv;
    rc = nt_geom(a, &g, &x, &y, &tc, &ts, &ts1, &tv);
    return rc != RAD_OK ? rc : exl3r4d_launch_nt(&d, g.bits, EXL3R4D_NPW, s);
}

/* The GSCR region: one shared [T][2 * K] f16, 256-B aligned (the A fragment loads take 16 bytes a lane off
 * the row's base, mat_a * K keeps the halves apart, K % 128 keeps the 128-block boundaries 256-B aligned). */
int64_t exl3r4d_nt_scratch(const RadArgs* a)
{
    /* Called at declare with no tensors (core/build/rad_builder.cpp: a scratch hook depends only on the
     * geometry): sized off the band's M and K alone. */
    Exl3Geom g;
    if (exl3_geom_parse(a->p, a->n_p, EXL3_NT, 0, &g) != RAD_OK) return -1;
    return (g.M * 2 * g.cols[0] * 2 + 255) / 256 * 256 + (g.M >= EXL3R4D_T2_MIN ? EXL3R4D_T2_CHUNK_BYTES : 0);
}

/* ------------------------------------------------------------------ the weight's layout hooks (K 1..8, mul1)
 *
 * The decode band's hooks (layout.c) refuse what the MoE ops' shipped configuration does not decode (K 2..6,
 * the stacked form on the wrong op); the dense op takes every K the Plan's decode instantiates (2..8) and a
 * weight is stacked only when its encoding says so (plane 3) -- the seam, not the op, carries the halves.
 * The conversions are the same (k4's byte shuffle, kx's symbol-at-a-time form), one record over [N, K]
 * (the down's mats 1 form), the tags k2 .. k8. */
static const char* const nt_tag[9] = { 0, "exl3r4d.trellis.k1", "exl3r4d.trellis.k2", "exl3r4d.trellis.k3",
                                      "exl3r4d.trellis.k4", "exl3r4d.trellis.k5",
                                      "exl3r4d.trellis.k6", "exl3r4d.trellis.k7",
                                      "exl3r4d.trellis.k8" };

static int nt_check(int operand, const RadParam* p, int n_p, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes, int* q, int64_t* rows, int64_t* cols, int* bits)
{
    const Exl3Positions* pos = exl3_positions(EXL3_NT);
    *q = -1;
    for (int c = 0; c < pos->n_cls; ++c)
        for (int t = 0; t < EXL3_N_P; ++t)
            if (pos->plane[c][t] == operand) {
                *q = t;
                break;
            }
    if (*q < 0) return RAD_E_UNSUPPORTED;
    int b, cb, stacked;
    (void)stacked;
    if (exl3_encoding_read(enc, &b, &cb, &stacked) != RAD_OK) return RAD_E_DTYPE;
    if (b < 1 || b > 8) return RAD_E_DTYPE;
    if (cb != EXL3_CB_MUL1) return RAD_E_DTYPE;
    Exl3Geom g;
    if (exl3_geom_parse(p, n_p, EXL3_NT, 0, &g) != RAD_OK) return RAD_E_SHAPE;
    if (n_planes != 1 || !sel || !planes || sel[0] < 0 || sel[0] >= (int)enc->n_planes) return RAD_E_SHAPE;
    static const char* const name[EXL3_N_P] = { "codes", "suh", "suh.1", "svh", 0 };
    if (!name[*q] || strcmp(enc->plane[sel[0]].role, name[*q]) != 0) return RAD_E_SHAPE;
    int64_t er, ec;
    exl3_plane_extents(g.rows[0], g.cols[0], *q, &er, &ec);
    if (planes[0].rank != 2) return RAD_E_SHAPE;
    if (*q == EXL3_P_CODES ? (planes[0].shape[0] != er || planes[0].shape[1] != ec)
                           : planes[0].shape[0] * planes[0].shape[1] != er * ec)
        return RAD_E_SHAPE;
    if (*q != EXL3_P_CODES && planes[0].dtype != RAD_F16) return RAD_E_DTYPE;
    if (*q == EXL3_P_CODES && planes[0].dtype != exl3_codes_dtype(b)) return RAD_E_DTYPE;
    if (g.bits != b) return RAD_E_DTYPE;
    *rows = g.rows[0];
    *cols = g.cols[0];
    *bits = b;
    return RAD_OK;
}

int exl3r4d_nt_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                      const RadTensor* planes, int n_planes, RadLayout* out)
{
    int q, bits;
    int64_t rows, cols;
    const int rc = nt_check(operand, p, n_p, enc, sel, planes, n_planes, &q, &rows, &cols, &bits);
    if (rc != RAD_OK) return rc;
    if (!out) return RAD_E_INVAL;
    memset(out, 0, sizeof *out);
    if (q != EXL3_P_CODES) {   /* the scales: the same bytes, 256-B aligned (the decode rows' L-ALIGN) */
        out->tag = "exl3r4d.f16.vec";
        out->dtype = RAD_F16;
        out->rank = 2;
        out->shape[0] = planes[0].shape[0];
        out->shape[1] = planes[0].shape[1];
        out->align = 256;
        out->bytes = planes[0].shape[0] * planes[0].shape[1] * 2;
        return RAD_OK;
    }
    out->tag = nt_tag[bits];
    out->dtype = EXL3R4D_DT_TRELLIS;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = cols;
    out->align = 256;   /* every tile (32 K bytes) starts on 16 B */
    out->bytes = rows * cols * bits / 8;
    return RAD_OK;
}

int exl3r4d_nt_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                        const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes)
{
    int q, bits;
    int64_t rows, cols;
    const int rc = nt_check(operand, p, n_p, enc, sel, planes, n_planes, &q, &rows, &cols, &bits);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) {
        if (!dst || !planes[0].data || dst_bytes < planes[0].shape[0] * planes[0].shape[1] * 2)
            return RAD_E_INVAL;
        memcpy(dst, planes[0].data, (size_t)planes[0].shape[0] * planes[0].shape[1] * 2);
        return RAD_OK;
    }
    if (!dst || !planes[0].data || dst_bytes < rows * cols * bits / 8) return RAD_E_INVAL;
    const int64_t rb = exl3_codes_row_bytes(cols, bits);
    if (bits == 4) exl3r4d_k4_to_record(planes[0].data, rb, rows, cols, (uint32_t*)dst);
    else exl3r4d_kx_to_record(planes[0].data, rb, rows, cols, bits, (uint32_t*)dst);
    return RAD_OK;
}

int exl3r4d_nt_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                          const void* src, int64_t src_bytes, const RadTensor* planes, int n_planes)
{
    int q, bits;
    int64_t rows, cols;
    const int rc = nt_check(operand, p, n_p, enc, sel, planes, n_planes, &q, &rows, &cols, &bits);
    if (rc != RAD_OK) return rc;
    if (q != EXL3_P_CODES) {
        if (!src || !planes[0].data || src_bytes < planes[0].shape[0] * planes[0].shape[1] * 2)
            return RAD_E_INVAL;
        memcpy(planes[0].data, src, (size_t)planes[0].shape[0] * planes[0].shape[1] * 2);
        return RAD_OK;
    }
    if (!src || !planes[0].data || src_bytes < rows * cols * bits / 8) return RAD_E_INVAL;
    const int64_t rb = exl3_codes_row_bytes(cols, bits);
    if (bits == 4) exl3r4d_k4_from_record((const uint32_t*)src, rows, cols, planes[0].data, rb);
    else exl3r4d_kx_from_record((const uint32_t*)src, rows, cols, bits, planes[0].data, rb);
    return RAD_OK;
}
