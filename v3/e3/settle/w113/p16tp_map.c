/* p16tp_map.c -- w113 #148 (../w113.md): the P16-TP output map a kernel applies at its store, as plain C (no table).
 * A serving rank's GEMM writes its stored output column j (local, 0 .. N_local - 1) to local natural column
 *     16 * p16tp_tile(a, j >> 4) + (j & 15).
 * Parameters (per tensor, per serving TP t; nest = 4, the TP4 rank sets the interleave lives in):
 *     np        stacked row parts (ssm_qkv 3: q | k | v; ssm_z, attn_qg, attn_k, attn_v 1)
 *     c[p]      part p's 16-row tiles per TP4 rank (27B: qkv 32, 32, 96; z 96; qg 192; k / v 16)
 *     C[p]      prefix sums of c (C[0] = 0, C[np] = 8 nb)
 *     nb        128-row blocks per TP4 rank (= C[np] / 8: qkv 20, z 12, qg 24, k / v 2)
 *     m         4 / t (TP4 sub-ranks inside a serving rank: TP1 4, TP2 2, TP4 1)
 * The divisions by c[p] and nb (not powers of two) are, on the GPU, multiply-shifts with per-launch magic numbers as
 * w111's il16 kernel does for nb (exact for the ranges here; the launch checks it). One part: p = q = 0 and the loops
 * vanish. Build: gcc -O2 -shared -fPIC -o libp16tp.so p16tp_map.c */
typedef struct { int np, m, nb; int c[3]; int C[4]; } p16tp_t;

int p16tp_tile(const p16tp_t* a, int k) {
    int p = 0;
    while (p + 1 < a->np && k >= a->m * a->C[p + 1]) ++p;
    const int o = k - a->m * a->C[p];
    const int s = o / a->c[p];                       /* TP4 sub-rank inside this rank */
    const int i = a->C[p] + (o - s * a->c[p]);       /* tile index inside the TP4 rank's stored order */
    const int j = 8 * (i % a->nb) + i / a->nb;       /* gperm(16 * 8 nb, 16): the TP4 rank's natural tile */
    int q = 0;
    while (q + 1 < a->np && j >= a->C[q + 1]) ++q;
    return a->m * a->C[q] + s * a->c[q] + (j - a->C[q]);
}

/* fills out[k] = p16tp_tile(k) for k < n (for the harness) */
void p16tp_all(const p16tp_t* a, int n, int* out) {
    for (int k = 0; k < n; ++k) out[k] = p16tp_tile(a, k);
}
