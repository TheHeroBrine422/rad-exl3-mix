/* exl3i_plan.h -- the per-lane bit plan of EXL3-V3's decode (fractional K = KA + n/16, sub-1 salted 0-bit reads),
 * generalizing exl3r4d's Plan<BITS> (exl3r4d_common.hip.h). Plain C, used by the HIP kernels (__device__) and by the
 * host test (test_plan.c), which emulates the lane's extraction and checks every lane / run / rate against the oracle.
 *
 * The record (stored plane = the kernel's operand) is exllamav3's tile stream: tile (tk, tn) at (tk nt + tn) of
 * 8 K u32 words (K = KA + popcount(MASK)/16: bpb = 16 KA + popcount bits per 16 positions, WORDS = bpb / 2), one
 * MSB-first ring of 32 WORDS bits. Position p's 16-bit window ends at ring bit S(p) = D(0) + .. + D(p).
 *
 * A lane (n = lane & 15, h = lane >> 4) owns two runs of four positions, q = base + 8 r + i (i 0..3, r 0..1),
 * base = 32 (n & 7) + 4 (n >> 3) + 16 h (exl3r4d's fragment order). Per run: e3 = S(q3) (window end of the last
 * position; e3 in [1, R], since S can be 0 at sub-1), the 32 ring bits ending at e3 come from words (de - 1, de),
 * de = (e3 - 1) / 32, by alignbit(w[de - 1], w[de], shift), shift = 32 (de + 1) - e3; state i = ubfe(lo, off[i], 16),
 * off[i] = e3 - S(q_i). Needs D(q1) + D(q2) + D(q3) <= 16, i.e. KA + 1 <= 5: KA <= 4 (the V3 menu's fractional
 * rates; integer K keeps exl3r4d's plan). xr[i] = salt(q_i) when D(q_i) = 0 (sub-1), else 0: value = cb(state ^ xr). */
#ifndef EXL3I_PLAN_H
#define EXL3I_PLAN_H
#include <stdint.h>

#ifdef __HIPCC__
#define EXL3I_HD __host__ __device__ __forceinline__
#else
#define EXL3I_HD static inline
#endif

typedef struct {
    int woff[2][2];      /* word indices (de - 1, de) per run */
    unsigned shift[2];   /* alignbit shift per run */
    unsigned off[2][4];  /* state i's offset below e3 */
    unsigned xr[2][4];   /* salt XOR (0 unless a 0-bit position) */
} Exl3iPlan;

EXL3I_HD int exl3i_d(int q, int ka, unsigned mask) { return ka + (int)((mask >> (q & 15)) & 1u); }

EXL3I_HD int exl3i_bpb(int ka, unsigned mask) {
    int b = 16 * ka; for (int i = 0; i < 16; i++) b += (int)((mask >> i) & 1u); return b;
}

EXL3I_HD int exl3i_S(int q, int ka, unsigned mask) {   /* window end of position q (q >= 0) */
    const int bpb = exl3i_bpb(ka, mask);
    int s = (q >> 4) * bpb;
    for (int j = 0; j <= (q & 15); j++) s += exl3i_d(j, ka, mask);
    return s;
}

EXL3I_HD unsigned exl3i_salt16(int q) { return (0x9E37u * (unsigned)((q & 15) + 1)) & 0xFFFFu; }

EXL3I_HD void exl3i_plan(Exl3iPlan* P, int lane, int ka, unsigned mask) {
    const int n = lane & 15, h = lane >> 4;
    const int base = 32 * (n & 7) + 4 * (n >> 3) + 16 * h;
    const int WORDS = exl3i_bpb(ka, mask) / 2, R = 32 * WORDS;
    for (int r = 0; r < 2; r++) {
        const int q0 = base + 8 * r;
        int e3 = exl3i_S(q0 + 3, ka, mask) % R;
        if (e3 == 0) e3 = R;
        const int de = (e3 - 1) >> 5;
        P->shift[r] = (unsigned)(32 * (de + 1) - e3);
        P->woff[r][0] = (de - 1 + WORDS) % WORDS;
        P->woff[r][1] = de;
        const int s3 = exl3i_S(q0 + 3, ka, mask);
        for (int i = 0; i < 4; i++) {
            P->off[r][i] = (unsigned)(s3 - exl3i_S(q0 + i, ka, mask));
            P->xr[r][i] = exl3i_d(q0 + i, ka, mask) == 0 ? exl3i_salt16(q0 + i) : 0u;
        }
    }
}
#endif
