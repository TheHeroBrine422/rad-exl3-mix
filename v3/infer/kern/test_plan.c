/* test_plan.c -- exl3i_plan.h against the oracle (v3/infer/ref/exl3i_ref.c), host only: for every rate of the V3 menu
 * with KA <= 4 (and integer K 1..5 as a control), random tile words; every lane's two runs extract their four states the
 * way the kernel does (alignbit + ubfe on the planned words / shift / offsets, salt XOR), and states and values must
 * equal the oracle's for those ring positions. Build: gcc -O2 -ffp-contract=off test_plan.c ../ref/exl3i_ref.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include "exl3i_plan.h"

int exl3i_frac_mask(int n);
uint16_t exl3i_codebook(uint32_t state, int i8);
int exl3i_decode_tile(const uint32_t* words, int ka, int mask, int i8, uint16_t* states, uint16_t* values);

static unsigned alignbit(unsigned a, unsigned b, unsigned s) { return (unsigned)((((uint64_t)a << 32) | b) >> s); }
static unsigned ubfe(unsigned x, unsigned off, unsigned w) { return (x >> off) & ((1u << w) - 1u); }

int main(void) {
    int fails = 0, cases = 0;
    srand(7);
    for (int ka = 0; ka <= 5; ka++)
        for (int n = 0; n < 16; n += 2) {
            if (ka == 0 && n == 0) continue;
            if (ka == 5 && n) continue;                     /* integer K5 only: control */
            const unsigned mask = (unsigned)exl3i_frac_mask(n);
            const int WORDS = exl3i_bpb(ka, mask) / 2;
            int bad = 0;
            for (int t = 0; t < 40; t++) {
                uint32_t w[64]; uint16_t st[256], va[256];
                for (int i = 0; i < WORDS; i++) w[i] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
                exl3i_decode_tile(w, ka, (int)mask, 1, st, va);
                for (int lane = 0; lane < 32; lane++) {
                    Exl3iPlan P; exl3i_plan(&P, lane, ka, mask);
                    const int nn = lane & 15, h = lane >> 4, base = 32 * (nn & 7) + 4 * (nn >> 3) + 16 * h;
                    for (int r = 0; r < 2; r++) {
                        const unsigned lo = alignbit(w[P.woff[r][0]], w[P.woff[r][1]], P.shift[r]);
                        for (int i = 0; i < 4; i++) {
                            const int q = base + 8 * r + i;
                            const unsigned s = ubfe(lo, P.off[r][i], 16);
                            const uint16_t v = exl3i_codebook(s ^ P.xr[r][i], 1);
                            if (P.off[r][i] + 16 > 32 || s != st[q] || v != va[q]) bad++;
                            cases++;
                        }
                    }
                }
            }
            printf("KA %d n %2d (K %-6g mask %04x): %s\n", ka, n, ka + n / 16.0, mask, bad ? "FAIL" : "ok");
            fails += bad;
        }
    printf("%d lane-position cases, %d fail -> %s\n", cases, fails, fails ? "FAIL" : "PASS");
    return fails != 0;
}
