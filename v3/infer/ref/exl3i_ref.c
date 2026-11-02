/* exl3i_ref.c -- THE ORACLE for EXL3-V3's weight codes (encoding "exl3i"): plain C99, slow and readable on purpose.
 * Every V3 decoder (importer, f16 pre-decode, GEMV, int8 path) is checked bit for bit against this file.
 *
 * Tile format (exllamav3 1.6.3 / the rxm-frac-quarter fork's pack_trellis_frac, DESIGN-v3 §1c.7):
 *   - a 256-weight tile at rate K = KA + n/16 (n even, KA >= 0) is 16 K uint16 words = 8 K uint32 words, read as ONE
 *     MSB-first bit ring of R = 256 K bits (32-bit words, word w holds ring bits [32 w, 32 w + 32), bit 31 first);
 *   - ring position i (0..255) takes D(i) = KA + bit (i mod 16) of MASK new bits, MASK = frac_mask(n) (the even
 *     spread: bit i set iff floor((i + 1) n / 16) - floor(i n / 16) = 1); integer K: MASK = 0, D = K;
 *   - S(i) = D(0) + ... + D(i) (ring bit where position i's window ends); its 16-bit STATE is ring bits
 *     [S(i) - 16, S(i)), circular;
 *   - its VALUE is codebook(state ^ salt(i)) when D(i) = 0 (sub-1: the 0-bit step keeps the state and reads a
 *     position-salted index, salt(i) = (0x9E37 (i mod 16 + 1)) mod 2^16), codebook(state) otherwise;
 *   - codebook "mul1": v = fp16(fma(fp16(1024 + bytesum(state * 0x83DCD12D)), K_INV, K_BIAS)) with one rounding,
 *     K_INV = fp16 0x1EEE, K_BIAS = fp16 0xC931; "mul1i8": q = clamp(rint(f32(v) * (1 / (4 K_INV))), -127, 127),
 *     value fp16(q * 4 K_INV) (what the quantizer's search used: codebook.cuh mul1_i8_round);
 *   - ring position i sits at tile element (r, c) of exl3i_perm(i) (exllamav3's tensor_core_perm).
 * Build with -ffp-contract=off. */
#include <math.h>
#include <stdint.h>
#include <string.h>

static float h2f(uint16_t h) {               /* fp16 -> f32, exact */
    const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else { float f = ldexpf((float)m, -24); memcpy(&u, &f, 4); u |= s; }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

static uint16_t d2h(double x) {              /* f64 -> fp16, round to nearest even (one rounding) */
    uint16_t s = 0;
    if (x < 0 || (x == 0 && signbit(x))) { s = 0x8000; x = -x; }
    if (isnan(x)) return 0x7e00;
    if (x >= 65520.0) return s | 0x7c00;
    if (x < ldexp(1.0, -14)) {                /* subnormal: units of 2^-24 */
        double q = x * ldexp(1.0, 24); double r = nearbyint(q);
        return s | (uint16_t)r;               /* r <= 1024 (1024 = smallest normal) */
    }
    int e; double m = frexp(x, &e);           /* x = m 2^e, m in [0.5, 1) */
    double q = m * 2048.0;                    /* 11 significant bits */
    double r = nearbyint(q);
    if (r == 2048.0) { r = 1024.0; e += 1; }
    if (e + 14 >= 31) return s | 0x7c00;
    return s | (uint16_t)(((e + 14) << 10) | ((uint32_t)r - 1024));
}

int exl3i_frac_mask(int n) {                  /* n = 16 x the fractional part (even) */
    int m = 0, i;
    for (i = 0; i < 16; ++i) if (((i + 1) * n) / 16 - (i * n) / 16) m |= 1 << i;
    return m;
}

uint16_t exl3i_salt(int i) { return (uint16_t)((0x9E37u * (uint32_t)((i & 15) + 1)) & 0xFFFFu); }

/* mul1 codebook value of a 16-bit state (fp16 bits); i8 = 1: the int8-rounded variant */
uint16_t exl3i_codebook(uint32_t state, int i8) {
    const uint32_t x = (state & 0xFFFFu) * 0x83DCD12Du;
    const uint32_t bs = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    const double h = 1024.0 + (double)bs;     /* exact fp16 (an integer < 2048) */
    const double kinv = h2f(0x1EEE), kbias = h2f(0xC931);
    const uint16_t v = d2h(h * kinv + kbias); /* h * kinv exact in f64 (11 x 11 bits), + kbias exact: one rounding */
    if (!i8) return v;
    const float s4 = 4.0f * (float)kinv, inv = 1.0f / s4;
    float q = rintf(h2f(v) * inv);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return d2h((double)(q * s4));             /* f32 product, then fp16 (as __float2half_rn(q * s4)) */
}

/* exllamav3's tensor_core_perm: ring position i -> tile element r * 16 + c */
int exl3i_perm(int i) {
    const int t = i >> 3, j = i & 7, r0 = (t & 3) * 2, c0 = t >> 2;
    const int r = r0 + ((j & 1) ? 1 : 0) + ((j & 2) ? 8 : 0), c = c0 + ((j & 4) ? 8 : 0);
    return r * 16 + c;
}

/* One tile: words = 8 K uint32 (the tile's 16 K uint16 read as uint32, little-endian), ka / mask as above.
 * states[256] and values[256] (fp16 bits) in RING order; either may be null. Returns 0, or -1 on a bad rate. */
int exl3i_decode_tile(const uint32_t* words, int ka, int mask, int i8, uint16_t* states, uint16_t* values) {
    int bpb = 16 * ka, i, S = 0;
    for (i = 0; i < 16; ++i) bpb += (mask >> i) & 1;          /* bits per 16 positions */
    if (ka < 0 || ka > 8 || mask < 0 || mask > 0xFFFF || bpb % 2 || bpb == 0) return -1;
    const int nw = bpb / 2, R = nw * 32;                      /* 256 positions = 16 x bpb bits = 8 bpb ... */
    for (i = 0; i < 256; ++i) {
        S += ka + ((mask >> (i & 15)) & 1);
        int start = (S - 16) % R; if (start < 0) start += R;
        const int wi = start >> 5, o = start & 31;
        const uint64_t v = ((uint64_t)words[wi] << 32) | (uint64_t)words[(wi + 1) % nw];
        const uint32_t st = (uint32_t)((v >> (48 - o)) & 0xFFFFu);
        const int d = ka + ((mask >> (i & 15)) & 1);
        if (states) states[i] = (uint16_t)st;
        if (values) values[i] = exl3i_codebook(d ? st : (st ^ exl3i_salt(i)), i8);
    }
    return 0;
}
