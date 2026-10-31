/* Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE. */
/* layout.c -- the codes plane's stored form: exllamav3's tile-stream record (common/exl3_layout.h), tile (tk, tn) of
 * 16 x 16 weights as 8*K little-endian u32 words, tiles k-major (a stacked gate_up: gate's record then up's). Lossless
 * and derived from nothing but the bits. common/exl3_layout.c is the bit-at-a-time oracle; K = 4 has a byte-shuffle
 * fast path (tile position p = 8L + j holds W_hat[k = 2(L%4) + {0,1,8,9}[j%4]][n = L/4 + 8(j/4)]), checked against
 * the oracle both ways. The relayout runs behind the disk read, so it has to keep pace. */
#include "exl3r4d.h"

#include <string.h>

/* ------------------------------------------------------------------ K = 4, fast */

static uint8_t swapnib(uint8_t b) { return (uint8_t)((b << 4) | (b >> 4)); }

void exl3r4d_k4_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                          uint32_t* record) {
    const int64_t nt = rows / 16, kt = cols / 16;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            const uint8_t* R = codes + tn * 16 * row_bytes + tk * 8;  /* row c at R + c*row_bytes */
            uint32_t* w = record + (tk * nt + tn) * 32;
            for (int cc = 0; cc < 8; ++cc) {
                const uint8_t* a = R + cc * row_bytes;
                const uint8_t* b = R + (cc + 8) * row_bytes;
                for (int l = 0; l < 4; ++l)
                    w[4 * cc + l] = (uint32_t)swapnib(a[l]) << 24 | (uint32_t)swapnib(a[l + 4]) << 16 |
                                    (uint32_t)swapnib(b[l]) << 8 | (uint32_t)swapnib(b[l + 4]);
            }
        }
}

void exl3r4d_k4_from_record(const uint32_t* record, int64_t rows, int64_t cols, uint8_t* codes,
                            int64_t row_bytes) {
    const int64_t nt = rows / 16, kt = cols / 16;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            uint8_t* R = codes + tn * 16 * row_bytes + tk * 8;
            const uint32_t* w = record + (tk * nt + tn) * 32;
            for (int cc = 0; cc < 8; ++cc) {
                uint8_t* a = R + cc * row_bytes;
                uint8_t* b = R + (cc + 8) * row_bytes;
                for (int l = 0; l < 4; ++l) {
                    const uint32_t x = w[4 * cc + l];
                    a[l] = swapnib((uint8_t)(x >> 24));
                    a[l + 4] = swapnib((uint8_t)(x >> 16));
                    b[l] = swapnib((uint8_t)(x >> 8));
                    b[l + 4] = swapnib((uint8_t)x);
                }
            }
        }
}

/* ------------------------------------------------------------------ any K (R23), fast enough for the loader
 *
 * The same permutation for every K: tile (tk, tn) holds 256 symbols in trellis-position order, position p at
 * (k = 16 tk + r, n = 16 tn + c) with (r, c) = exl3_pos_to_tile(p), each symbol K bits MSB-first in a stream of
 * u32 words that are themselves MSB-first (stream bit b is bit 31 - b % 32 of word b / 32; common/exl3_decode.c
 * exl3_tile_pack). A canonical row keeps its K-bit symbols LSB-first from bit k K (exl3_codes_get), so a tile's
 * 16 symbols of one row are 2K whole bytes at byte 2K tk. One tile: its 16 row segments read once, then the 256
 * symbols through a 64-bit accumulator. Checked against common/exl3_layout.c both ways (tests/gpu/layout_test.c). */
static void tile_order(int tr[256], int tc[256]) {
    for (int p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
}

void exl3r4d_kx_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols, int bits,
                          uint32_t* record) {
    int tr[256], tc[256];
    tile_order(tr, tc);
    const int64_t nt = rows / 16, kt = cols / 16, seg = 2 * bits;   /* a row's 16 symbols: 2K bytes */
    const uint32_t mask = (1u << bits) - 1u;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            uint8_t sym[16][16];   /* [c][r]: row n = 16 tn + c, column k = 16 tk + r */
            for (int c = 0; c < 16; ++c) {
                uint8_t b[18] = { 0 };
                memcpy(b, codes + (tn * 16 + c) * row_bytes + tk * seg, (size_t)seg);
                for (int r = 0; r < 16; ++r) {
                    const int o = r * bits;
                    sym[c][r] = (uint8_t)(((uint32_t)b[o >> 3] | (uint32_t)b[(o >> 3) + 1] << 8) >> (o & 7) & mask);
                }
            }
            uint32_t* w = record + (tk * nt + tn) * 8 * bits;
            uint64_t acc = 0;
            int nb = 0;
            for (int p = 0; p < 256; ++p) {
                acc = acc << bits | sym[tc[p]][tr[p]];
                nb += bits;
                if (nb >= 32) {
                    nb -= 32;
                    *w++ = (uint32_t)(acc >> nb);
                    acc &= (1ull << nb) - 1ull;
                }
            }
        }
}

void exl3r4d_kx_from_record(const uint32_t* record, int64_t rows, int64_t cols, int bits, uint8_t* codes,
                            int64_t row_bytes) {
    int tr[256], tc[256];
    tile_order(tr, tc);
    const int64_t nt = rows / 16, kt = cols / 16, seg = 2 * bits;
    const uint32_t mask = (1u << bits) - 1u;
    for (int64_t tk = 0; tk < kt; ++tk)
        for (int64_t tn = 0; tn < nt; ++tn) {
            uint8_t sym[16][16];
            const uint32_t* w = record + (tk * nt + tn) * 8 * bits;
            uint64_t acc = 0;
            int nb = 0;
            for (int p = 0; p < 256; ++p) {
                if (nb < bits) {
                    acc = acc << 32 | *w++;
                    nb += 32;
                }
                nb -= bits;
                sym[tc[p]][tr[p]] = (uint8_t)(acc >> nb & mask);
                acc &= (1ull << nb) - 1ull;
            }
            for (int c = 0; c < 16; ++c) {
                uint8_t b[18] = { 0 };
                for (int r = 0; r < 16; ++r) {
                    const int o = r * bits;
                    const uint32_t v = (uint32_t)sym[c][r] << (o & 7);
                    b[o >> 3] |= (uint8_t)v;
                    b[(o >> 3) + 1] |= (uint8_t)(v >> 8);
                }
                memcpy(codes + (tn * 16 + c) * row_bytes + tk * seg, b, (size_t)seg);
            }
        }
}

