/* decbench.c -- time rad-convert's error-report decode of one exl3i weight (w123, #161): exl3i_decode_rows (the plugin's
 * decode hook = the oracle) over a ffn_gate_up-sized stacked weight [34816 x 5120] with random trellis words, once in one
 * call (what rad-convert does for a quantiser's own scheme) and once split over T threads by 128-row blocks (the oracle
 * is reentrant); checks the two outputs are bit-identical.
 *   cc -O2 -fopenmp -ffp-contract=off -I$P/common -I$P/include decbench.c $P/common/exl3i_decode.c $P/common/exl3_decode.c -lm
 *   ./a.out [M=40 (K2.5)] [THREADS=8] */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "exl3i_decode.h"

int main(int argc, char** argv) {
    const int m = argc > 1 ? atoi(argv[1]) : 40, T = argc > 2 ? atoi(argv[2]) : 8;
    const int64_t rows = 34816, cols = 5120, rw = cols / 16 * (m / 2);
    uint32_t* rec = malloc(sizeof *rec * (size_t)(rows / 16 * rw));
    uint16_t *suh = malloc(2 * cols), *suh1 = malloc(2 * cols), *svh = malloc(2 * rows);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int64_t i = 0; i < rows / 16 * rw; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; rec[i] = (uint32_t)s; }
    for (int64_t i = 0; i < cols; ++i) suh[i] = suh1[i] = 0x3C00;   /* 1.0 */
    for (int64_t i = 0; i < rows; ++i) svh[i] = 0x2C00;
    Exl3iView v = {rows, cols, m, EXL3I_CB_MUL1I8, rec, suh, suh1, svh};
    float *a = malloc(sizeof *a * (size_t)(rows * cols)), *b = malloc(sizeof *b * (size_t)(rows * cols));
    double t0 = omp_get_wtime();
    if (exl3i_decode_rows(&v, 0, rows, a)) return 1;
    double t1 = omp_get_wtime();
    omp_set_num_threads(T);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t b0 = 0; b0 < rows; b0 += 128) exl3i_decode_rows(&v, b0, 128, b + b0 * cols);
    double t2 = omp_get_wtime();
    printf("{\"m\": %d, \"rows\": %lld, \"cols\": %lld, \"one_call_s\": %.2f, \"threads\": %d, \"split_s\": %.2f, \"identical\": %s}\n", m,
           (long long)rows, (long long)cols, t1 - t0, T, t2 - t1, memcmp(a, b, sizeof *a * (size_t)(rows * cols)) ? "false" : "true");
    return 0;
}
