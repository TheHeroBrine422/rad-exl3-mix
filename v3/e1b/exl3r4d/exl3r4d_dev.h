/* Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE. */
/* exl3r4d_dev.h -- what the C host side (nt.c) hands the device side (exl3r4d_nt.hip): one plain
 * struct of kernel arguments and the extern "C" entry that launches it. No radiance types cross
 * this line except RadStream (an opaque pointer), so the .hip file needs nothing of rad_plugin.h.
 *
 * The struct is the kernel's argument block, passed by value: it has to stay under the engine's
 * 768-byte kernel-argument limit (docs/PLUGIN.md §5). Today it is about 240 bytes. */
#ifndef EXL3R4D_DEV_H
#define EXL3R4D_DEV_H

#include <stddef.h>
#include <stdint.h>

/* T2 (the two-pass prefill form, exl3r4d.hip): the dense op takes it from EXL3R4D_T2_MIN rows when its scratch holds
 * the decoded weight chunk past the GSCR region (nt.c sizes it; the chunk pointer travels in the argument block's
 * `raw`, which the dense op does not otherwise use). */
#ifndef EXL3R4D_T2_MIN
#define EXL3R4D_T2_MIN 512
#endif
#define EXL3R4D_T2_CHUNK_BYTES (192LL << 20)

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EXL3R4D_WGQ   /* argument-block fields of the MoE library this was cut from: off (the block is unchanged) */
#define EXL3R4D_WGQ 0
#endif
#ifndef EXL3R4D_SUHC
#define EXL3R4D_SUHC 0
#endif

/* One parity class of experts: the four plane tables (suh1 only for gate_up) and the class's
 * weight geometry. A table is the engine's pointer array when its *_s0 is 0, else a stacked base
 * with *_s0 bytes an entry (the form a tool holding plain buffers passes). */
typedef struct Exl3r4dCls {
    const void* codes;      /* the stored trellis record (layout.c), one entry an expert */
    const void* suh;        /* f16 [cols] */
    const void* suh1;       /* f16 [cols], gate_up only (the up half's input scales) */
    const void* svh;        /* f16 [rows] */
    const void* dn_suh;     /* f16 [rows / 2], gate_up with h_rot only: DOWN's suh of the same experts (v3 EPI 1) */
    int64_t     codes_s0, suh_s0, suh1_s0, svh_s0, dn_suh_s0;
    int32_t     rows, cols; /* this class's [N, K]; gate_up rows = 2 * width (gate then up) */
} Exl3r4dCls;

typedef struct Exl3r4dGemm {
    Exl3r4dCls     cls[2];
#include "exl3r4d_gemm_fields.h"
} Exl3r4dGemm;

/* The dense linear's device launch (exl3r4d_nt.hip): K 1..8, npw column tiles a wave (1, 2, 4), a->xr the GSCR
 * region, a->raw T2's decoded-weight chunk (or null). RAD_OK or a negative RAD_E_*; never allocates or synchronises. */
int exl3r4d_launch_nt(const Exl3r4dGemm* a, int bits, int npw, void* stream);

/* Dynamic LDS a launch may ask for: 64 KiB less room for static __shared__. */
#define EXL3R4D_LDS_MAX (64 * 1024 - 256)

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3R4D_DEV_H */
