/* Derived from radiance-exl3 (https://github.com/Dyluhn/radiance-exl3, Apache-2.0, Copyright 2026 Dylan Johnston) and radiance's libr4d (Apache-2.0); EXL3 format per exllamav3 (MIT). Modified for this project; see NOTICE. */
/* exl3r4d.h -- what the host files of the gfx1201 EXL3 library share. */
#ifndef EXL3R4D_H
#define EXL3R4D_H

#include "exl3_abi.h"
#include "exl3_decode.h"
#include "exl3_layout.h"

#if EXL3_IFACE_VERSION != 8
#error "exl3r4d is written against EXL3_IFACE_VERSION 8 (include/exl3_abi.h: the dense op gemm_exl3_nt)"
#endif
#if EXL3_DECODE_API_VERSION != 1 || EXL3_LAYOUT_API_VERSION != 1
#error "exl3r4d is written against EXL3_DECODE_API_VERSION 1 and EXL3_LAYOUT_API_VERSION 1 (common/)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The stored codes plane: exllamav3's tile-stream record (common/exl3_layout.h), gate's record then
 * up's for a stacked gate_up (layout.c). Plugin-private (rad_types.h: >= RAD_DT_PLUGIN_BASE is sized by the
 * layout hook); the logical [rows, cols] stays the shape. */
enum { EXL3R4D_DT_TRELLIS = RAD_DT_PLUGIN_BASE + 0 };

/* The dense row (nt.c). */
int     exl3r4d_nt_launch(const RadArgs* a, RadStream s);
int64_t exl3r4d_nt_scratch(const RadArgs* a);
int     exl3r4d_nt_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                          const int* sel, const RadTensor* planes, int n_planes, RadLayout* out);
int     exl3r4d_nt_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                            const int* sel, const RadTensor* planes, int n_planes, void* dst,
                            int64_t dst_bytes);
int     exl3r4d_nt_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                              const int* sel, const void* src, int64_t src_bytes,
                              const RadTensor* planes, int n_planes);
/* The stored codes record, both ways (layout.c). */
void exl3r4d_k4_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                          uint32_t* record);
void exl3r4d_k4_from_record(const uint32_t* record, int64_t rows, int64_t cols, uint8_t* codes,
                            int64_t row_bytes);
void exl3r4d_kx_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols, int bits,
                          uint32_t* record);
void exl3r4d_kx_from_record(const uint32_t* record, int64_t rows, int64_t cols, int bits, uint8_t* codes,
                            int64_t row_bytes);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3R4D_H */
