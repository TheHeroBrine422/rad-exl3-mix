#pragma once
// exl3hashm (#177, w139): cb_hash.cuh plus the optional 256-entry map (EXL3_HASH_MAPPED, exl3hashm.py writes it).
// exl3hash (#100c, w48): IQR-V4's hash codebook on EXL3's 16-bit trellis states, for the quantizer only (see
// exl3hash.py). q(s) = bytesum(((s * A + B) mod 2^32) & 0x3F3F3F3F) - 126 (libr4d/r4d_iqr_dec.h's MAD branch; for
// s < 2^24, s * A mod 2^32 is __umul24(s, A)), value = q * 4 k_inv: mul1_i8's int8 grid and step, so the two
// codebooks differ only in which q a state reads (both have sd ~36.9 steps). exl3hash.py writes EXL3_HASH_A / _B.
#ifndef EXL3_HASH_A
#error "EXL3_HASH_A: include the cb_hash.cuh that exl3hash.py generates"
#endif

// codebook.cuh's mul1 paths hand over products p = s * 0x83DCD12D (mod 2^32); the inverse gives s back
#define EXL3_MUL1_INV 0x8172A6A5u
static_assert(0x83DCD12Du * EXL3_MUL1_INV == 1u, "mul1 multiplier inverse");

__device__ __forceinline__ half exl3_hash_value(uint32_t state)
{
    const uint32_t h = ((state & 0xFFFFFFu) * (uint32_t) EXL3_HASH_A + (uint32_t) EXL3_HASH_B) & 0x3F3F3F3Fu;
#ifdef EXL3_HASH_MAPPED
    const int q = (int) exl3_hash_map[__dp4a(h, 0x01010101u, 0u)];   // exl3hashm (#177): HG / HC output map
#else
    const int q = (int) __dp4a(h, 0x01010101u, 0u) - 126;
#endif
    const float s4 = 4.0f * __half2float(__ushort_as_half(0x1eee));
    return __float2half_rn((float) q * s4);
}

__device__ __forceinline__ half2 exl3_hash_product_2(uint32_t x0, uint32_t x1)
{
    return __halves2half2(exl3_hash_value(x0 * EXL3_MUL1_INV), exl3_hash_value(x1 * EXL3_MUL1_INV));
}
