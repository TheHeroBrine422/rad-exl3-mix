// asmlit.hip -- #175 step 0 (w137): every inline-asm form of exl3r4d_common.hip.h / exl3i_rate.hip.h with its SGPR
// constant ("s") against the same instruction with the constant in the asm text (inline constant with op_sel_hi 0, or
// one 32-bit literal): outputs bit for bit over 2^26 inputs (a hash of the index, all 32-bit patterns of the low half).
#include <hip/hip_runtime.h>
#include <cstdio>
#define NF 12
__device__ __forceinline__ void forms(unsigned x, unsigned y, unsigned (&o)[NF], unsigned (&n)[NF]) {
  // exl3r4d hash_pair
  asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(o[0]) : "v"(x), "s"(0x83DC0000u));
  asm("v_pk_mul_lo_u16 %0, %1, 0x83dc0000 op_sel_hi:[0,1]" : "=v"(n[0]) : "v"(x));
  asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(o[1]) : "v"(x), "s"(0x83DC0000u));
  asm("v_pk_mul_lo_u16 %0, %1, 0x83dc0000" : "=v"(n[1]) : "v"(x));
  asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(o[2]) : "v"(x), "s"(0xD12Du), "v"(y));
  asm("v_mad_u32_u16 %0, %1, 0xd12d, %2" : "=v"(n[2]) : "v"(x), "v"(y));
  asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(o[3]) : "v"(x), "s"(0xD12Du), "v"(y));
  asm("v_mad_u32_u16 %0, %1, 0xd12d, %2 op_sel:[1,0,0,0]" : "=v"(n[3]) : "v"(x), "v"(y));
  // exl3i_i4n
  asm("v_pk_mad_i16 %0, %1, %2, %3" : "=v"(o[4]) : "v"(x), "s"(0x00050005u), "s"(0xFE08FE08u));
  asm("v_pk_mad_i16 %0, %1, 5, 0xfe08fe08 op_sel_hi:[1,0,1]" : "=v"(n[4]) : "v"(x));
  asm("v_pk_ashrrev_i16 %0, %1, %2" : "=v"(o[5]) : "s"(0x00080008u), "v"(x));
  asm("v_pk_ashrrev_i16 %0, 8, %1 op_sel_hi:[0,1]" : "=v"(n[5]) : "v"(x));
  asm("v_pk_min_i16 %0, %1, %2" : "=v"(o[6]) : "v"(x), "s"(0x000F000Fu));
  asm("v_pk_min_i16 %0, %1, 15 op_sel_hi:[1,0]" : "=v"(n[6]) : "v"(x));
  // fin16 / fin8 / i8 closed form
  asm("v_pk_mad_u16 %0, %1, %2, %3" : "=v"(o[7]) : "v"(x), "s"(0x00020002u), "s"(0x64016401u));
  asm("v_pk_mad_u16 %0, %1, 2, 0x64016401 op_sel_hi:[1,0,1]" : "=v"(n[7]) : "v"(x));
  asm("v_pk_sub_u16 %0, %1, %2" : "=v"(o[8]) : "v"(x), "s"(0x65FF65FFu));
  asm("v_pk_sub_u16 %0, %1, 0x65ff65ff" : "=v"(n[8]) : "v"(x));
  asm("v_pk_mad_u16 %0, %1, %2, %3" : "=v"(o[9]) : "v"(x), "s"(0x00020002u), "s"(0x00710071u));
  asm("v_pk_mad_u16 %0, %1, 2, 0x00710071 op_sel_hi:[1,0,1]" : "=v"(n[9]) : "v"(x));
  asm("v_pk_lshrrev_b16 %0, %1, %2" : "=v"(o[10]) : "s"(0x00020002u), "v"(x));
  asm("v_pk_lshrrev_b16 %0, 2, %1 op_sel_hi:[0,1]" : "=v"(n[10]) : "v"(x));
  asm("v_pk_max_u16 %0, %1, %2" : "=v"(o[11]) : "v"(x), "s"(0x00010001u));
  asm("v_pk_max_u16 %0, %1, 1 op_sel_hi:[1,0]" : "=v"(n[11]) : "v"(x));
}
__global__ void k(unsigned* bad, unsigned* ex) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned x = (i & 0xFFFFu) | ((i * 0x9E3779B9u) & 0xFFFF0000u), y = i * 0x85EBCA6Bu ^ (i >> 7);
  unsigned o[NF], n[NF];
  forms(x, y, o, n);
  for (int f = 0; f < NF; f++)
    if (o[f] != n[f]) { atomicAdd(&bad[f], 1u); ex[3 * f] = x; ex[3 * f + 1] = o[f]; ex[3 * f + 2] = n[f]; }
}
int main() {
  unsigned *bad, *ex, hb[NF], he[3 * NF];
  hipMalloc(&bad, sizeof hb); hipMalloc(&ex, sizeof he); hipMemset(bad, 0, sizeof hb); hipMemset(ex, 0, sizeof he);
  k<<<(1u << 26) / 256, 256>>>(bad, ex);
  if (hipDeviceSynchronize() != hipSuccess) { puts("launch failed"); return 2; }
  hipMemcpy(hb, bad, sizeof hb, hipMemcpyDeviceToHost); hipMemcpy(he, ex, sizeof he, hipMemcpyDeviceToHost);
  int tot = 0;
  for (int f = 0; f < NF; f++) {
    printf("form %2d: %u of %u differ", f, hb[f], 1u << 26);
    if (hb[f]) printf("  e.g. x %08x s-form %08x literal %08x", he[3 * f], he[3 * f + 1], he[3 * f + 2]);
    puts(""); tot += hb[f] != 0;
  }
  printf("ASMLIT %s\n", tot ? "FAIL" : "PASS");
  return tot != 0;
}
