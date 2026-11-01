#pragma once
#include <cuda_fp16.h>
#include <cstdint>
typedef void (*fp_qtf2)(const float*, float*, uint16_t*, half*, uint16_t*, const half2*);
struct Qtf2Entry { int ka; uint32_t mask; fp_qtf2 (*fn)(); int nt; int shmem; };
