#!/usr/bin/env python3
"""patch_qt_ldsbar.py SRC -- the integer quantizer's speed fix (V3 kernel pass, v5): in quant/quantize_tiles_optimized.cuh
every forward step's __syncthreads() becomes an LDS-only barrier (the cost arrays live in LDS; __syncthreads also waits
for every outstanding global history store, s_wait_storecnt, so each step stalled on it; the traceback reads the
history only after argmin_cost's full barrier). -DQT_FULLSYNC restores __syncthreads. Idempotent; refuses a changed file."""
import re, sys
p = f"{sys.argv[1]}/quant/quantize_tiles_optimized.cuh"; s = open(p).read()
if "qt_lds_barrier" in s: sys.exit(f"{p}: already patched")
lines = s.split("\n")
SITES = [201, 209, 286, 357, 412, 459, 547]       # 1-based, the forward lambdas' step barriers (K1 x2, K2/3, K4/5, K6 x2, K7/8)
for ln in SITES:
    assert lines[ln - 1].strip() == "__syncthreads();", (ln, lines[ln - 1])
    lines[ln - 1] = lines[ln - 1].replace("__syncthreads();", "qt_lds_barrier();")
inc = lines.index('#include "quantize_tiles_kernel.cuh"')
lines[inc + 1:inc + 1] = ["",
    "// forward-step barrier: LDS ordering only (costs in LDS); the history stores to global need not complete per step",
    "__device__ __forceinline__ void qt_lds_barrier()", "{",
    "#if defined(__HIP_PLATFORM_AMD__) && defined(__HIP_DEVICE_COMPILE__) && !defined(QT_FULLSYNC)",
    '    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");', "    __builtin_amdgcn_s_barrier();",
    '    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");', "#else", "    __syncthreads();", "#endif", "}"]
open(p, "w").write("\n".join(lines)); print(f"patched {len(SITES)} barriers in {p}")
