# Our changes to exllamav3 (the quantizer core)

`quantize/ourq.py` drives exllamav3's EXL3 quantizer (LDLQ + trellis search). Our changes to exllamav3 itself are
this patch series against upstream **turboderp-org/exllamav3 @ `a3bfbfd0`** (v1.6.3, MIT licence, © Turboderp):

| patch | what |
|---|---|
| 0001 | in-place output-scale fold (memory for the 248k-row head) |
| 0002 | quarter-step / fractional-K quantizer instances (V3 menu), canonical frac mask |
| 0003–0005 | int8-rounded mul1 codebook (`EXL3_MUL1_I8`), used by both decode paths |
| 0006 | int4-grid codebook (`EXL3_MUL1_I4`, measurement only) |
| 0007 | cubic int8 codebook i8c (`EXL3_I8_CUBIC`, #78) |
| `pending/` | not applied (dead-proxy-compute skip, an experimental quantize.py) |

Rebuild the tree we run (checked 2026-11-04: identical to our fork's tree at `eb5a2139`):

    git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3 && git checkout a3bfbfd0
    git am /path/to/rad-exl3-mix/quantize/exl3-patches/000*.patch

The fractional (qtf2) trellis kernels and the codebook builds that `ourq.py --qtf2 / --codebook` compile at run
time are in this repo: `v3/e3/exl3frac2/`, `v3/e3/exl3frac/`, `v3/e3/exl3cb/`. Python environment:
`quantize/exllamav3-environment.txt`.
