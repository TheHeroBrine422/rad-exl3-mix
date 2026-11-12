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
| 0008 | block-diagonal two-sided LDLQ for a diagonal H_out (`prepare_H_out_blocks`, `ldlq_2hess_blocks`; #115, w81, gate PASS 11-08): a 1-D H_out selects it, `hout_dense` keeps the dense path |
| 0009 | proxy metric moves `weight_q` to `weight_r`'s device: the two-sided path on the LM head (> 5e8 weights, CPU weight) crashed there (w94, #125; trunk tensors unaffected) |
| 0010 | the scale refit un-rotates H on the job's device (the card), on the CPU on out-of-memory: while the un-rotation and the refit fit on the card, the host BLAS does not decide suh / svh's last bit (GitHub issue 2; `pipeline/PORTING.md`, "The scale refit and the host CPU") |
| `pending/` | not applied (dead-proxy-compute skip, an experimental quantize.py) |

Rebuild the tree we run (checked 2026-11-04: identical to our fork's tree at `eb5a2139`; with 0008, checked 2026-11-08
by w86: upstream `a3bfbfd0` + 0001–0008 = `eb5a2139` + 0008, tree `8b6ab82b`. 0008 is in the fork since 11-08
(HEAD `6242f7ae`, applied by the lead after the freeze review); 0009 since 11-09 (HEAD `b0052f7c`; `git am` of 0001–0009 on a3bfbfd0 = the fork's tree, checked).
0010 (2026-11-10): `git am` of 0001–0010 on a3bfbfd0 = tree `e3a7a8b8`; not in the fork HEAD named above (`b0052f7c`)):

    git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3 && git checkout a3bfbfd0
    git am /path/to/rad-exl3-mix/quantize/exl3-patches/00*.patch

The fractional (qtf2) trellis kernels and the codebook builds that `ourq.py --qtf2 / --codebook` compile at run
time are in this repo: `v3/e3/exl3frac2/`, `v3/e3/exl3frac/`, `v3/e3/exl3cb/`. Python environment:
`quantize/exllamav3-environment.txt`.
