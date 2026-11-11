# sub1fuse (#142, w108): qtf2's KA = 0 (sub-1) forward pass with the 0-bit steps fused

Rule: `../SUB1FUSE.md` (committed alone, a99c06d, before any measurement). Results: `../data/sub1fuse/`.
Not merged: the change is `qtf2-sub1fuse.patch` against `v3/e3/exl3frac2/qtf2.cuh` (HEAD 518a7c3 / a99c06d); the
patched file is `qtf2.cuh` here. Only additions: the KA = 0 kernels call `qtf2_forward0`; every KA ≥ 1 instance and
the KA = 0 traceback / argmin are the same code (the traceback reads the history through `qtf2_adr0`).

## What it does

In the KA = 0 kernel (512 threads, 65,536 states, costs in global memory) every position used to read and write all
state costs in global memory behind a block barrier. A sub-1 state only moves at a branching position, so:

- each thread keeps the costs of 128 states it owns in registers (`acc[64]` half2) across (0, 0) steps; the (0, 1) min
  step's two candidates (bit 15) are the thread's own; global memory and barriers only where costs cross threads
  ((0, 1) / (1, 1) writes, (1, 0) / (1, 1) reads);
- codebook reads come from a 1,024-entry byte-sum table in LDS: `decode_mul1_product_2`'s value depends only on the byte
  sum of the 32-bit product, so `tab[sum]` (filled by the kernel with `decode_mul1_product_2` itself) gives the same
  value for every codebook build (mul1, i8, i8c, i4); products by additions only (no 32-bit multiply per value);
- state ownership puts a wave's lanes on state bits 10-14: the table reads of one instruction then hit distinct LDS
  banks (lanes 8 states apart conflicted ~2× more: simulated 2.8 vs 1.1 dwords per bank, measured 2× slower);
- the 15-bit costs and the history bytes (kernel-internal scratch) are stored at `qtf2_adr0(node)`, a bit permutation
  that makes both the writer's and the reader's accesses 16 / 8-byte vectors with the lanes contiguous;
- kernels compile only the step kinds their MASK has (registers: occupancy 7 → 8 waves / SIMD at K0.125-0.375).

Same arithmetic per state in the same order (half FMA chain, salted / plain reads, first-step masks, strict-less /
packed-minimum picks): bit-identical output, judge (a). Test builds: `QTF2_NOFUSE` (today's per-step path, = the old
kernel), `QTF2_F0NOGLOB` / `QTF2_F0NOTAB` (timing only, output invalid). With the fusion on, `QTF2_LUT=0/1` no longer
changes the KA = 0 path (both read the byte-sum table; the 65,536 LUT is still built by the binding, unused at KA 0).

## Files

| file | what |
|---|---|
| `qtf2.cuh`, `qtf2-sub1fuse.patch` | the patched kernel / the patch (`git apply v3/e3/exl3frac2/sub1fuse/qtf2-sub1fuse.patch`) |
| `fusecheck.py` | judge (a): `run TREE SRC VARIANT OUT` (child, one build) and `cmp REF.pt NEW.pt` |
| `speed.py` | tiles/s of the KA = 0 instances by launch batch (qtf2sub1.py's method, best of 3) |
| `final.sh` | the final suite on one card (identity i8 + mul1, speed ref vs new; as run, it compared against the ref outputs of the dev loop's identical ref build, this copy regenerates them) |
| `build.py`, `env-dad1.sh` | dad1's build helper and environment |
| `banksim.py` | the LDS bank-conflict simulation behind the lane mapping |
| `gpubox.sh` | one real K0.125 layer through `rxq job` on gpubox card 0, today's qtf2 vs this one (identity + seconds) |

## How it was run (dad1: RX 9070 XT, gfx1201)

`/data/w108/`: `fork/` (git archive of the fork at b0052f7c: `exllamav3/exllamav3_ext`), `stub/exllamav3/util/cuda_flags.py`
(exl3frac2.load needs only that module: the stub avoids importing the package), `ref/` (HEAD's `v3/e3/exl3frac2`),
`new/` (the same + this `qtf2.cuh`), `pylib/` (ninja), `tools/` (these scripts + `env.sh`: `ROCM_HOME=/opt/rocm`,
`PYTORCH_ROCM_ARCH=gfx1201`, `ROCR_VISIBLE_DEVICES=0` = the 9070 XT, never the gfx1036 iGPU). GPU runs under
`flock /root/card0.lock`. `bash tools/final.sh /data/w108`.

## Results (details and raw files: `../data/sub1fuse/README.md`)

- **(a) identity: PASS.** Both codebooks (i8, mul1), all seven KA = 0 instances, LUT = 1 / 0, batches 32 / 128, 2,456
  tiles (qtf2check's set + a second Gaussian set): 0 differing indices / values; every KA ≥ 1 instance identical; 4
  runs deterministic; the `QTF2_NOFUSE` build of the new source = the old kernel.
- **Speed, i8, batch 128 (production's), tiles/s old → new:** K0.125 12.0k → 45.3k (×3.77), K0.25 ×3.26, K0.375 ×2.89,
  K0.5 14.1k → 36.5k (×2.60), K0.625 ×2.11, K0.75 ×1.80, K0.875 16.5k → 28.3k (×1.71). mul1 similar.
- **Real layer, gpubox card 0 (R9700), `rxq job`, today's settings (524bc868), K0.125 i8, layer 10 (4 group jobs, 6
  tensors):** every tensor file bit-identical to today's kernel; tensor seconds 167.6 → 62.0 (×2.70), job wall 186 → 82 s
  (×2.27, process start and loads unchanged). The kernel ratio implied on the R9700 is ×3.3 at K0.125 (dad1 ×3.8).
- **Projection (R9700-calibrated):** the seven KA = 0 levels' job wall ≈ 21.8 → 12.2 card-h (K0.125 3.53 → 1.48 h).
- **Step 2 (macro-steps) skipped** by the rule: global traffic + barriers are 5.0% of the K0.125 kernel time
  (`QTF2_F0NOGLOB` timing build: 48.9k → 51.5k tiles/s; < 30%). What is left is the per-position codebook read of every
  state, which a macro-step trellis would still need (its emission costs are sums of the same reads).
