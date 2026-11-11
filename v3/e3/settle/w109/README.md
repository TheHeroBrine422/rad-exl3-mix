# #144 · w109 · swj level-pass speed

Rule: `../w109.md` (b6433ad, committed before data). Data: `../data/w109/`. LOG: 2026-11-11 "#144".
**Runtime:** none (quantizer-side pass time only; same format and kernels).

## Result
- **Profile** (dad1 RX 9070 XT, today's `rxq.swjfit --fused --no-cache`, 2 sources, 8 layers, 100 qad docs, 143k tokens,
  synchronized timers, `../data/w109/prof2.json`): f32 GEMMs **70%** (sources 46%, bf16 reference gate / up 25%), bf16
  forward 13%, fused grid sums 10%, load + decode + boundary I/O + fit 6%. The 9070 XT is as fast as the R9700 here
  (S=2: 49.6 vs 52.9 min a level, scaled).
- **(a) output-identical:** the f32 GEMM is hipBLASLt's sequential FMA over k (`gemmorder.txt`: 8/8 outputs, every
  split / interleave fails); layouts, merged N, stacked documents and every BLAS backend give the same bits at the same
  ~15-17 TFLOP/s (`gemmbench.json`); Triton's f32 `tl.dot` is slower (≤ 11 TF) and not identical (`trigemm.txt`). The grid
  kernel is compute-bound: a launch-order change (same bits) gives x0.91-1.05, and the identical tile configs are slower
  (`gridkernel.txt`). Async boundary writes + read prefetch + one aa² sum (`--fast`): identical, −3.8% at S=8, +1.4% at
  S=12 → below the 5% bar, dropped. **What remains is sources per pass:** t(S) ≈ 52 + 32.4 S s (dev), per level 83 / 57 /
  46 / 39 / 37 s at S = 1 / 2 / 4 / 8 / 12; 16 GB holds S=12 (peak 11.3 GiB, one layer a stage).
- **(b) proposal** (in scope: (a) leaves swj at ~29% of the ladder): `--numerics bf16` = the reference gate / up as bf16 x
  bf16 GEMMs (f32 accumulation: exact products, other order) + each source weight as two bf16 terms in one GEMM over 2K.
  2.1-2.3x faster passes; held-out gain −0.06% / −0.05% loss (better), spread 1.19% → **PASS**; gate c moved on 1.4-1.5%
  of channels, fewer than today's code moves between the R9700 and the 9070 XT (2.2%, `judge-machine.txt`).
  Fewer fit tokens **FAIL** (half the fit half: −5.6% of the held-out gain vs a 1.2% spread). Coarser grid not tried
  (grid sums 10-21% of a pass); `--grid-fast` (non-IEEE divide) was a no-op.
- **Projection** (`proj109.py`, `projection.txt`): swj on card 0, 36 levels: today S=2 30.8 h (ladder 3.33-3.49 card-days),
  S=8 21.4 h (2.93-3.10), S=12 20.3 h (2.89-3.06); with (b) S=12 9.2 h (2.43-2.59, swj 15%). On dad1 instead: card 0
  2.04-2.21 days, dad1 19.0 h (today, S=12) / 8.6 h ((b), S=12).

## Files
| file | what |
|---|---|
| `swjfit109.py` | copy of `rxq/swjfit.py` + `--prof` (section timers), `--fast` (I/O overlap, shared aa²), (b) options `--ref-bf16`, `--src-split N`, `--grid-fast`, `--fit-sub K` |
| `swjtri109.py` | grid sums with the launch grid reordered (identical), the shared aa², the non-IEEE-divide variant; bench |
| `swj-numerics.patch` | **the proposal**: `rxq/swjfit.py --numerics f32|bf16` (split per tensor: no extra VRAM) + `rxrun/run.py` passing `[post.swj] numerics`; applies to 640ecfd / b6433ad (`git apply -p1`) |
| | **Stamps:** `rxq.engine.code_stamp` hashes every `rxq/*.py`, so merging the patch moves the code stamp even at the default `f32`, and `[post.swj] numerics = "bf16"` moves the settings stamp: merge before the ladder starts (freeze) or not at all |
| `dev.sh` | one dev pass on dad1 (truncated model, card0.lock); `PP=` another pipeline tree |
| `trunc.py` | a checkpoint / level cut to its first L layers (symlinks) |
| `cmp109.py`, `bstat.py` | identity judge (2a), held-out judge (2b) |
| `gemmbench.py`, `gemmorder.py`, `trigemm.py`, `trisweep.py` | GEMM / kernel identity and speed probes |
| `sumruns.py`, `proj109.py` | run table, ladder projection |

## Rerun (dad1)
`/data/w109`: `m8` = the bf16 27B cut to layers 0-7 (shards 1-3 from HF), `lv8/fs{0,1}` = w92's K2.5 i8 levels L0-7,
`meta/swj27-qad`, `pylib` (transformers 5.22.1, fla 0.6.0, `pip --target`), `env.sh`. Then e.g.
`code/v3/e3/settle/w109/dev.sh ref8 ref 100 "fs0 fs1 fs0:b fs1:b fs0:c fs1:c fs0:d fs1:d"` and
`PP=/data/w109/codeP/pipeline:/data/w109/pylib .../dev.sh patbf ref 100 "fs0 fs1" --numerics bf16` (codeP = pipeline +
the patch).
