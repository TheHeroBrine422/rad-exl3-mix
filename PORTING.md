# What is ported, and how each port was checked (2026-11-06)

Rule (task w52): no new quantizer behaviour; every port bit-identical to the experiment script it replaces, on small
cases (0.8B tensors or single 27B tensors). "Bit-identical" for files = the same tensor names, dtypes and values
(`torch.equal`) and the same metadata dict. Not file bytes: safetensors writes the `__metadata__` keys in hash order,
which differs between processes (found here: the first byte-level comparison failed on every file for that reason
alone, tensors and metadata equal).

| stage | rxq | original (rad-exl3-mix) | check | result |
|---|---|---|---|---|
| S2 tensor quant | `quant.py`, `engine.py` | `quantize/ourq.py` (ladder settings: i8, qtf2, scaled H, σ 0.025 / 0.005) | `tools/run4070.sh A`: 27B layer 10 (6 tensors) + layer 39 q/k/v/o at K2.5, one per-tensor job each, vs hconv `q-none` (ourq, 11-04) | RESULT_A |
| S2 with H_out | same (`--hout`, ldlq_2hess) | ourq `--hout DIR --sigma-reg-out 0.01` | `run4070.sh D`: 27B L10 in_proj_z / gate, L39 o_proj vs hconv `q-bn1664` | RESULT_D |
| S5 writers requant | same (`--hin`, `--wpre`) | ourq `--hin --wpre` (asym27 test08 G1 path) | `run4070.sh C`: rxq mkdenoise (GPU) on sib27test/ref2 → 48 0.8B writers vs d1103 `q-k3asym_r0` | RESULT_C |
| S5 G / F | `asym.py` Denoiser | `proxy2b/exp/w07-denoise/mkdenoise.py` | `tests/check_cpu.py`: 4 groups of ref2 (layers 0, 3), CPU fp64, default path and `MKD_LEAN=1` path | PASS / PASS (8 + 8 files, stats equal) |
| S1 H_in blend | `stats.blend_hin` | `proxy2b/exp/d1108/blend.py hin` | `check_cpu.py`: h08 calib + d1111 hinX at w 0.5 vs d1111 `hi-X0.5` | PASS (96 blended + 1 link) |
| S1 H_out bn | `stats.hout_blocks` | `v3/yaqa/hconv/mkhalf.py` (MKHALF_NORM=1) | `check_cpu.py`: hconv 27B snapshots (15) → train-N / test, vs mkhalf.py run on the same snapshots | PASS (6,400 files, mkhalf.json) |
| S4 swj math | `swj.acc / fit / resid` | `proxy2b/exp/w02-fixedscale/fsutil.py` swiglu_* | `check_cpu.py`: 4 random blocks, 81-point grid | PASS (sums, fit, residuals equal) |
| S4 swj container | `swj.apply_level` | `v3/yaqa/swj27/swjapply.py` | `check_cpu.py`: q-k3base + w36 k3lv scales | PASS (28 files) |
| S3 assembly | `job.assemble` | (new: ourq writes per layer itself) | level layout = ourq's (index, q-LNN, proxy.jsonl); refuses mixed stamps | layout only |

Not ported (called in place, DESIGN.md "Open items" 5): the level passes swjfit27.py (staged bf16 forward + fused
grid), sibmom27.py (sibling moments), mkhin27.py (xcorp H_in), collect.py (Σdy²); kfit2 / knap4; rad-convert.
Not ported because not in the V3 settings: ourq's --sigma-in-sweep, per-linear --hin files (Sketch B), --hout-diag,
--out-scales-off, --head-perm / --head-mean-row (the head job comes with the slab-head decision), the multi-level memo.

## Notes from the port

- **Per-tensor jobs equal the group run.** exllamav3 draws a Hessian group's input sign vector su once, under the
  seed of the group's first tensor; later tensors of the group reuse it. A job for tensor j > 0 replays that draw
  (seed the RNG with tensor 0's seed, finalize the group Hessian) before quantizing with its own seed. Checked on
  in_proj_z, up_proj, k_proj, v_proj (j = 1, 2) in unit A.
- **CPU refit fallback and thread count:** RESULT_A2
- **ourq's --save-wq rowerr** is computed from the weight after quantize_exl3 has regularized it in place (scaled,
  rotated): the per-row error is not in the original basis. Ported unchanged for parity; flagged in DESIGN.md.
