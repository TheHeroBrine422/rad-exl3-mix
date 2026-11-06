# swj27: C10 "swj" (joint gate / up scale refit) on existing 27B levels

**Question.** Does swj transfer to Qwen3.8-27B? On the 0.8B, swj (#85, `proxy2b/exp/w02-fixedscale`) refits the
output scales of gate and up through the SwiGLU on frozen codes: K3 −3.1%, K2 −8.6% (k{3,2}base), and on edy H_out K3
0.969 / 0.972 / 0.970, K2 0.910 / 0.914 / 0.912 (#85-03). Only svh changes, so the 27B test needs **no requantization**:
fit the scales of the existing K2.5 swap arms (`base`, `s1`; optionally `bn`, `bns1`), write them into the level's
shards, pack, mix, score, pair. Pass rule, commands and costs: **`v3/yaqa/SWJ27.md`**.

**Runtime cost: none.** Same format, same kernels, same bpw: only the f16 svh vectors of gate / up change.

## Pipeline

1. **Slice** (`mkmeta.py`): a qad-pool slice from `mix/calib_slice.py` (pools-v3, category-proportional, cap 4096),
   tokens only (no logits needed).
2. **Fit** (`swjfit27.py`): the bf16 27B forward in staged layer passes on one card (`eval/gold27/mkgold27.py` /
   `mkhin27.py`'s staging: transformers' own decoder layers, one document at a time with a fresh cache, bf16
   boundaries on disk). A forward pre-hook on each MLP computes swiglufit.py's sums for every source in the same pass:
   - x = the MLP input; a = silu(W_g x)(W_u x) in f32 from the bf16 weights; ĝ, û from the source;
   - per channel and grid point c (0.6..1.4 by 0.01): Σ a p, Σ p², p = silu(c ĝ) û; and Σ a²; f64 per even / odd
     document half;
   - c_g = the grid argmin, c_u in closed form, clamped to [0.5, 2] (fsutil.py's `swiglu_fit`, unchanged).
   Sources: `DIR:wq` (ourq `--save-wq` files: swiglufit.py's input) or `DIR:level` (the shards, decoded by
   `exl3deq.py` in f32: the 27B levels have no wq files). Output per source: `scales/<key>.safetensors` {r = 1, c},
   `swiglu.jsonl` (in-sample and held-out residuals per layer, as swiglufit.py), `swjfit.json`.
3. **Container form** (`swjapply.py`): the level with svh ×= c for every gate / up, rounded to f16 as stored.
   Everything else is byte for byte the level's: codes, suh, other tensors, stamp.json, index.
4. **Swap arm** (`swapfit.sh`): the edited level packed as `exl3i src=… codebook=mul1i8 rate=2.5`, mixed into V2 3.00,
   census, scored on `golden-search27`. These are `swap27k.sh`'s pack / mix and `swap27.sh`'s score: same home,
   build and golden. Then paired against the unedited arm (`tools/paired.py`) and `rule.py`'s verdict.

## Files

| file | what |
|---|---|
| `swjfit27.py` | the staged fit (any number of sources in one pass; `--fused`; resumable: checkpoint generations, per-stage sums) |
| `swjtri.py` | `--fused`: the grid sums in one Triton kernel, plus `check()` against the torch path |
| `exl3deq.py` | torch EXL3 decoder: fractional K (qtf2), the 0-bit salt, mul1 / i8 / i8c / i4. A vectorised `volunteer/checker/exl3dec.py`; `v3/sib27/deq.py` is integer-K only |
| `swjapply.py` | scales → the level's svh / suh (container form) |
| `mkmeta.py` | tokens-only META (kld.json docs + tokens.i32) from a golden + calib_slice args; the golden's header is kept for `ledger_add.py` |
| `cmpswj.py` | compares two swj outputs (scales bit for bit, grid flips, jsonl) |
| `swapfit.sh` | gpubox driver: meta, plan, fit, apply, arms, rule |
| `rule.py` | SWJ27.md's verdict from the pair lines |
| `swjbench.py` | grid-sum timing at the 27B shape, torch vs fused |
| `gate08.sh`, `gate08f.sh` | the 0.8B gates (4070) |

## Gate on the 0.8B (4070, 2026-11-05 from 9:41 PM CST; raw: `results/`)

`gate08.sh`: g08-fit1 (87 fit-pool docs, 100,625 tokens), 3 stages `[0, 8, 16, 24]`, three sources in one pass:

| check | result |
|---|---|
| `k3wq` (q-k3base `--save-wq`, the 27B path in wq mode) vs w02's `q-k3base_swj` (11-03) | **bit-identical**: 48 scale files, swiglu.jsonl |
| its evalq on g08-search1 vs w02's `k3base_swj.f32` | **byte-identical** |
| `k3lv` (same level, gate / up decoded from the shards in f32) vs `k3wq` | c_g: 2.07% of channels move by one grid step (max 0.04); c_u: mean \|Δ\| 3.6e-4 (max 0.039). The decoded W and `--save-wq` differ by the f16 rounding of suh / svh and storage (3.4e-4–4.0e-4 relative, `exl3deq.py check`) |
| `k15lv` (K1.5 i8 level, fractional decode) vs d1106's `q-k1.5edy_swj` (from the deleted wq) | the same size: c_g 2.15% one-step moves, c_u mean \|Δ\| 3.2e-4 |
| container path: swjapply (q-k3base + k3lv's scales) → decode all 150 tensors → evalq | vs `k3base_swj` (wq path) **0.99986** (z −0.36); control: decoded unedited level vs `k3base` 0.99997 (z −0.08) |
| the swj gain via the container path | deq-k3 → deq-k3swj **0.9692** (z −21.9) = k3base → k3base_swj 0.9693 (z −22.4) |
| edited shards = level × c | `exl3deq.py check` vs k3lv's wq: worst 3.1e-4 relative (f16 svh), PASS |

Decoder (`exl3deq.py`, 4070 CPU): W_hat bit-exact against `exl3dec.py` on 12 tensors (K3 and K1.5, i8). W matches
the `--save-wq` files to ≤ 4.0e-4 relative on 12 tensors.

Resume (CPU, 6 docs × 256 tokens, 2 sources): runs killed mid-stage (checkpoint every document) and rerun give
outputs identical to an uninterrupted run.

Resume and `--no-cache`, same CPU setup: the use_cache=False forward (mkhin27's, the default on gpubox) gives scales and
jsonl identical to the fresh-cache forward (mkgold27's, the one the bit-exact gate above used).

`gate08f.sh` (`--fused`, swjtri.py):

| check | result |
|---|---|
| `swjbench.py` at the 27B shape (J 17,408, 2,048 positions, 81 grid points), 4070 | torch 203 µs, fused **6.4 µs** per position, layer and source (31.7×); max relative difference of the sums 2.3e-7 |
| built-in check on the first block of each stage (0.8B) | 3.6e-7, 3.6e-7, 4.3e-7 (abort threshold 1e-5) |
| fused fit vs exact (k3wq) | c_g: 0.48% of channels one grid step apart (near-ties); c_u mean \|Δ\| 5e-5 |
| KL, fused-fit arm vs exact | **0.99980** (z −0.59): indistinguishable |
| 0.8B pass speed | 10.8–24k tok/s with 1 source (forward-bound) vs 1.25k tok/s exact with 3 sources |

GPU use on the 4070: ~9 min in three holds of `/data/yb/gpu.lock` (VRAM ≤ 1.4 GiB).

## Not covered here

- Nothing ran on gpubox's GPU (the lead queues it). On gpubox only the CPU steps ran (`meta`, `plan`, in
  `/data/q27/w36/t`): the qad slice and the stage plan for 3 sources.
- ROCm specifics: the Triton kernel on gfx1201 (fla's Triton kernels run there), f32 GEMM speed. The fused path checks
  itself on its first block and aborts above `--fused-tol` (1e-5).
