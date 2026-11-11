# Where the quantizer's time goes

## 0. Update 2026-11-11: after T1 (block-diagonal H_out), on the R9700 (card 0), today's stack

Source: rxq job logs of full 400-tensor levels run through the ladder's job path (w92 fs0 / fs1 K2.5, w95 K0.125 / K1,
w99 K6; settings ed97adb7 / 80b5e6d7: H_out blocks, #117 order, tile CD, C9, i8). `tools/profagg.py`, raw
`data/profile/post-t1-2026-11-11.txt`. Sections 1-3 below are the 11-06 profile (66 levels, mul1, dense H_out, asym):
kept for the stage method; their ladder totals are superseded.

| level | Σ tensor wall | job wall (256 jobs, serial) | trellis | prep | refit | ldlq + H_out | load / other |
|---|---|---|---|---|---|---|---|
| K0.125 | 3.26 h | 3.53 h | **90%** | 3% | 3% | 2% | 2% |
| K1 | 1.04 h | 1.32 h | 68% | 10% | 9% | 7% | 5% |
| K2.5 | 0.84 h | 1.14 h | 61% | 12% | 12% | 9% | 7% |
| K6 | 0.78 h | — | 57% | 14% | 12% | 9% | 7% |

Per tensor at K2.5 (s): attn_qg 20.1, ffn_down 13.7, ffn_gate 8.8, ffn_up 7.4, ssm_qkv 5.8, ssm_out 5.5, attn_output 5.5,
ssm_z 2.9, attn_k / attn_v 0.9. At K0.125 every family is ~4.5× that (attn_qg 92 s).

- **T1 delivered:** H_out's per-tensor cost fell from 178 s / layer (dense) to ~8% of the time; a K2.5 level with the
  full stack is ~1.1 card-h of jobs (was ~3.2 h).
- **The trellis search is again the quantizer** (57-68% at K ≥ 1, 90% at K0.125). K6 is no longer ~2× K4 (mul1 gone,
  i8): it costs about what K2.5 does. **Sub-1 levels are the expensive ones** (K0.125 ≈ 3× a K2.5 level).
- **~25% of job wall is outside the tensors' own timers** (job wall 4,111 s vs Σ tensor 3,030 s at K2.5): process
  start, extension init, group Hessian load, save. Persistent workers / concurrency (w105) target this and the idle
  card time inside small tensors (attn_k / v ~1 s).
- **Ladder estimate (50 levels, swj ≤ K4.5):** ~8 sub-1 levels × ~1.3-3.5 h + ~42 levels × ~1.1-1.3 h ≈ 70-80 card-h of
  jobs, plus swj passes (~0.75 h × the levels ≤ K4.5), on card 0 alone ≈ 4 days serial before concurrency gains.
- **Levers now:** (1) concurrency / persistent workers (w105, measuring); (2) T2 qtf2 kernel speed (bit-identical-codes
  judge, so it can land mid-ladder without a restamp); decide T2 after w105's numbers. Pack (T4) stays off the card's
  critical path if it overlaps the next level's quant.

## 1-3. The 11-06 profile (4070 stage shares, pre-T1 R9700 totals)


Sources: (1) **4070 stage profile** — rxq's per-tensor jobs with stage timers (`rxq/prof.py`: device synchronized at
every stage boundary, exclusive seconds), 27B layer 10 (GDN + MLP) and layer 39 (attention), ladder settings (i8,
qtf2, scaled H, σ 0.025 / 0.005), RTX 4070 capped at 7.2 GiB (`tools/run4070.sh`, raw: `data/profile/`);
(2) **R9700 per-tensor totals** from gpubox's swap-test quant logs (ourq's per-tensor seconds, the same code and
settings, 64 layers; `/workspace/backup/gpubox/data/swap27/quant-*.log`); (3) gpubox pack / score logs.
The ladder count used for ranking: **66 levels × 400 trunk tensors** (v3.md "Ladder"; heads separate), i.e. per
level 64 × {gate, up, down}, 48 × {ssm_qkv, ssm_z, ssm_out}, 16 × {q, k, v, o}.

## 1. Per tensor, by stage (4070, seconds)

Wall seconds per tensor (one tensor per family; layer 10 GDN + MLP, layer 39 attention) and the shares of the
three stages that matter. ffn_down at K4 / K6 was skipped (its refit falls back to the CPU on this card: lock budget).

| family | wall K2 | K2.5 | K4 | K6 | trellis share (K2 / 2.5 / 4 / 6) | scale search | refit | peak GiB (K2.5) |
|---|---|---|---|---|---|---|---|---|
| ssm_qkv | 5.9 | 7.1 | 5.6 | 11.7 | 76% / 79% / 75% / 84% | 7 / 7 / 7 / 7% | 7 / 6 / 7 / 4% | 2.98 |
| ssm_z | 3.5 | 4.3 | 3.3 | 7.0 | 77% / 80% / 77% / 84% | 7 / 7 / 7 / 7% | 8 / 6 / 8 / 4% | 2.31 |
| ssm_out | 3.8 | 4.4 | 3.7 | 7.5 | 73% / 78% / 72% / 82% | 6 / 7 / 6 / 7% | 8 / 7 / 9 / 4% | 2.44 |
| attn_qg | 6.8 | 8.5 | 6.3 | 13.9 | 79% / 80% / 80% / 85% | 7 / 7 / 7 / 7% | 7 / 6 / 7 / 3% | 3.31 |
| attn_k | 1.0 | 1.2 | 1.0 | 2.0 | 52% / 54% / 53% / 59% | 18 / 19 / 18 / 21% | 8 / 7 / 8 / 4% | 1.62 |
| attn_v | 1.0 | 1.2 | 1.0 | 2.1 | 53% / 54% / 53% / 59% | 19 / 20 / 18 / 21% | 8 / 7 / 8 / 4% | 1.62 |
| attn_output | 3.9 | 4.5 | 3.6 | 7.5 | 73% / 75% / 73% / 82% | 6 / 6 / 6 / 7% | 8 / 7 / 9 / 4% | 2.44 |
| ffn_gate | 9.7 | 11.7 | 9.2 | 19.6 | 80% / 82% / 79% / 86% | 7 / 7 / 7 / 7% | 7 / 6 / 7 / 4% | 3.77 |
| ffn_up | 9.6 | 11.7 | 9.1 | 19.7 | 81% / 83% / 80% / 86% | 7 / 7 / 7 / 7% | 7 / 6 / 8 / 4% | 3.77 |
| ffn_down | 102.7 | 103.4 | – | – | 8% / 10% / – / – | 1 / 1% | **88 / 87%** (CPU fallback) | 6.67 |

All stages, summed over the nine tensors other than ffn_down:

| K | wall s | load | prep | scale search | ldl | ldlq GEMMs | **trellis** | refit | rest |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 45.2 | 2.3% | 0.6% | 7.3% | 0.7% | 1.5% | **76.8%** | 7.5% | 3.3% |
| 2.5 | 54.5 | 1.5% | 0.6% | 7.4% | 0.7% | 1.3% | **79.2%** | 6.1% | 3.2% |
| 4 | 42.7 | 1.5% | 0.7% | 7.2% | 0.8% | 1.6% | **76.3%** | 7.8% | 4.2% |
| 6 | 90.9 | 0.2% | 0.3% | 7.9% | 0.5% | 0.8% | **83.7%** | 3.7% | 2.8% |

**With H_out** (two-sided ldlq_2hess, bn diagonal, K2.5; unit D):

| family | wall s | ldlq (F-update GEMMs) | trellis | other stages | peak GiB |
|---|---|---|---|---|---|
| ssm_z | 10.6 (4.3 without) | 5.44 (51%) | 3.59 | 1.6 | 2.52 |
| attn_output | 10.3 (4.5) | 5.45 (53%) | 3.64 | 1.2 | 2.44 |
| ffn_gate | 49.2 (11.7) | **35.2 (72%)** | 9.94 | 4.1 | **7.10** |

Stage groups: load = H_in + weight (+ H_out) read and unpack, and a per-tensor job's replay of its group's su draw;
prep = finalize (H scaling, damping, sign + Hadamard rotation) + regularize (channel scales, Hadamards);
scale = the global-scale search (g_scale: 10 coarse + 5 fine trial trellis quants on ~3 sampled tiles per
diagonal, × drift); ldl = scaled_ldl (unrotate H, per-tensor block LDL) (+ prepare_H_out); ldlq = LDLQ's error-feedback
GEMMs (ldlq_2hess's with H_out); trellis = the Viterbi search of every tile inside LDLQ (qtf2); refit = su / sv refit;
rest = proxy metrics, back-rotation, pack, save.

- **The trellis search is the quantizer.** Without H_out, 76-84% of every tensor's time is the Viterbi search inside
  LDLQ, and another ~7% is the scale search's trial quantizations (the same kernel): **~85% is one kernel**. Everything
  else (Hessian load, rotation, block LDL, LDLQ's feedback GEMMs, metrics, pack, save) is < 10% together.
- **K dependence:** flat from K2 to K4 (K4 is even ~5% faster than K2), fractional K2.5 is ~1.2× K2 (the qtf2
  fractional kernel), **K6 is ~2.1× K4**. The mul1 K6-8 levels and the i8 K5.25-7 levels are the expensive ones.
- **H_out's cost is the F update, not the trellis:** ldlq_2hess's own GEMMs are 72% of gate_proj (35 s), its trellis
  time is unchanged (9.9 vs 9.65 s one-sided). The small anti-diagonal batches do not slow the trellis kernel on this
  card. Peak VRAM rises to 7.1 GiB (gate / up with H_out: no longer an 8 GB-card job).
- **Small cards:** every family except ffn_down peaks at ≤ 3.8 GiB (K2.5), ≤ 3.1 GiB at K4 / K6; ffn_down needs the
  refit's ~7+ GiB or the CPU fallback (88% of its time here: 90 s at 4 threads, 42 s at torch's default thread count).
- **Profiler overhead:** the per-trellis-call device sync costs ~0.3-1.5 s per tensor; the profiled K2.5 totals are
  14-22% above hconv's unprofiled ourq times of the same tensors on the same card (which also exclude load / save).

## 2. Per tensor on the R9700 (gpubox ourq logs, mean seconds per tensor; same code and settings)

| family | K2.5 (s1) | K2.5 + H_out (bn) | K4 (base4) | K4 + H_out (bn4) |
|---|---|---|---|---|
| ssm_qkv | 4.7 | 19.2 | 4.0 | 19.9 |
| ssm_z | 2.6 | 8.5 | 2.2 | 8.5 |
| ssm_out | 2.8 | 8.5 | 2.4 | 8.4 |
| attn_qg | 5.1 | 26.1 | 4.3 | 25.9 |
| attn_k | 0.8 | 1.0 | 0.7 | 1.0 |
| attn_v | 0.8 | 1.0 | 0.7 | 0.9 |
| attn_output | 2.8 | 8.5 | 2.4 | 8.4 |
| ffn_gate | 6.9 | 47.5 | 5.8 | 47.6 |
| ffn_up | 6.9 | 47.6 | 5.8 | 47.4 |
| ffn_down | 10.9 | 46.5 | 9.7 | 47.7 |
| **per layer** | **34.7** | **177.8** | **29.7** | **179.4** |
| **per level** (64 layers) | **37 min** | **190 min** | **32 min** | **191 min** |

The 4070 runs the same tensors ~1.6× slower than the R9700 (gate 11.7 vs 6.9 s; the stage shares carry over: the
trellis search is the same kernel). Per level on gpubox: **pack ~1,600 s** (rad-convert exl3i, CPU, conv.lock;
1,551-1,706 s over 7 arms), **score ~150 s** (golden-search27). A ladder level without H_out is thus ~37 min of card
plus ~27 min of CPU pack; with H_out ~190 min of card.

## 3. Optimization targets, ranked by seconds saved × ladder jobs

| # | target | now | after (est.) | saved over the ladder | safe for autoresearch? (judge) |
|---|---|---|---|---|---|
| T1 | **ldlq_2hess with a diagonal H_out**: the rotated diagonal H_out is block-diagonal (128 × 128 blocks after the output Hadamard), so its LDL factor L_out is too. The current anti-diagonal sweep multiplies the full dense L_out (`bmm(dE, Ln[cols][:, :, :n_hi])`, then `F[:k_hi, :n_hi] += Lk^T · right`): ~620 TFLOP per 27B gate / up / down, 330 per q_proj. Restricting each tile's update to its own 128-column block: 7-16 TFLOP (40-90× less); a block-wavefront order (a tile depends only on later rows and later columns of its own 128-block) also gives bigger trellis batches (≤ 1,407 → ~330 steps) | 178 s / layer (R9700, H_out) | ~40-45 s / layer | **only if H_out is adopted**: ~135 s × 64 layers × 66 levels ≈ **158 card-h** (of ~210) | yes: same algorithm in exact arithmetic, float order differs → codes differ; judge = two-sided proxy error ≤ reference (per tensor, within seed spread) + time |
| T2 | **trellis search kernel (qtf2 Viterbi)**: 76-84% of every tensor without H_out (4070: gate 9.65 of 11.7 s at K2.5), +7% more in the scale search's trial quants (same kernel); K6 costs ~2.1× K4 | ~32 card-min / level (R9700) | 1.3× kernel → ~25 | ~9 card-min × 66 ≈ **10 card-h** (1.3×); the same share on donors' cards | yes, the cleanest: judge = **identical codes** (bit-exact, as qtf2check.py) + time; qtf2 already had a 10-version speed pass (#37), so expect diminishing returns |
| T3 | **ffn_down's refit OOM on ≤ 12 GB cards**: the refit holds H (17408², 1.2 GB), its damped copy H_err (1.2 GB), (Q Qᵀ) ∘ H (2 × 1.2 GB) and H_orig: falls back to the CPU (90 s at 4 threads, 42 s at the default count, 4070) | 42-90 s per ffn_down on small cards (the other families' GPU refit: 0.1-0.7 s) | ~1-2 s (chunked A = (QQᵀ)∘H, no H_err clone: the damping correction is tr(EᵀE)·δ) | donor time: 64 ffn_down × 66 levels × ~90 s ≈ **105 donor-card-h** if every ffn_down goes to a ≤ 12 GB card; 0 if the scheduler sends ffn_down to ≥ 16 GB cards only | yes: codes identical (refit is after the codes); judge = identical trellis, proxy error ≤ reference, VRAM peak |
| T4 | **pack (rad-convert exl3i)**: CPU, one at a time under conv.lock | ~1,600 s / level | parallel tensors / threads, maybe ~5× | ~22 h of serialized CPU wall time over 66 levels (critical path of the swap / ladder pipeline, not card time) | yes: judge = **byte-identical container** + time (outside this repo: radiance's rad-convert) |
| T5 | **scale search** (trial trellis quants on sampled tiles): 7-8% per tensor (18-21% on the small attn_k / attn_v) | ~2.5 card-min / level | warm start from the neighbouring level's g_scale + fine grid only (~2.5× fewer trials) | ~1.5 card-min × 66 ≈ **1.6 card-h** | partly: changes g_scale → codes change; judge = proxy error ≤ reference |
| T6 | **H_in load / unpack, finalize, scaled LDL, LDLQ GEMMs, rest**: 3-5% together on the 4070 (Hessians in page cache) | ~1.5 card-min / level | — | < 1 card-h | not worth a task; for donors the H_in **download** matters more (per-tensor jobs: cache a group's Hessian across its 2-3 tensor jobs on one worker) |
| T7 | **level passes** (not per-tensor): swj fit ~0.75 card-h per level (1.5 h for two levels in one pass, SWJ27.md) → ≤ K3.5 levels only (~27 of 66): ~20 card-h; **asym moments ~2.5-3 card-h per level** (ASYM27.md) → ~170-200 card-h if adopted for all 66 levels | | more sources per swj pass (VRAM-bound); for asym: test whether one level's moments serve its neighbours (ASYM27.md) | up to ~150 card-h (asym, if adopted and neighbours share moments) | no (changes results; a design decision, not a speed task) |

**Judges for the autoresearch tasks** (written before any attempt):
- *Identical codes* (T2, T3, T4): every `<key>.trellis` torch.equal to the reference job's on the 10 profile tensors at
  K2 / K2.5 / K4 / K6 (`tests/check_quant.py`), and for T3 `suh` / `svh` within 1 f16 ulp (the CPU / GPU refit
  difference); time = the tensors' summed wall time from `log.jsonl`, same card, same lock-free conditions.
- *Proxy error ≤ reference* (T1, T5): the codes may change. For T1 the score must be the **two-sided** objective
  tr(Eᵀ H_in E H_out) / tr(Wᵀ H_in W H_out) = Σ_n h_n e_nᵀ H_in e_n / Σ_n h_n w_nᵀ H_in w_n (diagonal H_out h, original
  basis, E = W − the decoded weight), not the logged `proxy_err`: with H_out on, the log holds the refit's one-sided
  error. Compute it from the decoded weight (`--save-wq`'s wq, or the trellis decoded as in rad-quant-at-home's
  rowerr_tool.py; NOT ourq's rowerr.npz, see PORTING.md). Pass = per tensor ≤ reference × 1.005 (the seed spread is
  ~0.5%, #41) and the geometric mean over the tensors ≤ 1.000, on ≥ 2 rotation seeds.

Notes:
- T1 is the largest single item but conditional on H_out (#33-28, open). If H_out is not adopted, T2 (the kernel) and
  T3 (small-card donors) are what matters; with asym adopted, T7's moment sharing dominates everything.
- The 4070's ffn_down numbers are dominated by T3 and are not representative of the R9700 (whose refit runs on the
  card): use the R9700 table for card-hours, the 4070 table for stage shares.
- Profiling overhead: the timers synchronize the device at every stage and every trellis call; the K2.5 per-tensor
  totals with the profiler on are 14-22% above hconv's unprofiled ourq run of the same tensors on the same card (an
  upper bound: ourq's per-tensor seconds exclude load and save). Shares are slightly biased toward stages with many
  short calls (the trellis: ~320-1,088 calls per tensor).
