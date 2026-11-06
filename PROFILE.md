# Where the quantizer's time goes (2026-11-06)

Sources: (1) **4070 stage profile** — rxq's per-tensor jobs with stage timers (`rxq/prof.py`: device synchronized at
every stage boundary, exclusive seconds), 27B layer 10 (GDN + MLP) and layer 39 (attention), ladder settings (i8,
qtf2, scaled H, σ 0.025 / 0.005), RTX 4070 capped at 7.2 GiB (`tools/run4070.sh`, raw: `data/profile/`);
(2) **R9700 per-tensor totals** from gpubox's swap-test quant logs (ourq's per-tensor seconds, the same code and
settings, 64 layers; `/workspace/backup/gpubox/data/swap27/quant-*.log`); (3) gpubox pack / score logs.
The ladder count used for ranking: **66 levels × 400 trunk tensors** (v3.md "Ladder"; heads separate), i.e. per
level 64 × {gate, up, down}, 48 × {ssm_qkv, ssm_z, ssm_out}, 16 × {q, k, v, o}.

## 1. Per tensor, by stage (4070, seconds)

PROFILE_TABLE_4070

Stage groups: load = H_in + weight (+ H_out) read and unpack, and a per-tensor job's replay of its group's su draw;
prep = finalize (H scaling, damping, sign + Hadamard rotation) + regularize (channel scales, Hadamards);
scale = the global-scale search (g_scale: 10 coarse + 5 fine trial trellis quants on ~3 sampled tiles per
diagonal, × drift); ldl = scaled_ldl (unrotate H, per-tensor block LDL) (+ prepare_H_out); ldlq = LDLQ's error-feedback
GEMMs (ldlq_2hess's with H_out); trellis = the Viterbi search of every tile inside LDLQ (qtf2); refit = su / sv refit;
rest = proxy metrics, back-rotation, pack, save.

PROFILE_FINDINGS_4070

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
| T2 | **trellis search kernel (qtf2 Viterbi)**: 80-90% of every tensor without H_out (4070 profile: gate 9.65 of 11.7 s; ssm_qkv 5.6 of 7.1 s); +7% more in the scale search's trial quants | ~32 card-min / level (R9700) | 1.3× kernel → ~25 | ~9 card-min × 66 ≈ **10 card-h** (1.3×); the same share on donors' cards | yes, the cleanest: judge = **identical codes** (bit-exact, as qtf2check.py) + time; qtf2 already had a 10-version speed pass (#37), so expect diminishing returns |
| T3 | **ffn_down's refit OOM on ≤ 12 GB cards**: the refit holds H (17408², 1.2 GB), its damped copy H_err (1.2 GB), (Q Qᵀ) ∘ H (2 × 1.2 GB) and H_orig: falls back to the CPU (90 s at 4 threads, 4070) | 90-103 s per ffn_down on small cards (vs ~1 s on GPU) | ~1-2 s (chunked A = (QQᵀ)∘H, no H_err clone: the damping correction is tr(EᵀE)·δ) | donor time: 64 ffn_down × 66 levels × ~90 s ≈ **105 donor-card-h** if every ffn_down goes to a ≤ 12 GB card; 0 if the scheduler sends ffn_down to ≥ 16 GB cards only | yes: codes identical (refit is after the codes); judge = identical trellis, proxy error ≤ reference, VRAM peak |
| T4 | **pack (rad-convert exl3i)**: CPU, one at a time under conv.lock | ~1,600 s / level | parallel tensors / threads, maybe ~5× | ~22 h of serialized CPU wall time over 66 levels (critical path of the swap / ladder pipeline, not card time) | yes: judge = **byte-identical container** + time (outside this repo: radiance's rad-convert) |
| T5 | **scale search** (trial trellis quants on sampled tiles): 5-8% per tensor | ~2.5 card-min / level | warm start from the neighbouring level's g_scale + fine grid only (~2.5× fewer trials) | ~1.5 card-min × 66 ≈ **1.6 card-h** | partly: changes g_scale → codes change; judge = proxy error ≤ reference |
| T6 | **H_in load / unpack, finalize, scaled LDL, LDLQ GEMMs, rest**: 3-5% together on the 4070 (Hessians in page cache) | ~1.5 card-min / level | — | < 1 card-h | not worth a task; for donors the H_in **download** matters more (per-tensor jobs: cache a group's Hessian across its 2-3 tensor jobs on one worker) |
| T7 | **level passes** (not per-tensor): swj fit ~0.75 card-h per level (1.5 h for two levels in one pass, SWJ27.md) → ≤ K3.5 levels only (~27 of 66): ~20 card-h; **asym moments ~2.5-3 card-h per level** (ASYM27.md) → ~170-200 card-h if adopted for all 66 levels | | more sources per swj pass (VRAM-bound); for asym: test whether one level's moments serve its neighbours (ASYM27.md) | up to ~150 card-h (asym, if adopted and neighbours share moments) | no (changes results; a design decision, not a speed task) |

Notes:
- T1 is the largest single item but conditional on H_out (#33-28, open). If H_out is not adopted, T2 (the kernel) and
  T3 (small-card donors) are what matters; with asym adopted, T7's moment sharing dominates everything.
- The 4070's ffn_down numbers are dominated by T3 and are not representative of the R9700 (whose refit runs on the
  card): use the R9700 table for card-hours, the 4070 table for stage shares.
- Profiling overhead: the timers synchronize the device at every stage and every trellis call; the K2.5 per-tensor
  totals with the profiler on are within RESULT_OVERHEAD of hconv's unprofiled ourq run of the same tensors on the same card.
