# rxq (`pipeline/`): the V3 quantizer as one pipeline (design, 2026-11-06; merged into rad-exl3-mix 2026-11-07)

Decided 2026-11-06 (user + Code 000): quantizer tooling lives outside radiance; radiance keeps the runtime and eval
primitives (plugin, KLD modes, mix assembly). This repo is the encoder, the post-quantization fits, the knapsack
driver and (later) QAD, as one Python pipeline whose unit of work is a job that rad-quant-at-home can hand out.
The research record stays in rad-exl3-mix; every setting here points at its evidence there.

Status of this document (2026-11-07, w76): S1-S5, the job unit, the level passes and S7 / S8 are implemented in rxq and
checked bit-identical against the research tools (PORTING.md), except the Σdy² collection (collect.py: a two-card GPU
backward, wrapped in place, `rxq/hout.py`). The adopted settings are on in `config/v3.toml`: H_out at K ≥ 4, tile CD
on attn_qg (two-sided with H_out), sub-1 input damping 0.05, swj at every level, i4 + qtf2 for the sub-1 levels, C9 α
on the writers (user 11-07; computed inside the writers' quant jobs). The open one is a switch, off: asym writers
(#105; per mix only, #111). GPU checks that w76 could not run (no GPU) are prepared for the lead: `tools/gpucheck27.sh`, `tools/cdhout27.sh`
(CDHOUT27.md). Settings: `config/v3.toml` (the one file the freeze pins). Where the time goes: PROFILE.md.

## 1. Data flow

```
 bf16 checkpoint ─┐
 calib H_in ──────┼─ [S1 statistics prep] ─ H_in (± xcorp blend) ─┐
 xcorp H_in ──────┘                         H_out (bn diag) ──────┤      once per model
 Σdy² snapshots ──────────────────────────────────────────────────┘
                                                                   │
                          ┌────────────────────────────────────────┘
                          ▼
 per level (K, codebook): [S2 tensor quant] × 400 trunk tensors (+ head) ── per-tensor jobs (donors)
                          ▼
                          [S3 assemble level] ── q-LNN shards (ourq layout)                CPU, minutes
                          ▼
                          [S4 swj level pass]  bf16 forward + MLP hooks → gate/up svh        level pass (big card)
                          ▼
                          [S6 healing]         deferred (#89): frozen codes, scales only      later
                          ▼
 all levels:              [S7 library + knapsack] rxq.knap: levels → knap5 → picks            CPU
                          ▼        └─ hand-off: knap-<tag>-<base>-<target>.mix + frontier line → search stream
 per mix:                 [S5 asym writers]    (if #105 passes; #111: per mix only) sibling moments
                                               of the mix's siblings → denoise → writer requants (S2 jobs)
                          ▼
                          [S5b C9 α]           writers' row factors into svh, on the shipped codes (plain writers: made
                                               in S2's writer jobs, folded with S4's apply; asym writers: per mix)
                          ▼
                          [S8 pack] rad-convert exl3i (mul1i8 / mul1i4 / mul1), radmix mix, census   CPU (conv.lock)
                          ▼
                          swap-pair hygiene    rxq.pairs: two mixes / levels compared refuse to pair if they differ
                                               outside the declared arm
```

S1 runs once per model (and when a statistics setting changes). S2 is the bulk (71 levels × 400 trunk tensors =
28,400 tensor jobs + heads). S3-S4 run once per level. The writer passes run last, in this order (#87-07 / #87-08:
H_out → swj → writers): asym needs the final siblings, so with #111 it runs per mix after the knapsack; C9 folds into
the shipped writers' svh after everything else. S7 / S8 run per release size and variant. `python -m rxq.post plan -K K`
prints the post passes a level gets under the settings.

## 2. Stages

### S1 statistics prep (`rxq/stats.py`, `rxq/hin.py`, `rxq/hout.py`; CPU / level passes)
- **H_in:** the published calib Hessians (packed upper triangle, sum over 10.26M tokens). Optional **xcorp blend**
  (#101, **dropped 2026-11-06** at 27B: pooled ~0.993 vs ≤ 0.98; the switch stays off): per-token means mixed,
  `(1 - w) calib + w xcorp`, stored at calib's scale (`blend_hin`, port of d1108/blend.py).
- **H_in collection** (`rxq/hin.py`, port of `eval/gold27/mkhin27.py` on `rxq/forward.py`'s staging): any corpus's
  input Hessians in the published layout, one card, staged layer passes (~1-2 card-h on the R9700 for 1.14M tokens).
  V3 itself uses the published calib Hessians; this is for blends and other models.
- **H_out:** diagonal Σdy² from the 27B collection (`v3/yaqa/collect.py`, YAQA_DIAG + DIAG_EXTRA "dy"), run in place
  by `rxq/hout.py collect` with hdiag27.sh's production environment (not ported: a two-card GPU backward that cannot
  run on a CPU, so a port could not be checked under the rules; the 27B collection exists). Estimator **bn**
  (#33-27): each 128-sequence block scaled to the same per-tensor total, averaged (`hout_blocks`, port of
  mkhalf.py; `rxq.hout estimate` = every block, the production estimator). #33-28: not adopted for levels ≤ K3.5
  (K2.5 geo 0.9801), **adopted for levels ≥ K4** (K4 geo 0.9643, 11-07): `inputs.hout.apply_k_min = 4`.
- Outputs are input dirs for S2 (`hessians/`, `hout/<id>/`), part of the datapack rad-quant-at-home serves.

### S2 tensor quant (`rxq/quant.py`, `rxq/engine.py`; GPU)
The ladder's quantizer, unchanged: exllamav3 fork eb5a2139 (patch series in rad-exl3-mix quantize/exl3-patches),
kernels swapped in at run time from rad-exl3-mix v3/e3 (exl3frac, exl3cb, exl3frac2 = qtf2), driven as ourq.py drives
it. Per tensor:
1. **load:** bf16 weight (`(in, out)` fp32), the group's H_in (or the asym G from `hin/`), optional `wpre` target
   transform (asym: W F), optional H_out diagonal;
2. **finalize** (shared per Hessian group): H / rows, damping σ·mean diag (0.005 input group, 0.025 rest, #38; the
   input group takes **0.05 below K1**, user 11-07: `quant.apply_sub1_damping`),
   input sign flips su (drawn under the group's first seed) and the 128-Hadamard;
3. **regularize:** output-channel RMS scales (always on, #48), output Hadamard, input-channel scales;
   **scale search** (g_scale: coarse + fine grid of trial trellis quants on sampled tiles, × LDLQ drift, interpolated
   between exllamav3's table points);
4. **scaled LDL:** P diag(su) H diag(su) P per tensor, block-16 LDL (`scaled_hessian`);
5. **LDLQ** with the **trellis search** (qtf2 Viterbi over 2^16 states per 256-weight tile; codebook i8 / i4 / mul1;
   i4 with qtf2 since w53's patch, 11-07); two-sided **ldlq_2hess** when H_out applies (K ≥ 4); **tile CD** (cd3,
   attn_qg, #77; `rxq/tilecd.py`, w50's module merged) after LDLQ, or after ldlq_2hess on the two-sided objective
   tr(Eᵀ H E M): M = H_out in the quantizer's basis, block diagonal with dense 128 × 128 blocks (the output Hadamard
   mixes each 128-column group), so the 8 tiles of a group are decided in turn with exact cross terms; with M = I it is
   the one-sided CD bit for bit (tests/check_tilecd.py);
6. **refit:** su / sv closed-form refit in the H metric (2 rounds);
7. **pack** the trellis, write `t-<key>.safetensors` + its proxy-log record.
Bit-identical to ourq.py per tensor (PORTING.md), including tensors that are not the first of their group (the job
replays the group's su draw).

### S3 assemble (`rxq/job.py assemble`, CPU)
Job outputs of one level → the ourq layout (q-LNN.safetensors per layer, index, config.json, proxy.jsonl,
stamp.json). Refuses missing tensors and mixed settings / code stamps. rad-convert's `exl3i src=DIR` import and every
swap-test tool (swapfit.sh, asym27.sh, deq.py) read this layout. **Note:** stamp.json carries rxq stamps, not
ourq's `args`; asym27's levelargs.py cannot derive ourq arguments from an rxq level (S5 is driven from the
settings file instead).

### S4 swj (`rxq/swj.py`, `rxq/swjfit.py`; level pass)
Joint gate / up output-scale refit through the SwiGLU on frozen codes (#85; **#104 PASS at 27B K2.5**, −6.5%,
confirmed on verify; stacks with H_out; K4 head-matched −1.1%; user 11-07: swj at every level). Needs the bf16 27B's
MLP inputs over ~0.93M qad tokens: a staged forward on one card, all sources (levels) in one pass (~1.5 card-h for 2
levels on the R9700). **Ported** (11-07): the level pass `rxq/swjfit.py` (port of swjfit27.py on `rxq/forward.py`,
level sources decoded by `rxq/deq.py`, the fused Triton grid `rxq/swjtri.py` copied), the grid sums / fit and the
container edit (`swj.apply_level`). Output: the level with gate / up svh × c. Not a donor job (bf16 model + boundaries
~20 GB on disk, a 32 GB card).

### S5 writers asym ρ0 (`rxq/asym.py`, `rxq/sibmom.py`, `rxq/post.py`; per mix: moments pass + per-group + S2 jobs)
(#87; **#105 queued at 27B**, OPEN. **[2026-11-07, #111]** per-level asym writers next to siblings of another K
fail on the 0.8B (8–24% worse than plain); a per-mix writer pass keeps the gain. The settings allow only
`post.asym.scope = "per-mix"` (config.validate refuses per-level); off until #105 and the user's rollout pick.)
Parts: (a) sibling moments H / C / N of the 128 writers' inputs through the mix's own quantized siblings
(`rxq/sibmom.py`, port of sibmom27.py, ~2.5-3 card-h at 27B, 262 GB of moments); (b) per writer group the denoised
metric G and target transform F (`Denoiser`, port of mkdenoise.py: default and lean paths, byte-identical); (c) the
128 writers requantized as S2 jobs with `hin = G`, `wpre = F` (bit-identical to ourq --hin --wpre); (d) spliced
into the mix's levels (`rxq.pairs splice`). `post.plan_asym` lists the steps; the per-mix driver itself is not
written (it waits for #105). Must run after swj and anything else that changes a sibling.

### S5b writers C9 α (in the S2 writer jobs; `rxq/post.py c9` per mix; adopted 11-07)
(#112: 0.8B PASS on both stacks; #112-01 keeps gdn_out α = 3; #112-02 (27B, swj K2.5, fixed codes) geo 0.9925 vs the
rule's ≤ 0.99, verify keeps 0.96 of the gain: PARTIAL, **adopted by the user 11-07**.) Per writer row n,
e_n = (q_n − t_n) M (q_n − t_n)ᵀ / t_n M t_nᵀ, c_n = (1 − e_n)^(−α/2), α = gdn_out 3, attn_o −1, down 0 (t = W,
M = calib H_in; with asym writers t = W F, M = G); port of w75's c9e27.py, scales bit-identical. C9 depends only on
the writer's own codes, W and the static H_in, so for plain writers the quant job computes it right after the
quantization (`post.c9_tensor`: decode the job's own trellis / suh / svh with rxq.deq, the f64 quadratic forms;
`c9-<key>.safetensors` beside the tensor file, its c9.jsonl line in log.jsonl). Assembly writes LEVEL/c9 (c9_scales'
layout); `rxq.post apply LEVEL SWJ,LEVEL/c9 DST` folds swj's and C9's scales into svh in one pass (disjoint tensors:
the same as applying them in turn). swj is fitted on the level without C9, as #112-02 measured it. Cost: the
separate level pass took 859 s a 27B level (w75); in the job it is ~4 · out · in² f64 FLOP a writer (≈ 0.8 TFLOP for a
5120 × 6144 writer; 10-25 s on 4 colo threads, card time measured by `tools/gpucheck27.sh c9`). With asym writers
(#105, per mix) C9 runs after the writer requants: `rxq.post c9 --dn`.

### S6 healing (deferred, #89)
Frozen-code scale healing per level; after the freeze, beside the ladder. Re-run S7 after it.

### S7 library + knapsack + search hand-off (`rxq/knap.py`; CPU)
- **library:** the levels that exist, named as the knapsack names its options (i8-K2.375, i4-K1, m1-K6.5): dir, K,
  codebook, stamps, swj applied, packed donor.
- **solve:** knap5.py (rad-exl3-mix v3/levels/reprice, the 11-07 re-pricing solver: knap4 + the V3R menu) run in
  place from a symlink tree, switches from the settings file's `[knap]` (pricing, outside the stamp) and the
  library (KN_I4 / KN_I8 / KN_M1 = the levels present; KN_SWJ when swj is on; KN_CD only if `[knap].cd_factor` is set).
  With the 11-07 ladder as the library it reproduces w63's main-arm solve byte for byte (tests/check_knap.py).
  kfit2's sensitivity refit on fit-pool swaps is not driven from here yet: the model file is `data/v3/kfit2-model-final.json`.
- **hand-off:** a solve's picks → `knap-<tag>-<base>-<target>.mix` (radmix's format: library packs as donors, act lines
  for a8 modes; head / embedding / V2 options from a donor map) + the frontier line `<name>  bpw X  pred KLD Y`: the
  two inputs search/auto/streamq.py reads. The V3 stream itself is not built.
- Not ported: sub-1 pricing (knap5 KN_SUB1 needs a ratios file: `[knap].sub1_ratios`), A4 options (off, #46-03).

### S8 pack (`rxq/mix.py` text, rad-convert / radmix / census called; CPU)
`rad-convert ... exl3i src=LEVEL codebook=mul1i8|mul1i4|mul1 rate=K`, then radmix into the release mix, then the
census check. rxq writes the recipe and the mix files byte for byte as swap27's tools do (tests/check_mix.py, 13
arms) and the pack script with the lead's 11-07 lock rule (conv.lock + ram.lock exclusive). ~1,600 s per 27B level
on gpubox (PROFILE.md).

### Swap-pair hygiene (`rxq/pairs.py`; CPU)
A swap test compares two models that differ only in its arm. `rxq.pairs levels A B --arm SPEC` (ourq-layout levels,
tensor by tensor) and `rxq.pairs mixes A.rad B.rad --arm SPEC` (packed containers, weight by weight: planes,
encoding, options but the recipe's src=, act mode) refuse a pair that differs outside the arm (families, layers or
keys; `--planes svh` restricts what may differ inside it) or whose arm is identical too. `same` is the no-op gate
(job tensors vs a level), `splice` builds BASE + a job's tensors for exactly the arm. tests/check_pairs.py: 22 cases.

## 3. Job unit

**One job = one tensor (or a few tensors of one layer) at one level.** Spec (JSON, `rxq/job.py`):

```json
{"type": "rxq.quant", "version": 1, "level": "K2.5-i8", "K": 2.5, "codebook": "i8", "seed_offset": 0,
 "keys": ["model.language_model.layers.10.mlp.gate_proj"],
 "settings_stamp": "a557009cc0bcc7ef",
 "inputs": {"ckpt": "weights/", "hessians": "hessians/", "hin": null, "wpre": null, "hout": "hout/bn1920/"}}
```

- **Inputs** (the datapack, by relpath): the tensor's bf16 weight (cut per layer or per tensor), its group's H_in
  file (5120² / 6144² / 17408² packed fp32: 50 / 72 / 606 MB), optional H_out diagonal (KB), optional G / F (asym).
- **Outputs:** `t-<key>.safetensors` (trellis int16 (k/16, n/16, 16K), suh f16, svh f16, codebook marker),
  `log.jsonl` (ourq's proxy record + stage times + peak VRAM), `job.json` (spec, stamps, device, seconds). Writers
  with C9 on (S5b): `c9-<key>.safetensors` (row scales "c", ones "r"; written before the tensor file) and the c9.jsonl
  line in log.jsonl. Optional `wq/` (dequantized f16 weight, ourq's --save-wq) for verification by score
  (rad-quant-at-home VERIFY.md). A donor's C9 file is checkable like its codes: recompute it from the returned tensors.
- **Why per tensor:** the per-layer job's VRAM peak is ffn_down (17408² H plus the refit), and a layer's tensors are
  independent given the group's su draw, which the job replays. Measured peaks (4070, PROFILE.md): every family but
  ffn_down ≤ 3.8 GiB at K2-K6 without H_out; ffn_down 6.7 GiB at a 7.2 GiB cap with the refit on the CPU (needs ~8+ GiB
  on the card); with H_out, gate / up reach 7.1 GiB. So per-tensor jobs let 8 GB cards take ~84% of the tensors
  (everything but ffn_down) without H_out, and the scheduler sends ffn_down (and lm_head, and gate / up with H_out) to
  ≥ 12-16 GB cards. Overhead: the group's H_in is read and finalized once per tensor instead of once per group (q/k/v:
  3×, gate/up and GDN qkv/z: 2×): measured 0.01-0.2 s per tensor with the Hessian in page cache (load) and < 0.05 s
  for the su replay, i.e. < 2% of a layer; on a donor the cost is the download (H_in 50-606 MB per group): lease a
  group's tensors to one worker, or cache Hessians on the worker.
- **Idempotent:** each tensor file is written .part → fsync → rename; a rerun skips finished tensors.
- **Stamps:** `settings_stamp` = sha256 of config sections meta / inputs / quant / post (rxq/config.py); `code_stamp`
  = sha256 of rxq's modules (tilecd.py among them since the merge), exllamav3's quantize.py and the kernel sources. Assembly refuses a level whose
  tensors disagree on either. A donor's result is accepted by score (rad-quant-at-home VERIFY.md: decoded codes'
  proxy error inside the band for its (tensor, K)), so the code stamp is recorded, not enforced, across vendors.
- **Level passes** (S4 swj fit, S5 moments, S1 collections) are not donor jobs: whole-model forwards on a 32 GB card
  with ~20-260 GB of scratch; they run on our cards (gpubox) or trusted big boxes (rad-quant-at-home TRUSTED_ONLY).

**Contract with rad-quant-at-home** (coordinate by files; w47 owns that repo): its `quant_layer` job type runs
`ourq.py --layers L-L` and requires `rowerr.npz`; an `rxq.quant` job type would run `python -m rxq job SPEC -o OUT`
(per tensor, `--save-wq` for the rowerr claim) and accept `t-<key>.safetensors` + `log.jsonl` + `job.json`. Proposed
in `local/workers/w52/rqah-jobtype.md` (rad-exl3-mix), not edited into that repo.

## 4. The settings file (`config/v3.toml`)

Every result-affecting setting, with its evidence (experiment # and status; brought to the 2026-11-07 record by w76,
values changed where the record says so: settings stamp `6ecb3426cdbe356d`, meta.version 2):

| setting | value | evidence | status |
|---|---|---|---|
| codebook by level | i4 K ≤ 1, i8 K1.125-7, mul1 K6.5-8 | #43, #90-02, #94; user 11-07: mul1 K6 / K6.25 dropped | adopted |
| qtf2 | on, i4 included | #37; w53's qtf2-i4.patch applied 11-07 (oracle-exact; `tools/gpucheck27.sh i4` re-checks on card 0) | adopted |
| scaled_hessian | on | ladder setting since 11-01 (no own INDEX row) | adopted |
| damping σ | 0.025; input group 0.005 | #38 (re-checked fractional i8, 11-06) | adopted |
| sub-1 input damping | input group 0.05 below K1 | #38 verdict; user 11-07 (LOG 9:25 AM CST) | **adopted** (on) |
| gain | 1.0 | #39 (flat) | adopted |
| output scales | always | #48 (auto rule +2.5%) | adopted |
| LDLQ drift | interpolated | ourq trunk review 11-02 | adopted |
| tile CD cd3 | on, attn_qg, every K (two-sided with H_out) | #77 (user 11-06), CD27 **passed** at K2.5 (0.9994 / 0.9997); rxq/tilecd.py merged 11-07; with H_out: CPU checks only, 27B K4 pair prepared (CDHOUT27.md) | adopted; K ≥ 4 + H_out pending its pair |
| H_out (bn, 1,920 seqs, σ_out 0.01) | on, K ≥ 4 (`apply_k_min`) | #33-28: K2.5 geo 0.9801 → not adopted ≤ K3.5; K4 geo 0.9643 **PASS** (11-07) | adopted (K ≥ 4) |
| xcorp H_in blend w 0.5 | switch off | #101 (0.8B pass; 27B pooled ~0.993) | dropped (user 11-06) |
| swj | on, every level | #85, #104 PASS (K2.5 −6.5%, verify-confirmed; K4 −1.1% head-matched); user 11-07 | adopted |
| writers asym ρ0 | switch off, per mix only | #87, #105 queued; #111 fail per level (config refuses per-level) | OPEN |
| C9 α (gdn_out 3, attn_o −1, down 0) | on, plain writers, computed in the writer jobs | #112 0.8B PASS; #112-01 α 3; #112-02 27B PARTIAL (geo 0.9925, verify keeps 0.96); user 11-07 | **adopted** |
| #100 hash codebook | switch off (refused) | #100 negative (real 0.8B run) | dropped (user 11-06) |
| healing | off (refused) | #89, deferred | later |
| dropped: i8c, YAQA full, trunk row-slab, act order, chained tail-biting, best-of-N seeds, group scales | — | #78-04, #33, #96, #103, #102, #41, #40 | dropped |

Interactions the config enforces: H_out only between `apply_k_min` and `apply_k_max` (a job naming an H_out dir at a
level where it does not apply is refused, and a level where it applies needs one; a missing per-tensor file is an
error, where ourq skipped the tensor); tile CD up to `quant.tile_cd.k_max`; asym only per mix; healing and #100 are
refused until implemented. `[knap]` (pricing switches for S7) is outside the stamp.

### Ladder rollout (w86, 2026-11-08)
- **Job generation** (`rxq/ladder.py`): levels and build order from `[ladder]` + `[quant.codebook]` (71 levels: pass 1
  = 24 coarse levels, pass 2 = 26, pass 2b = the 14 conditional i4 twins, pass 3 = the 7 sub-1 levels, last); one
  rad-quant-at-home `rxq_quant` job per Hessian group (or tensor), params = level / K / codebook / keys / the settings
  stamp / the settings file's datapack relpath / H_out snapshot / C9 writers / requirements, every input pinned by its
  sha256 from the datapack manifest. Requirements and card-hours from w81's 11-08 R9700 measurements (`python -m
  rxq.ladder req`, `estimate`).
- **H_out form** (`inputs.hout.ldlq`, #115): `blocks` by default since 11-08 (fork patch 0008, now
  quantize/exl3-patches/0008: block-diagonal two-sided LDLQ, same objective, ~37 min an H_out level vs ~190 dense;
  w81's gate PASS) or `dense` (the 11-07 path). `blocks` needs the fork with 0008 (rxq refuses otherwise); `rxq
  levelargs` maps ourq levels made before 11-08 to dense (w81's rxm-hout-blocks.patch, applied).
- **#117 residual block order** (`quant.block_order`, **on**, adopted by the user 11-08): route B (`rxq/blkorder.py`,
  port of w84's blkorder84.py; bit-identical to it, tests/check_blkorder84.py): the 272 residual readers' LDLQ visits
  their input 128-blocks in w82's consensus order; installed after tile CD, so CD runs inside the permutation; works with
  the dense and the block-diagonal two-sided LDLQ. No permuted checkpoint, Hessians or reverse permutation anywhere: the
  codes are stored in natural order with the natural signs (w84 G2: route A == route B bit for bit). `rxq levelargs`
  turns it off for ourq levels (ourq never reorders). `tools/blkorder27.sh` re-makes w84's p0perm readers on card 0.
- **CPU stand-in** (`rxq/cpukern.py`, test only): dry runs of the whole job path without a card (tests above; colo dry run
  on real 27B tensors: rad-exl3-mix `local/workers/w86/`).

## 5. Open items for the lead
1. **Run the prepared GPU checks** (w76 had no GPU): `tools/gpucheck27.sh cd` (tile CD on the real quantizer: rxq ==
   research module, M = I gate, two-sided ratios with H_out at K4; ~15 min card 0), `tools/gpucheck27.sh i4`
   (qtf2i4check + ourq vs rxq at K0.5 i4 + K2 qtf2 vs exl3cb; ~15 min), `tools/gpucheck27.sh c9` (C9 in the quant
   job vs c9e27.py on the card, bit identity, and the per-writer card time; ~5 min), and schedule `tools/insert-cdhout27.sh` (the
   27B K4 pair tile CD + H_out vs H_out on attn_qg, ~2-3.5 h card 0 + ~1.2 h CPU; rule CDHOUT27.md). Until that pair
   passes, K ≥ 4 levels with tile CD rest on CPU checks; on FAIL set `quant.tile_cd.k_max = 3.875`.
2. **iceman's live ourq** (`/root/ourq`): the i4 patch changes ourq.py and exl3frac2.py, so their code stamp changes;
   copy them between a4cost / c9w27 / q2 steps (a resumed level would see a new stamp: OURQ_MIXED=1 or finish it
   first). Not copied by w76 (iceman unreachable 11-07 afternoon, VPN maintenance).
3. **ourq --save-wq rowerr:** ourq computes `rowerr.npz` from `w` after quantize_exl3, which has regularized that
   tensor in place (scaled, rotated, × g_scale): the per-row errors are not in the original basis. Ported as is
   (parity); rad-quant-at-home recomputes its claim from wq with rowerr_tool.py, so it is unaffected.
4. **H_out cost:** ldlq_2hess makes a K ≥ 4 level ~5× slower (~3.2 h vs ~37 min on the R9700; 72% of a gate_proj is
   its dense F-update GEMMs, PROFILE.md T1); the rotated diagonal H_out is block diagonal (128), which allows a ~40-90×
   cheaper update that is the same algorithm in exact arithmetic (autoresearch target). It also raises gate / up's
   peak VRAM to 7.1 GiB. The ladder's K ≥ 4 levels and donor hours need re-pricing.
5. **Not ported:** collect.py (the Σdy² backward: GPU-only, wrapped by rxq/hout.py); the per-mix asym driver (waits for
   #105); kfit2's refit driver (S7 uses the existing model file); sub-1 pricing in the knapsack (`[knap].sub1_ratios`).
6. **Vendoring:** rxq imports the kernels from a rad-exl3-mix checkout (RXQ_RXM) and the exllamav3 fork (PYTHONPATH,
   RXQ_EXL3_SRC); the knapsack solver and the research tools the checks compare against also come from RXQ_RXM. For
   the volunteer release: vendor exl3frac2 / exl3cb / exl3frac into this repo and pin the fork as a patch series.
