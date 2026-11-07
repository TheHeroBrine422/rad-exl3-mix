# rad-exl3-quantizer: the V3 quantizer as one pipeline (design, 2026-11-06)

Decided 2026-11-06 (user + Code 000): quantizer tooling lives outside radiance; radiance keeps the runtime and eval
primitives (plugin, KLD modes, mix assembly). This repo is the encoder, the post-quantization fits, the knapsack
driver and (later) QAD, as one Python pipeline whose unit of work is a job that rad-quant-at-home can hand out.
The research record stays in rad-exl3-mix; every setting here points at its evidence there.

Status of this document: the per-tensor and per-group parts of S1-S5 and the job unit are implemented and checked
bit-identical against the research scripts (PORTING.md); the level passes (swj's bf16 forward, the sibling moments,
the H_in / H_out collections) and S6-S8 are designed here and run through the existing rad-exl3-mix tools until
ported. Settings: `config/v3.toml` (the one file the freeze pins). Where the time goes: PROFILE.md.

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
                          [S5 asym writers]    sibling moments (level pass) → mkdenoise
                                               (per group) → 128 writer requants (S2 jobs)   level pass + jobs
                          ▼
                          [S6 healing]         deferred (#89): frozen codes, scales only      later
                          ▼
 all levels:              [S7 census + knapsack re-run] kfit2 refit on fit-pool swaps, knap4   CPU + card evals
                          ▼
                          [S8 pack] rad-convert exl3i (mul1i8 / i4 / mul1), radmix mix, census   CPU (conv.lock)
```

S1 runs once per model (and when a statistics setting changes). S2 is the bulk (66 levels × 400 trunk tensors =
26,400 tensor jobs + heads). S3-S5 run once per level, in this order (#87-07 / #87-08: H_out → swj → writers last,
on the level's final siblings). S7 / S8 run per release size and variant.

## 2. Stages

### S1 statistics prep (`rxq/stats.py`, CPU)
- **H_in:** the published calib Hessians (packed upper triangle, sum over 10.26M tokens). Optional **xcorp blend**
  (#101, OPEN): per-token means mixed, `(1 - w) calib + w xcorp`, stored at calib's scale (`blend_hin`, port of
  d1108/blend.py). The xcorp Hessians come from `eval/gold27/mkhin27.py` over the mX meta (a level-independent pass,
  ~1 card-hour; not ported).
- **H_out:** diagonal Σdy² from the 27B collection (`v3/yaqa/collect.py`, YAQA_DIAG; not ported: a backward pass on two
  cards). Estimator **bn** (#33-27): each 128-sequence block scaled to the same per-tensor total, averaged
  (`hout_blocks`, port of mkhalf.py; `split=False` = every block, the production estimator). OPEN (#33-28).
- Outputs are input dirs for S2 (`hessians/`, `hout/<id>/`), part of the datapack rad-quant-at-home serves.

### S2 tensor quant (`rxq/quant.py`, `rxq/engine.py`; GPU)
The ladder's quantizer, unchanged: exllamav3 fork eb5a2139 (patch series in rad-exl3-mix quantize/exl3-patches),
kernels swapped in at run time from rad-exl3-mix v3/e3 (exl3frac, exl3cb, exl3frac2 = qtf2), driven as ourq.py drives
it. Per tensor:
1. **load:** bf16 weight (`(in, out)` fp32), the group's H_in (or the asym G from `hin/`), optional `wpre` target
   transform (asym: W F), optional H_out diagonal;
2. **finalize** (shared per Hessian group): H / rows, damping σ·mean diag (0.005 input group, 0.025 rest, #38),
   input sign flips su (drawn under the group's first seed) and the 128-Hadamard;
3. **regularize:** output-channel RMS scales (always on, #48), output Hadamard, input-channel scales;
   **scale search** (g_scale: coarse + fine grid of trial trellis quants on sampled tiles, × LDLQ drift, interpolated
   between exllamav3's table points);
4. **scaled LDL:** P diag(su) H diag(su) P per tensor, block-16 LDL (`scaled_hessian`);
5. **LDLQ** with the **trellis search** (qtf2 Viterbi over 2^16 states per 256-weight tile; codebook i8 / i4 / mul1);
   two-sided **ldlq_2hess** when H_out is on; **tile CD** (cd3, attn_qg only, #77, w50's tilecd.py) after LDLQ;
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

### S4 swj (`rxq/swj.py`; level pass)
Joint gate / up output-scale refit through the SwiGLU on frozen codes (#85; **#104 PASS at 27B K2.5**, −6.5%;
K ≥ 4 pending, SWJ27 amendment 1). Needs the bf16 27B's MLP inputs over ~0.93M qad tokens: a staged forward on one
card, all sources (levels) in one pass (swjfit27.py, ~1.5 card-h for 2 levels on the R9700). Ported: the grid sums,
the fit and the container edit (`apply_level`, byte-identical). The level pass itself is called in place
(`fit_level` → swjfit27.py) until ported. Output: the level with gate / up svh × c. Not a donor job (bf16 model +
boundaries ~20 GB on disk, a 32 GB card).

### S5 writers asym ρ0 (`rxq/asym.py`; level pass + per-group CPU/GPU + S2 jobs)
(#87; **#105 queued at 27B**, OPEN.) Three parts: (a) sibling moments H / C / N of the 128 writers' inputs through the
level's own quantized siblings (sibmom27.py, a level pass, ~2.5-3 card-h, 262 GB of moments); (b) per writer group
the denoised metric G and target transform F (`Denoiser`, port of mkdenoise.py: default and lean paths,
byte-identical); (c) the 128 writers requantized as S2 jobs with `hin = G`, `wpre = F` (bit-identical to ourq
--hin --wpre). Must run after swj and anything else that changes a sibling.

### S6 healing (deferred, #89)
Frozen-code scale healing per level; after the freeze, beside the ladder. Re-run S7 after it.

### S7 census + knapsack re-run (rad-exl3-mix search/kfit2.py, knap4)
Not ported. Per level: proxy logs (census of K, codebook, proxy error per tensor) feed kfit2's sensitivity model;
fit-pool swap measurements calibrate it; knap4 picks per-tensor (level, compute mode) per size and variant. Re-run
after swj / asym / healing (they move low-K sensitivities).

### S8 pack (rad-convert; CPU)
`rad-convert ... exl3i src=LEVEL codebook=mul1i8|i4|mul1 rate=K`, then radmix into the release mix, then the census
diff (`swap27k.sh packmixk`). ~1,600 s per 27B level on gpubox (PROFILE.md), under conv.lock.

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
  `log.jsonl` (ourq's proxy record + stage times + peak VRAM), `job.json` (spec, stamps, device, seconds). Optional
  `wq/` (dequantized f16 weight, ourq's --save-wq) for verification by score (rad-quant-at-home VERIFY.md).
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
  = sha256 of rxq's modules, exllamav3's quantize.py, the kernel sources and tilecd.py. Assembly refuses a level whose
  tensors disagree on either. A donor's result is accepted by score (rad-quant-at-home VERIFY.md: decoded codes'
  proxy error inside the band for its (tensor, K)), so the code stamp is recorded, not enforced, across vendors.
- **Level passes** (S4 swj fit, S5 moments, S1 collections) are not donor jobs: whole-model forwards on a 32 GB card
  with ~20-260 GB of scratch; they run on our cards (gpubox) or trusted big boxes (rad-quant-at-home TRUSTED_ONLY).

**Contract with rad-quant-at-home** (coordinate by files; w47 owns that repo): its `quant_layer` job type runs
`ourq.py --layers L-L` and requires `rowerr.npz`; an `rxq.quant` job type would run `python -m rxq job SPEC -o OUT`
(per tensor, `--save-wq` for the rowerr claim) and accept `t-<key>.safetensors` + `log.jsonl` + `job.json`. Proposed
in `local/workers/w52/rqah-jobtype.md` (rad-exl3-mix), not edited into that repo.

## 4. The settings file (`config/v3.toml`)

Every result-affecting setting, with its evidence (experiment # and status, 2026-11-06):

| setting | value | evidence | status |
|---|---|---|---|
| codebook by level | i4 K ≤ 1, i8 K1.125-7, mul1 K6-8 | #43, #90-02, #94 | adopted |
| qtf2 | on | #37 | adopted |
| scaled_hessian | on | ladder setting since 11-01 (no own INDEX row) | adopted |
| damping σ | 0.025; input group 0.005 | #38 (re-checked fractional i8, 11-06) | adopted |
| sub-1 input damping 0.05 | switch, off | #38 verdict wording; not in ourq | **lead decides** |
| gain | 1.0 | #39 (flat) | adopted |
| output scales | always | #48 (auto rule +2.5%) | adopted |
| LDLQ drift | interpolated | ourq trunk review 11-02 | adopted |
| tile CD cd3 | attn_qg only, switch off | #77 (user adopted 11-06), w50 GPU impl, CD27 check | switch until CD27 passes |
| H_out (bn, 1,920 seqs, σ_out 0.01) | switch off | #33-27 done, #33-28 running | OPEN |
| xcorp H_in blend w 0.5 | switch off | #101 (0.8B pass, 27B pairs running) | OPEN |
| swj | on, K ≤ 3.5 | #85, #104 PASS | adopted (K ≥ 4 open) |
| writers asym ρ0 | switch off | #87, #105 queued | OPEN |
| #100 hash codebook | switch off (refused) | #100 planned | OPEN (format) |
| healing | off (refused) | #89, deferred | later |
| dropped: i8c, YAQA full, trunk row-slab, act order, chained tail-biting, best-of-N seeds, group scales | — | #78-04, #33, #96, #103, #102, #41, #40 | dropped |

Interactions the config enforces: tile CD with H_out is refused (tilecd.py does not implement ldlq_2hess); healing and
#100 are refused until implemented.

## 5. Open items for the lead
1. **i4 sub-1 levels need a qtf2 i4 build** (`exl3frac2` exists for mul1 / i8 / i8c only; sub-1 needs qtf2): w53
   appears to be building `exl3frac2-i4`; the ladder plan's "i4 at K ≤ 1" depends on it.
2. **Sub-1 input damping:** #38's verdict says 0.05 below K1; ourq always passes 0.005. Switch in the config, off.
3. **ourq --save-wq rowerr:** ourq computes `rowerr.npz` from `w` after quantize_exl3, which has regularized that
   tensor in place (scaled, rotated, × g_scale): the per-row errors are not in the original basis. Ported as is (parity);
   rad-quant-at-home recomputes its claim from wq with rowerr_tool.py, so it is unaffected. Check headslab.py's use.
4. **H_out cost:** if H_out is adopted, ldlq_2hess makes a level ~5× slower (37 → 190 min on the R9700; 72% of a
   gate_proj is its dense F-update GEMMs, PROFILE.md T1); the rotated diagonal H_out is block-diagonal (128), which
   allows a ~40-90× cheaper update that is the same algorithm in exact arithmetic (autoresearch target, judged by the
   two-sided proxy error ≤ reference). It also raises gate / up's peak VRAM to 7.1 GiB.
5. **Port the level passes** (swjfit27 forward, sibmom27, mkhin27, collect) into rxq once their settings are final;
   until then rxq calls them in place.
6. **Vendoring:** rxq imports the kernels from a rad-exl3-mix checkout (RXQ_RXM) and the exllamav3 fork (PYTHONPATH,
   RXQ_EXL3_SRC). For the volunteer release: vendor exl3frac2 / exl3cb / exl3frac (+ tilecd once merged) into this repo
   and pin the fork as a submodule or patch series, so a worker image builds from this repo alone.
