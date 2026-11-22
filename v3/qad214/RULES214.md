# RULES214 — V3.1 QAD pilot on the released picks at 2.50 / 3.00 / 4.00 (w214)

Written 2026-11-21 5:20 PM CST, before any #214 training step or healed score; committed alone. Amendments are appended below with a time and a
reason, never edits of the text above. Brief: `rad-local/workers/instr3/w214.md` (lead, user 11-21: 2.5–3.0 matter most, 4.0 second, ≥ 5.5 dropped).

## Question
How much does scale-only QAD (arm A: trunk suh / svh log multipliers on the frozen trellis codes, the mix's quantized head frozen) buy on the
**released EXL3-V3 picks** (HF `Hero422/qwen3.8-27b-exl3-v3-rad`, host-emb files, sizes 2.50 / 3.00 / 4.00), judged on the served path on a fresh
calib-verify slice, and does A8 in the loop (`--a8`, arm Q) matter at 4.0 (where arm A was ≈ null at 1 M in #181: 0.993)?
**Runtime: none** (scales fold into the container's f16 suh / svh planes).

## Arms
| arm | size | flags |
|---|---|---|
| 25A | 2.50 | arm A (re-establish #191's −6 % at 1 M on the released pick) |
| 30A | 3.00 | arm A (never tested at 3.0) |
| 40Q | 4.00 | arm A + `--a8` |
| 25Q | 2.50 | arm A + `--a8` — only if 40Q or 30A suggests A8-in-the-loop matters (lead decides; I report) |

## What changed from the brief's inputs (lead, 11-21, after w214's escalation: box2's #181 sources were cleaned)
- **Student = the released container.** `v3/qad214/rad2level214.py` reads the host-emb `.rad` (exl3i, mad; record = exllamav3's tile stream) into an
  ourq-layout level (trellis / suh / svh per trunk linear, K per tensor as released), the quantized R16 head (+ R16 permutation) and the GDN a / b
  weights (the container's K4 `ssm_ab`, decoded to bf16; qad200's `--ab`). The token table is the checkpoint's bf16 (the host-emb files carry the bf16
  table). host-emb is the same trunk / head / a-b as the vram board file; used because the student's bf16 embedding then equals the served one exactly.
  Check before training (reported): decode vs the bf16 checkpoint (relative error by K), the unhealed student's per-row KL vs radiance's rows on
  the same documents.
- **Packer = scale swap.** `swap214.py` writes f16(stored f16 × multiplier) into a copy of the container's suh / suh.1 / svh planes (apply_level's
  arithmetic). Gate: identity swap = byte-identical file; trained swap differs only inside trunk scale planes.
- **Judge = plain radiance `--kld-ref`** (`score214.sh`; box2 w206 install = the release's build, A8 as released, `--embedding-placement ram`).
- **Judge slice j214** (`slices/golden-j214`, box2): verify-pool documents of the whole-calib golden with no select / final / fit use in the ledger,
  none in the V4 reserve, none in any golden-verify27-w114* QAD slice, none in a QAD training use; 80 documents (cap 8,192 tokens), 121,641 gen rows
  (142,723 rows scored by radiance; every number below counts gen rows only, the board's rows), drawn per category in proportion to the candidates'
  gen rows (`mkjudge214.py`, seed 214). Ledger: role `check`, use `qad214-judge` (added before the first healed score). Verify is never trained on.
  Per-document paired readout vs the unhealed release on the same slice (`cmp214.py`: ratio of summed gen-row KL, 2,000 resamples, seed 0, 95 % CI).
  Only j214 is used; the earlier w114 slice is not read (its documents are excluded from j214), and no xcorp judge part (none at hand).
- **Training data** (the lead's amendment of #181's mix; the #181 xq / xc8k-b3 goldens were cleaned): xcorp + own traces from Thor's backup goldens,
  `xc8k-b2-c0..c2` and `tr32k-b2-c0` (top-512, all rows recorded; gen masks rebuilt from tokens with `genmask181.py`, exact on calib), each document
  from its start to 8,192 tokens; plus a seeded subset (seed 214) of the calib **qad + fit + search** pools (gen rows, the official gen.u8, each document
  from its start to 8,192 tokens), sized to ≈ 20 % of the mixed gen rows (#181's mix was ≈ 80 % xcorp). Verify-pool documents never. Mixing as #181:
  one seeded global shuffle by document over all corpora, 24 streams × 1,024 positions a step, states carried across segments. Stage 2 (4 M rows) would
  need more xcorp (`xc8k-b1`, 5 M tokens, on Thor); that copy and an amendment come first. Ledger: the calib training documents, role `train`, use
  `qad214` (before training).
- **Batch.** The brief says "4× batch" (#200's B4). At the pilot's budgets that leaves 12 optimizer steps at 1 M rows (3 at 0.25 M, with 8 warmup
  steps), so stage 1 uses **#191's normal batch** (24 × 1,024 positions a step, ≈ 47 steps at 1 M, warmup 8, the recipe that gave −6 % at 1 M);
  stage 2 (4 M rows) would use 4× batch (`--accum 4`, 49 steps at 4 M, warmup stretched 8/49 as #200's B4) if earned. #200 found B4 ≈ A4M at equal rows.
- Trainer: `v3/qad200/qad200.py` (#196 / #200 lineage: `--ab`, `--a8`, WSD branches, `--accum`, fixed eval; qmodel's Triton int8 → bf16 copy from w215,
  bitwise identical), lr 1e-3, trust clamp |log| ≤ 0.3, gauge as #89 (down_proj r, v_proj c fixed), seed 0, `--conv shift`. One whole student a card, one arm a card. Student: int8 codes (mad is an integer grid: q 4 k_inv, |q| ≤ 126,
  bit-exact check at load), f32 scales, bf16 activations (a8g128 STE for the Q arms).

## Stage 1 (budget 1 M gen rows; WSD checkpoints at 0.25 / 0.5 / 1 M = proper decayed one-epoch runs)
Order 25A / 30A / 40Q in parallel on box2 HIP cards 0 / 1 / 2 (card 3 for the served reads, or 25Q if earned). Each checkpoint is judged on j214 on the
served path vs the unhealed release (control scored once per size). 25A / 30A always run to 1 M. **Early stop (40Q, 25Q):** at the 0.5 M checkpoint, a
point estimate ≥ −1.0 % (ratio ≥ 0.990) vs unhealed → skip its 1 M read and stop it.

## Stage 2 (budget 4 M; checkpoints 2 / 4 M; same judge) — only for an arm whose 1 M read is ≤ −2 % (ratio ≤ 0.980) with the CI wholly below 1, or whose
curve is still clearly falling from 0.5 to 1 M (the 1 M ratio below the 0.5 M ratio by more than the 0.5 M-vs-1 M paired CI's half-width). Needs the
extra xcorp chunk(s) and its own amendment first. Report the curve.

## Reported (headline first, `out/w214.md`)
Per arm and checkpoint: ratio vs unhealed with CI, documents better / worse, per-category read (descriptive); the unhealed KL per size; training
curves (running train KL, fixed eval on held-out calib docs from the training pool, report only, never a pick); card-hours; the loader / packer checks;
the data deviations above. Raw results → `rad-exl3-mix/data/qad214/`. Not decided here: V3.1 recipe, adoption, any publication.

## Cards / hosts
box2 HIP cards 0–3 only (`/data/q27/locks/card<N>.lock`; card lock, then ram.lock shared for training / scoring; the packer's copy takes no GPU). RAM
per training job ≈ 17 GB at the student build (builds serialized by `--build-lock`); keep ≥ 15 GB free; `/data` ≥ 100 GB free. No iceman.

### Amendment 1 — 2026-11-21 6:52 PM CST, before any read of the new arms (lead, after 40Q's early stop): two control arms at 4.00
**Why.** 40Q's 0.25 M / 0.5 M checkpoints read 1.033 [1.004, 1.065] / 1.218 [1.037, 1.423] vs the unhealed 4.00 on j214 (31 / 49 and 18 / 62 documents
better / worse); the early-stop rule (0.5 M ratio ≥ 0.990) fired and 40Q was stopped at its 0.5 M checkpoint (card 2 freed). Its training log shows a
calib-batch spike in the last branch step (train KL 0.0126, calib 0.0395 vs 0.0068 a step earlier, grad norm 0.055) and its held-out fixed eval
(calib docs, torch student, A8 on) went 0.00684 → 0.00694 (0.25 M) → 0.01277 (0.5 M). #181's 4.0 lr screen (arm A, no A8) also ended worse than base at
lr 1e-3. A8-in-the-loop and lr 1e-3 on a ~0.006-KL model are confounded; two controls separate them.
**Change.** Added arms (nothing else changes: the released 4.00 pick, data, order, seed 0, lr schedule shape, normal batch, warmup 8, j214 judge):
| arm | size | flags |
|---|---|---|
| 40A | 4.00 | arm A, lr 1e-3 (card 2): separates A8-in-the-loop from the lr |
| 40A-lr3e-4 | 4.00 | arm A, lr 3e-4 (card 3): #181's next screen value |
Budgets 0.25 M and 0.5 M only (a checkpoint of budget T is a one-epoch run of budget T, so a main run decaying to 0.5 M reproduces 40Q's 0.5 M schedule
and data exactly); read each on j214 vs ctl-4.00. The decision rule is unchanged: a 0.5 M ratio ≥ 0.990 ends the arm (nothing beyond 0.5 M is run for
these controls in any case; 1 M for a control only if its 0.5 M read is ≤ -2 % with the CI below 1, the stage-2 trigger applied to the 1 M step). The spike
(batch, documents) is recorded in the report. 25Q is not launched (the lead).
