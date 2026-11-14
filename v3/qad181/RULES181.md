# #181 rules: QAD scale-up, arm A with much more data (w154, pre-registered)

Written 2026-11-13 ~10:20 PM CST, before any #181 training step or score. Committed alone before the first run;
amendments are appended below with a time and a reason, never edits of the text above them.

## Question
How far does #165's arm A (trunk suh / svh log multipliers on frozen codes, the mix's quantized head frozen) go with
one epoch over tens of millions of stored bf16 golden positions instead of the pilot's ~1 M, on a **searched** mix
(what ships), does the relative gain depend on size or on the mix being heterogeneous, and what does it cost?
**Runtime: none** (same codes, same container layout and kernels; only the stored f16 suh / svh change).

## Start points (mixes)
- **inc2.5** = the #171 2.5 incumbent `v3s171-2.5-block-00626` (mul1i8 trunk from 15 levels `q27v3-144dd6e1-k*-i8`
  with swj scales 6c57a3d1, box2 `/data/q27/ladder-src/<level>/final`), R16 head `u2.75` (ds8 `heads/q/R16s003/K2.75`,
  donor `hd-u2.75.rad`). Served form = the incumbent's hostd .mix + `act <w> a8g128` on every trunk weight + the bf16
  embedding (`emb-bf16.rad`) over V2 3.00 (w149's #178-03 recipe; GDN a/b bf16 as in the incumbent).
- **2b:** the #171 incumbents at 4.0 (`v3s171-4-block-00420`) and 6.0 (`v3s171-6-explore-00372`), each with its own
  head pick, same served form; and the **uniform control u2.5** = level `q27v3-144dd6e1-k2.5-i8` (final, swj) for every
  trunk tensor with inc2.5's head / embedding / act modes (only the trunk allocation differs from inc2.5).
- A mix's trunk is built as one "virtual level" (`vlevel.py`: each tensor's trellis / suh / svh / mul1 copied byte for
  byte from the level the mix picked it from) that the student loads and the judgement packs (one exact-name recipe rule
  per trunk weight with its own rate).

## Data (training only; never judged on)
- xcorp: 40 xq chunks (gold4, top-256; xq-c00..39, 40.0 M tokens, 8k pieces) + 5 xc8k-b3 chunks (gold3, top-512);
  own traces: 4 tr32k chunks (gold3, top-512, 32k turn-aligned chunks; ds8 / box2 count as ours); calib: the qad pool
  of pools-v3 (the published whole-calib golden, top-512, gen rows only). Copies of every chunk on both ds8 and box2
  `/data/q27/w154/gold/` (rsync over the 10 GbE link, byte-identical).
- Loss: the challenge scorer's KL on the top-K + tail cell (`qad.chunk_kl`; the tail cell is the mass outside the
  stored K ids, so K 256 and K 512 rows use the same formula; checked: p summed over the K ids + tail = 1.00000 on
  every chunk). No per-document weighting: copy-heavy agentic / tool pieces cost the 2.5 student 1–3 nats a row (smoke
  test) and weigh in by their KL, as in the scorer.
- **Window policy:** every document from its start up to 8,192 tokens (every xcorp piece whole; the 32k trace chunks'
  first 8,192 tokens, ~44% of their positions; calib docs capped at 8,192). Documents are cut into segments that carry
  the student's state (attention KV, GDN recurrent and conv state) from the segment before, detached (truncated
  backprop: the gradient does not flow into earlier segments). 24 virtual streams × 1,024 positions a step (24,576
  positions a step): stream s trains on documents s, s + 24, ... of one seeded global shuffle over all documents of all
  corpora (mixed by document, so every step mixes corpora in proportion to their size) and continues its current
  document across steps. The plan does not depend on the number of cards. Gate (box2, before this text): a 2,048-token
  window vs 1,024 + 1,024 with the carried state agree to bf16 shape noise (xcorp doc: the carried half differs less
  than the no-carry first half, median |Δ| 4e-5 vs 3e-4 a row).
- Single epoch: no document is trained twice; the data hold ~51 M positions, the last budget is 48 M.

## Fixed settings
Adam (0.9, 0.99), lr 1e-3, 8 warmup steps, |log| ≤ 0.3 clamp, gauge as #89 (down_proj r, v_proj c fixed), seed 0,
student: int8 codes, f32 scales, bf16 activations (f16-activation path; the served form runs A8), GDN causal conv
`shift_conv1d`, the bf16 embedding (= the served one), the mix's R16 head frozen. Data-parallel, one whole student a
card, grads summed with gloo, the same Adam step on every rank (parameter checksums compared at every checkpoint).
**Schedule (warmup-stable-decay):** constant lr, linear decay to 0 over the last 20% of a budget. For budgets
T = 1, 4, 16 M a branch forks the main run where it has seen 0.8 T positions and decays to 0 at T on the same data; the
main run decays over 38.4 → 48 M. Checkpoint T = a one-epoch run of budget T (it has seen exactly the first T positions).

## Judgement (served path)
- Per checkpoint: the scales applied to the virtual level (`rxq.swj apply_level`: suh *= r, svh *= c, f16 as stored),
  packed (rad-convert, `--no-error`, the per-tensor recipe), mixed into the served form above, census (`store/census.py
  --expect-act a8g128 --expect-codebook mul1i8 --expect-emb 'plain - bf16'`), scored with radiance rad10 (bf16 KV,
  max-model-len 16384) on
  - (c) `golden-verify27-w114` (82 clean verify docs, the pilot's ver slice: its 3rd use, role check), and
  - (x) `golden-x181`: 150 fresh xc8k-b2-c2 pieces (pickx, seed 181, 174,257 scored positions), excluding every xcorp
    document of any earlier slice's x part AND every document with a piece in the training chunks.
- The **control** = the unhealed virtual level through the same path (pack → mix → census → score). Gate before any
  healed score: the control with the base (V2) embedding scored on golden-s171d reproduces w147's B rows of the same
  incumbent byte for byte (box2), i.e. the virtual-level pack is the incumbent's trunk.
- Readouts: ratio healed / control per slice with a 95% document-bootstrap CI (`healeval.py cmp` arithmetic, 2,000
  resamples, seed 0), docs better / worse; adjacent checkpoints T_k / T_{k−1} the same way; per position bin
  (`posci.py`).
- **A checkpoint gains over another** if the c ratio's CI is below 1 and the x ratio is below 1 (the pilot's rule).
- **"Still learning"** signal: the running training KL (each batch unseen before its update), EMA 0.95 per step and
  per corpus (xq / xc8k / tr32k / calib), logged per step; and the monitor (32 search-pool calib docs, first 2,048
  tokens, the pilot's monitor rule; report only, never a pick).
- **Recommended budget per size** = the largest budget T whose checkpoint gains over the next smaller one by the rule;
  if 48 M gains over 16 M, the curve has not saturated and that is reported.

## 2b (other sizes, uniform control)
inc4.0, inc6.0, u2.5: the same recipe (data order, settings) at a 4 M budget (WSD to 4 M, no branches), unless the
inc2.5 curve has scored ckpt-16M vs ckpt-4M before 2b launches and it gains by the rule: then 2b uses 16 M (decided and
logged before their launch). Each judged healed / its own control on c and x. Readout (descriptive): the ratio per
size and inc2.5-4M vs u2.5-4M (relative gain on a heterogeneous vs a uniform trunk).

## 3 (adapters, conditional)
Only if ckpt-16M gains over ckpt-4M by the rule: an adapter arm (rank 8, then 16 if memory allows on one card) at the
same budget as the comparison point, warm-started from inc2.5's ckpt at that budget, judged on the torch path vs that
checkpoint, size-adjusted (+bpw of the adapters at int8 and bf16, decode cost as w125 estimated); rules appended as an
amendment before it launches.

## Cards / hosts
box2 cards 0, 2, 3 (w152 holds card 1) and ds8's 8 cards after w151's ladder (never a card w151 holds; locks
`/data/q27/locks/card*.lock`). The run may move between hosts at a saved step (same plan, any card count); the pack /
mix / score steps take conv + ram locks as hostd does.

## Reported
The curve (KL ratio vs control and adjacent ratios vs budget, CIs, both slices), the running training loss and
monitor curves, positions/s and card-hours per budget, card-hours per size at the chosen budget, a V3.1 recommendation
(budget per size; adapters yes / no). Ledger: training qad-pool docs (role train), monitor docs (monitor), the verify
slice's 3rd use (check).
