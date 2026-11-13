# Evaluation data for EXL3-V3 (decided 2026-11-04)

Why: V2's search reused one calib search pool, and best-of-many picks scored on reused slices were ~3% optimistic on
held-out data. V3's menu is far larger (66 levels, f16 / A8 / A4 modes, A1, sub-1, fast and accurate variants), so the
search needs fresh documents every round, and the user wants real-use coverage (agentic, long context).
(The ladder is now 71 levels, `v3.md` "Goal and scope"; A1 was dropped on 2026-11-07, #81-04.)

## Roles

**2026-11-13 (user): fit and search are merged into one pool.** Why: V3's search refits the knapsack on its own
results (as V2's stream did), so a separate fit pool only bought independence the merged pool keeps through slice
rotation (~25 decisions a slice, own offset per slice), finalists re-measured on fresh slices, and verify for every
final pick and published number. qad and verify are unchanged and stay strictly separate.


| data | role |
|---|---|
| calib, all | the published input Hessians: **V3's only input statistics** (2026-11-06: the xcorp and self-generated blends were dropped, below) |
| calib fit + search (**one pool from 2026-11-13**, user) | sensitivity fits, knapsack refits and search accept / reject: 1,134 docs; slices drawn least-used first by the ledger's per-document decision counts; the pool ids in pools-v3.json are unchanged (fit and search are one role now, no document moved) |
| calib search (merged into fit + search above) | search accept/reject (with xcorp and own traces); its 2 long traces = long-context guard during the search, scored as ≤ 32k turn-aligned chunks (`agent-search32`, `eval/traces/`; 2026-11-07) |
| calib qad | training: H_out collection, healing |
| calib verify | **final picks and every published number**; its long traces: rich and express as ≤ 32k chunks (`agent-verify32`) for finished mixes, and both whole (`agent-verify-whole`) only as a rare full-length sanity check (n = 2, reported together, never a selection or tuning target; user 2026-11-07); lua / requests stay the whole ver3-long / ver4-long goldens |
| public | final reporting only; at the end also full-length KLD on its 2 long agentic traces (user 2026-11-07) |
| RULER (`eval/ruler/`, synthetic) | long-context *capability* (user 2026-11-07): retrieval / tracing tasks at 4k–768k; no corpus, no pool rules |
| xcorp (`docs/CORPORA.md`) | fresh search slices (8k chunks); healing data. Not a quantizer-statistics source (H_in blend dropped at 27B, #101) and not a substitute for calib when screening on the 0.8B (#93); see "xcorp as a statistics source" |
| own agent traces (private) | fresh agentic search slices (32k chunks, each = the session's system block + consecutive turns from a user-role message, so every chunk looks like a real agent request; all chunks of a session used; bootstrap by session); 2–3 whole long sessions as a real-use guard before release; our machines only |

## Goldens
- Made by stock radiance (1.17): `radiance --model <bf16> --kld-record DIR --kld-corpus FILE.jsonl --kld-top 512`, the
  same top-512 form as the challenge's golden (~4 KB a position), bf16 KV, TP2 for the bf16 27B.
- xcorp: only each document's first 8k (as calib slices; ~53k documents, far more than needed); short documents whole,
  never packed together (cross-document attention would change the predictions).
- Sizes, built in steps: first ~5M positions of xcorp (8k chunks) + ~2–3M of own traces (32k chunks); extend toward
  20M + 5M only if the search uses them up. ~100 GB at 25M positions.
- Search slices draw without replacement from these: a document scored in one round is not reused for accept/reject.

## Caveats
- Own traces are another model's sessions without its reasoning, scored with Qwen as teacher: off-distribution in
  absolute KL, fine for ranking (paired, same text). They are a share of each slice, not all of it.
- 8k chunks rank quants as whole documents do (#52: KL rises with position alike for all quants; #73: V2's edge held
  from 8k to 250k). Attention / KV-cache changes, healing and H_out get long-context checks anyway.

## Publishing
- xcorp + its goldens on Hugging Face (proposed `Hero422/xcorp-v1`) under the challenge's licensing model: each
  document keeps its source dataset's licence (README source table), goldens CC BY 4.0. Own traces are never published.
- The 27B H_out diagonals (plain, block-normalized, Sketch B, xw; a few snapshots) as `Hero422/qwen3.8-27b-hout-diag`,
  after the swap test. Both uploads need the user's go per repo. **[2026-11-05]** The H_out diagonals are published
  (user's go; merged layout, 0.67 GiB, `v3/yaqa/hf/unpack.py` restores the per-tensor layout bit-exactly).

## Self-generated text (decided 2026-11-05)
- **Generation:** a near-lossless quant served on one card (IQR-V4's IQ5R-S-V4, fp8 KV: ~0.0015 KL, ~98% top-1 vs
  bf16), at low priority beside normal use (`xcorp/selfgen.py`). Its text stands in for the bf16 model's own: the
  difference is far below the on- vs off-policy gap #91 measures.
- **Scoring stays bf16:** Σdy² collections sample their labels from the bf16 model itself (only the context comes from
  the quant); teacher logits for healing / goldens come from a bf16 record pass afterwards (stock `--kld-record` on
  TP2, or the single-card staged recorder). The `qc-kld-serve` patch (record while serving) is not needed for this.
- **[2026-11-05, #91] Not used for V3's statistics:** on the 0.8B, self-generated text blended into H_out or H_in
  never helped on calib (H_in w 0.1–0.5: own agentic text −5…−15%, calib +1…+6%); healing on it likewise traded calib
  for agentic text. No 27B self-generated collection was run for V3 (user, 11-05: no more Σdy² collections); a small
  256-sequence 27B self-gen Σdy² set exists and stays unused. The generation path above remains for eval text
  (agentic goldens) and for later real-use variants.

## Search slices from 2026-11-08 (user): two-part, calib-sparing
- Each new 27B search slice = ~65% fresh xcorp (redrawn every round; ~53k documents, never reused) + ~35% fresh calib
  search-pool documents (rotated slowly; the pool is finite: 583 documents), optionally a few ≤ 32k chunks of the
  user's traces (our cards only). Parts scored and reported separately; decisions on the pooled ratio, **same sign
  required on both parts**. First: golden-search27b for #117 (w84). golden-search27 (103 docs, overused 11-04…11-08)
  is retired for decisions. Verify stays calib (verify pool) for final picks and published numbers.

## xcorp as a statistics source (2026-11-05 / 11-06; #93, #101)
- **Screening (#93 amendment 1):** re-scoring the adopted 0.8B stack on fresh xcorp keeps the sign of every step but
  only 52–91% of its calib gain (H_out and swj outside ±30%): xcorp does not replace calib for 0.8B decisions;
  calib-fitted statistics help most on calib-like text, and real-world text keeps ~75–80% of the gain.
- **Input-Hessian blend (#101):** H_in = (1 − w) calib + w xcorp. On the 0.8B, full mX at w 0.5 passed (calib
  K3 0.981 / K2 0.984, fresh xcorp 0.967 / 0.936; verify keeps the gain; agentic text +2–4% worse; on top of the full
  stack −1%). That recipe is recorded. **At 27B it shrank to ≲ 1%** (K2.5 pairs 0.988 and 0.997, pooled ~0.993 vs
  the ≤ 0.98 rule) and the user dropped it (2026-11-06; the remaining arms were cancelled). Broader corpora for
  statistics are a question for future models (BACKLOG).
- So xcorp's roles in V3 are evaluation (search slices) and training data (healing), not quantizer statistics.
