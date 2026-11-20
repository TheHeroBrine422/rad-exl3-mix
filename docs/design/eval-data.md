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

**Final picks (user 2026-11-13):** the top ~5 candidates per release band are scored on **ver4 + a fresh xcorp part**:
the existing `golden-ver4` (91 verify docs, frozen) as the calib part, plus a large never-used xcorp draw (several
hundred docs, 8k cap). The rest of verify stays unspent for later releases (V3.1 QAD checks, V4). Pick = best pooled
(paired, per-document bootstrap), and not significantly worse on the calib part (its own CI; the calib part is small,
so it vetoes, it does not rank). Published numbers stay calib verify; the xcorp part is reported separately as the
real-world number. Public is never read for this.
**[2026-11-13, overfit audit]** "The rest of verify stays unspent" no longer holds: the V4 reserve (236 docs,
`ver-2026-11-01.json`) was never in the ledger, and ledger-clean draws took 182 of its docs (golden-verify27-w114,
w119-headx-verify; 0.8B g08-verify1 / -2 took 93 more). Untouched today: 28 verify docs + ver4. ver3 is spent for
selection (~100 scorings, the #171 monitor). What V3.1 / V4 judge on is open (user); the reserve is now in the ledger
(`@reserve-v4`).

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

## Amendment 2026-11-14 (user + lead, after #181's finding): every slice scores the board's gen rows only
The board scores a row only if its next token is one the model writes itself (challenge README "The score"; calib's
official `gen.u8`). Our 27B xcorp / trace goldens were recorded with an all-ones mask, and on the x parts the prompt-side
rows carried 73-83% of the KL (#181, w154). From now on every golden that decides or reports anything carries the
board's mask, rebuilt from tokens by `v3/qad181/genmask181.py make` (verified identical to calib's official mask on all
10.25M rows; intersected with any mask the golden already has). Applies to the search slices s173a-f (whole + x), the
final-pick slice golden-final750 and its x part, and every new slice. The old all-ones masks are kept as `gen.u8.all`.
Decisions taken before this on x parts (the paused #171 search) are superseded by the restart, not re-read.

## Principle (user, 2026-11-15): offline gates decide adoption, not trials
Any allocator, ranker or predictive model with a decent rank correlation on any size band (a CI that excludes zero,
time-ordered / out-of-sample, not overfit) is worth a live trial through the search inbox, even if it fails a
pre-committed adoption rule. Inbox candidates are judged by the live stage-1 + confirmation rule, so a weak model costs
only evals. Pre-committed winner rules (e.g. RULES-187 §5) still govern making something a built-in driver move.


## Appendix: 27B data rules as they stood in AGENTS.md (moved here 2026-11-20)

The challenge scores mean KLD vs the bf16 model on a **private** corpus drawn like the two published ones
(`StillDeadcode/qwen3.8-27b-quant-challenge`: `calib` 10.26M tokens / 4,789 docs, `public` 1.69M / 596 docs).

- **Selection and tuning happen on calib only.** Sensitivity fits, knapsack refits, local search, QAD training,
  hyperparameter choices: calib.
- **Calib is split into four fixed, disjoint pools**, each with the corpus's category mix. Two rules:
  never move a document between pools, and never judge a decision on the pool it was tuned on. (fit and search are one pool since 2026-11-13: the search refits on its own results; independence comes from slice rotation, fresh slices for finalists and verify.)

  | pool | docs (pools-v3) | use | long traces |
  |---|---|---|---|
  | fit + search (merged 2026-11-13, user) | 551 + 583 | sensitivity measurements, knapsack refits, search accept-reject: one pool; draw least-used docs first (ledger decision counts); pool ids unchanged | ripgrep, cobra |
  | qad | 2,799 (largest) | QAD / healing training data | attrs, click |
  | verify | 856 | checking finished mixes, QAD before/after. Routine checks only: every reuse bakes in a little noise, so never iterate against it | rich, express, lua, requests (lua and requests moved from qad on 11-01) |

  - **Pool file: `pools-v3.json`, the only one to use.** On iceman `/root/challenge/pools-v3.json`; copy in
    `/workspace/rad-exl3-mix/eval/pools/`. `pool[i]` is the pool of calib document `i`. Read-only: never regenerate.
    History: `pools.json` (made by `mkpools.py`) was the original; on 2026-11-01 the user moved 11 qad docs to
    verify for ver3/ver4 (record `eval/pools/amend-2026-11-01.json`). Token counts before the move: fit 0.92M,
    search 1.42M, qad 5.74M, verify 2.18M.
  - **Long traces:** calib has only 8 (~250k tokens each), split as in the table (from the per-document `pool` list;
    the `traces` field inside pools-v3.json is stale, pre-move: ignore it).
  - **Long-context testing (user, 2026-11-07):** capability via RULER (`eval/ruler/`). The long traces are cut into
    ≤ 32k turn-aligned chunks (unscored system / tool header) for routine scoring; CIs on chunks resample by trace
    (`eval/traces/`). Both verify traces are also kept whole for rare full-length checks of finished mixes: a sanity
    check (n = 2), never a selection or tuning target, always reported together.
    The public corpus has 2 more long-agentic traces (~245k each): at the end, report full-length KLD on them with the
    other public numbers (report only, after selection, like all of public).
  - **Building slices:** `calib_slice.py CORPUS TOKENS CAP SEED POOL`.
  - **Document-use ledger: `eval/pools/ledger.jsonl`** — one line per (calib document, use): which golden or
    collection used it, its role (train / fit-hessian / fit-diag / fit / select / monitor / check / final), the
    line (v2, v3, ...), the scored position range and date. Built by `mkledger.py` from the goldens plus the
    hand-written roles in `ledger-roles.json`. To add new uses without a rebuild: `ledger_add.py` (goldens' kld.json,
    `--all USE DATE` for whole-calib collections; add the role to `ledger-roles.json` first; skips duplicates).
    `provenance.py` gives the per-golden pool mix and overlaps, and the "clean" verify docs no golden has touched.
    **Check it before building a verify slice, and add every new golden or collection that reads calib.**
  - **Old slices:** `golden-calA` (calib-A) is inside fit, so it is fit data. `golden-calB` / `golden-calBbig`
    (calib-B) are mostly verify but contain the first 8k tokens of ripgrep, cobra, click and requests (now search
    and qad). They are fine for comparisons made before any search or QAD; after that, verify only on slices built
    from the verify pool.
- **V3.1 QAD data (user, 2026-11-16):** after the V3 final picks, QAD may train on the qad pool **plus the fit + search
  pool** (~8M tokens; selection is finished, so the bias no longer matters for it). **Verify stays a real holdout:** no
  verify document (ver4, golden-final750, agent-verify32, any verify slice) is ever trained on, and every V3.1 number
  is read on verify only (never on s195 / phase-1 slices, whose calib parts become trained-on). Ledger: role train.
- **Published input Hessians** (`hessians/calib/`, sum of x x^T over every calib token, verify pool included) are
  fine to quantize against: a per-layer covariance over 10M tokens can't memorize documents, and every entry uses
  them. Anything *we* fit with many free parameters per token follows the pools: QAD and our own YAQA output-side
  Hessians on qad, local-search accept/reject on search.
- **Public is held out.** Use it only to report numbers at the end; it never selects, not even among finalists or as
  a tie-break (user, 2026-11-01). Never run a tuning loop against public, never fit anything on it.
- **Private is the real test.** A gap between our public numbers and the leaderboard means we overfit somewhere.
- **Overfit audit every ~2 days (user, 2026-11-08).** Go through the decisions since the last audit (LOG): which
  slice decided each one and how often it has been reused, whether anything was fitted or picked outside its pool,
  whether a rule was changed after its data, whether adopted settings hold on verify (and score them there once
  if they have no verify rows), and that public was never touched. Log it as a dated LOG entry ("Overfit audit")
  with the actions, and rotate the search slices: 27B search slices are two-part and calib-sparing (~65% fresh
  xcorp redrawn every round + ~35% fresh calib search-pool documents; same sign required on both parts;
  `docs/design/eval-data.md`); retire any slice that has decided many pairs. Audits: 2026-11-08, 2026-11-10, 2026-11-13 (w141; next due ~11-15, and before the first ver4 read of the finals if that comes first).
- **Report uncertainty.** Compare models paired on the same documents, with a per-document bootstrap interval
  (`/workspace/rad-exl3-mix/mix/paired.py`, `/workspace/rad-exl3-mix/search/abreport.py`). On ~100k-position slices the paired SE is ~±1.5%; do
  not treat smaller differences as real. Small slices cap documents at 8k tokens — long-context behavior needs
  whole traces (TP2 for >~170k at high bpw).
- Never train or calibrate on the public corpus.
- **Our own pi session logs (`/workspace/pi-sessions`) are an allowed training / statistics corpus** (user,
  2026-11-02, deliberate: it is data made in this research, which is public anyway; supersedes the earlier "not the
  user's traces" rule). Use them like xcorp: QAD / healing data, Hessian and H_out collections, long-context
  statistics; never for selection or verification (they are not drawn like the private corpus). They contain private
  material (IPs, key paths, emails, DMs): never publish them or text derived from them; keep builds under `/data` or
  `local/`, and record their use in the ledger as a non-calib corpus.
- **xcorp is allowed for training too** (healing, QAD; user 2026-11-04: real-world use is the goal, not the board).
- **Search pool from xcorp (user, 2026-11-04):** xcorp (+ a share of the user's traces) may be used for **search
  accept/reject**, drawing fresh documents per round, with our own bf16 goldens. **Every published number stays on
  calib verify** (drawn like the private corpus). **Final picks** (user, 2026-11-13): ver4 (calib verify, frozen) + a
  fresh xcorp part, pooled best with the calib part as a veto (`docs/design/eval-data.md`); keep the rest of verify
  unspent. Fitting and search share the merged fit + search pool (2026-11-13).
- **The user's traces never leave our own machines** (iceman, the 4070, colo, Thor, the VM, the dad's RDNA4 boxes): no donated-compute job may
  carry them or anything derived per document (goldens, slices). Donor jobs use xcorp / calib-based data only;
  trace-based evals run on our cards. The user's dad's boxes count as ours (user, 2026-11-04). **box2 (the lent 4× R9700 box) counts as ours too (user, 2026-11-12):** trace / session-derived data and xcorp goldens may go there.
  Corpus list: `docs/CORPORA.md` (light, public), details and locations in `local/notes/corpora.md`.
- **Unpublished material others share** (papers, PDFs, DMs, screenshots) stays out of the repo: notes in
  `local/research/`, files under `/workspace/references/` (user 2026-11-04).

### V3 size metrics (moved from AGENTS.md 2026-11-20)

- **Size metrics (user, 2026-11-13, refined for the V3 card 2026-11-18 ~1:55 PM):** the V3 card presents
  two numbers: (A) **leaderboard** = trellis + trunk + head (the LM) × 8 / 26,895,998,464 — the `vram/`
  table's single bpw column, what the board scores; (B) **VRAM usage** — `vram/` = A; the `host-emb/`
  table shows the VRAM usage in GiB (trunk + head; the bf16 table is in host RAM) and how much lower it is
  than the `vram/` twin (= the vram file's trellis table, 0.23–1.04 GiB) — no bpw column for host-emb.
  head × 8 / 25,625,844,224 (the bf16 table is in host RAM), its own table. Tower + drafter in neither
  (optional, noted "come on top", IQR-V4 convention). Both formulas stay the record for other reports.
