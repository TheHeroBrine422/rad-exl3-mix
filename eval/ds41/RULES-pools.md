# DS4.1-F-1017 data pools (user, 2026-11-21) — rules written before any pool is used

Corpus: our d100b collection, `corpus/main100b.jsonl` (45,021 documents, 100,038,739 tokens; HF
`Hero422/deepseek-v4.1-flash-1017-quant-calib`; colo `/data/work/dscorpus/main100b.jsonl`) and g4 (`corpus/long1m.jsonl`, 8
whole long traces, 7.42M tokens).

1. **Unit = group:** a document's `session` if it has one (pieces of one long trace / session stay together), else its `name`.
2. **Assignment = hash, per category, token-weighted, fixed forever:** within each category, groups are ordered by
   sha256("ds41pools-v1:" + group) and cut by cumulative tokens into, in order:
   - **holdout** — the first ~2 % of the category's tokens (≈ 2M in total), split into two disjoint slices **h1 / h2** (alternating
     groups in hash order). **Never used** for anything — not selection, not tuning, not checks — until the user says so (final
     report only). Not even read for statistics.
   - of the rest: **fit + search 40 %**, **qad 25 %** (healing the error of OUR codes), **verify 15 %**, **reserve 20 %**.
3. **Reserve** is untouched until the user releases part of it (when we run short of clean data).
4. **g4 long traces:** 4 → long-context verify (whole-trace checks of finished mixes, sanity only, always reported together),
   4 → reserve; chosen by the same hash order.
5. **Already computed over all of d100b and allowed:** dense and per-expert Hessians, routing counts, expert-input dumps, embedding
   counts (covariances / counts over 100M tokens cannot memorize documents — the 27B rule for published Hessians).
   Anything we fit with many free parameters per token (QAD, our own output-side Hessians, search accept / reject) follows the pools.
6. **Never judge a decision on the pool it was tuned on; never iterate against verify.** A document-use ledger
   (`eval/ds41/ledger.jsonl`) records every golden slice built from these pools (documents, role, date). Overfit audits apply
   whenever many decisions rest on KLD from these pools.
7. The pool file `eval/ds41/pools-ds41.json` is read-only once made; never regenerate.
