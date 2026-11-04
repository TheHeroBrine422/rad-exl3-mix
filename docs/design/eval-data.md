# Evaluation data for EXL3-V3 (decided 2026-11-04)

Why: V2's search reused one calib search pool, and best-of-many picks scored on reused slices were ~3% optimistic on
held-out data. V3's menu is far larger (66 levels, f16 / A8 / A4 modes, A1, sub-1, fast and accurate variants), so the
search needs fresh documents every round, and the user wants real-use coverage (agentic, long context).

## Roles

| data | role |
|---|---|
| calib, all | the published input Hessians |
| calib fit | sensitivity model fits (kfit2 / knap4) |
| calib search | search accept/reject (with xcorp and own traces); its 2 long traces = long-context guard during the search |
| calib qad | training: H_out collection, healing |
| calib verify | **final picks and every published number**; its 2 long traces = the finals' long-context check |
| public | final reporting only |
| xcorp (`docs/CORPORA.md`) | fresh search slices (8k chunks); healing data |
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
  after the swap test. Both uploads need the user's go per repo.
