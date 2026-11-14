# #178-03 — GDN a/b at high bpw: T4 vs T5 / T6 vs bf16 on the 5.0 / 6.0 / 6.5 incumbents — w149, 2026-11-13

Follow-up of #178 (w142, `../ab178/`, tested 2.5 / 4.0 only) and #178-02 (w147, T4 served). Rule: `RULES-17803.md`
(committed alone as 264c3f4 before any score). Results: `data/ab178/17803/` (README there).

| file | what |
|---|---|
| `RULES-17803.md` | the pre-registration: arms, containers (bf16 embedding), slice s173f, statistic, trunk price per size, per-size choice |
| `run17803.sh` | ds8 runner: per size the incumbent's .mix + `act <w> a8g128` on every `blk.*` weight + `token_embd.weight emb-bf16.rad` → radmix → census (full byte check vs donors and base, `--expect-act a8g128 --expect-codebook mul1i8 --expect-emb 'plain - bf16' --max-version 3`) → B scored; per T arm a copy, w142's `abpatch.py` with w142's donors (`/data/q27/w142/tq/ab-T{4,5,6}.safetensors`), census of the copy with B's mix + base (must show exactly the 96 a/b weights as "differs from base", nothing else), score, copy deleted; B mixes deleted once a size is done. Card locks tried non-blocking only (never queues on a held card) |
| `docsums17803.py` | per-document KL sums (`rxrun.hostd.doc_sums`) of every score → JSON (runs on ds8) |
| `ab17803.py` | RULES-17803's table: d (pmean over c / x), paired stratified bootstrap CI, price g = b_s × bpw saved, N = d − g, RULES-178 reading, per-size choice |
| `posthoc17803.py` | descriptive only, written after the data: top-document KL shares, d without the 2 largest-change documents per part, median per-document log ratio |
| `census-base-needs-mix.patch` | for the lead: `store/census.py` runs its `--base` comparison only inside `if mix:`, so `--base` without `--mix` compares nothing yet prints `[checked: base]`. The patch makes that combination an error |

Run (ds8): copy `abpatch.py` (from `../ab178/`), `store/census.py` (as `code/store/census.py`) and these scripts to
`/data/q27/w149/code/`, then `nohup bash code/run17803.sh &` (~35 min on 3 cards: 12 scores × ~5 card-min), then
`python -I code/docsums17803.py > docsums17803.json`; on the VM `ab17803.py docsums17803.json REGISTRY ab17803.json`
and `posthoc17803.py docsums17803.json REGISTRY posthoc17803.json`.
