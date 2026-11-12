# eval/slices — fresh two-part 27B search slices (w120, 2026-11-12)

`docs/design/eval-data.md` "two-part, calib-sparing": each search slice = an xcorp part (fresh pieces of the 27B bf16
xcorp goldens, never scored) + a calib part (fresh search-pool documents, cut from the published calib golden). Parts
are scored and read separately; decisions need the same sign on both. The readout: `eval/docstat/readout.py`.

| file | what |
|---|---|
| `mkslice27.sh NAME:XBATCH:SEED ...` | builds `golden-search<NAME>-{c,x}` on the host with the calib golden + night1105's xcorp goldens (gpubox), CPU only, ram.lock shared; restartable; writes `registry.jsonl` and `ledger/<golden>-kld.json` for the ledger |
| `excl.py` | the calib exclusion list from the ledger (`never`: no ledger row at all; `no27b`: only 0.8B / 2B proxy uses) + slices built earlier in the same run |
| `pickx.py` | pickx84 (stratified xcorp pieces, earlier slices' xcorp documents excluded) + `--cap-cat CATS:N`: pieces of those categories cut to their first N scored positions (prefix cut, corpus `maxtok`) |

## Sizes (why: no document may hold more than 20% of a part's KL)
Per-document KL density (doc mean ÷ part mean) on three past slices (`readout.py dens`; V2-3.00-like K2.5 arms on 27c /
27d, U1 on 27f): the xcorp part's heavy documents are **agentic / tools / long / reasoning** pieces (density 2–9×; the
rest ≤ 2.7×). Largest single-document share at 8k: xcorp 54% (27c, 65 pieces), 28% (27d), 23% (27f); calib 19%
(27c, 26 docs), 12%, 11%. Monte Carlo over those 325 xcorp / 158 calib documents (4,000 draws):

| part | documents | per-doc cap (scored) | positions | max share median / p95 | P(> 20%) |
|---|---|---|---|---|---|
| xcorp | 130 | 8k (as 27d–27f) | ~207k | 21% / 33% | 0.56 |
| xcorp | 130 | heavy 4,096, rest 8k | ~191k | 13% / 20% | 0.04 |
| xcorp | **130** | **heavy 2,048, rest 8k** | **~182k** | **11% / 15%** | **0.001** |
| xcorp | 160 | heavy 2,048 | ~224k | 10% / 13% | 0.000 |
| calib | 40 | 8k | ~50k | 15% / 22% | 0.11 |
| calib | **≥ 45** | **4,096** | **~45k** | **≤ 12% / 16%** | **< 0.01** |
| calib | 60 | 8k (as 27d–27f) | ~75k | 11% / 15% | 0.001 |

Defaults: xcorp 130 pieces with the heavy categories capped at 2,048 scored positions; calib `CTOK` 45,000 tokens at
cap 4,096 (≈ 45–55 documents). The readout still caps any document at 20% of its part (`r_cap`) and flags it.
Top-1% positions hold ~50% of the xcorp part's KL (27d, K2.5): the tail decides; read p99 beside the mean.

## Calib budget (lead decides)
Search-pool documents in no ledger row: **93** (≈ 83k tokens at cap 4,096) → **two** calib parts at the default size.
Another **121** were used only by the 0.8B / 2B proxy slices (g08-search1 / g2b-search1; other models): `MODE=no27b`
makes them eligible (≈ 165k more tokens → 3–4 more parts). The design's ~35% calib share (by tokens) is out of reach
at either budget with 130-piece xcorp parts (calib ≈ 20%); options: smaller xcorp parts, or reuse calib parts across
two rounds (the design's "rotated slowly"). Fresh xcorp is plentiful (b2-c0 / c1 / c2: ~550 unused pieces each).

## Build (not run yet: the lead decides the calib budget first)
**Not built by w120:** w115's rule (`v3/e3/settle/w115.md`) may build **golden-search27g** from the same 93
never-used documents (60k tokens: ~65 of them). Building ours first would starve it, so the names below start after
it and the lead orders the builds. On gpubox (stage the code at the commit whose ledger excludes every used document;
after w115's 27g is in the ledger):
```
CODE=/data/w120/code OUT=/data/slices bash $CODE/eval/slices/mkslice27.sh 27h:xc8k-b2-c0:11201
MODE=no27b CODE=... OUT=/data/slices bash $CODE/eval/slices/mkslice27.sh 27i:xc8k-b2-c1:11202 27j:xc8k-b2-c2:11203 27k:xc8k-b2-c0:11204   # after the lead's OK
```
Then on the VM: add `golden-search27g-c` ... to `eval/pools/ledger-roles.json` (role select, decisions 0 until used) and
`ledger_add.py` the copied `ledger/*-kld.json`. Score the reference containers on every new slice (V2 at the board
sizes; lead note 2) as eval units before any candidate, and read `readout.py dens` on V2 3.00's rows: a document above
20% there is cut (its pieces' `maxtok`) before the slice decides anything.
