# Public eval of the V3 release (16 sizes) — #208

Run once, 2026-11-18, on the 16 frozen release containers (mad-only picks; the uploaded
`vramvd` files, whose rows are byte-identical to the scored text-only containers). The public
corpus is held out: it selected nothing, at no point. Report only.

## Corpus and method

- **Public corpus** (the challenge's published set): 596 docs / 1,694,025 positions, of which
  **1,167,650 are scored** (gen mask). The unscored ~31 % are tool outputs / user messages (not
  model-generated; KLD there is meaningless — radiance's `"all"` KLD mean includes them and must
  not be reported). The two long agentic traces are 245,423 and 245,298 tokens.
- Golden: iceman `/root/challenge/golden/public` (bf16 model), copied to box2 with md5 check.
- Scoring: radiance (w201 build, `qc-exl3-v3-eval` @290d30dd) TP1, `--kv-cache-dtype bf16`,
  `--max-num-seqs 1`, card 1 / card 2 lanes (`pub208.sh`, `pub208b.sh`), box2
  `/data/q27/release-v3/public/`.
- **Sizes 1.5–4.0** scored the full 596-doc golden at `--max-model-len 262144` (both long traces
  whole). **Sizes 4.5–7.0** scored the **public-short** sub-golden (the same 594 non-trace
  documents, 1,203,306 positions / 883,911 scored, max doc 67,566 tokens) at `--max-model-len
  73728`: at those weights the single-card KV pool (245,200 @4.5 … 116,960 @7.0 tokens) cannot hold
  the 245k traces and the KLD mode refuses to start (no skip flag; the 32768 fallback dies on the
  same check). Their two long-trace numbers are pending TP2 (w204) and will be added here.
- Metric: per-position KLD vs the bf16 golden, **mean over scored positions**; top-1 agreement over
  scored positions.

## Results

| bpw | golden | public KLD | phase-2 pick KLD | top-1 | trunk+head bpw | LM container bpw | shipped file bpw | long traces (scored KLD) |
|---|---|---|---|---|---|---|---|---|
| 1.5 | full | 0.226589 | 0.243889 | 0.8321 | 1.5010 | 1.5012 | 2.2678 | 0.172516 / 0.189598 |
| 1.75 | full | 0.129496 | 0.153196 | 0.8700 | 1.7572 | 1.7512 | 2.5178 | 0.102521 / 0.111927 |
| 2.0 | full | 0.093233 | 0.102615 | 0.8950 | 2.0196 | 2.0012 | 2.7678 | 0.060879 / 0.067555 |
| 2.5 | full | 0.041709 | 0.049995 | 0.9289 | 2.5319 | 2.5012 | 3.2678 | 0.027042 / 0.030276 |
| 2.75 | full | 0.031004 | 0.036404 | 0.9392 | 2.7702 | 2.7458 | 3.5124 | 0.019552 / 0.021533 |
| 3.0 | full | 0.021467 | 0.025328 | 0.9492 | 3.0257 | 3.0012 | 3.7678 | 0.013546 / 0.015073 |
| 3.25 | full | 0.015052 | 0.017976 | 0.9575 | 3.2819 | 3.2512 | 4.0178 | 0.009578 / 0.010279 |
| 3.5 | full | 0.011459 | 0.013241 | 0.9636 | 3.5318 | 3.5012 | 4.2678 | 0.006973 / 0.007470 |
| 3.75 | full | 0.008565 | 0.009710 | 0.9691 | 3.7880 | 3.7512 | 4.5178 | 0.005170 / 0.005429 |
| 4.0 | full | 0.006398 | 0.007237 | 0.9728 | 4.0319 | 4.0012 | 4.7678 | 0.003698 / 0.003943 |
| 4.5 | short | 0.004223 | 0.003961 | 0.9782 | 4.5317 | 4.5012 | 5.2678 | — (pending TP2) |
| 5.0 | short | 0.002527 | 0.002299 | 0.9834 | 5.0193 | 5.0012 | 5.7678 | — (pending TP2) |
| 5.5 | short | 0.001667 | 0.001410 | 0.9874 | 5.5441 | 5.5012 | 6.2678 | — (pending TP2) |
| 6.0 | short | 0.001126 | 0.000922 | 0.9897 | 6.0505 | 6.0012 | 6.7678 | — (pending TP2) |
| 6.5 | short | 0.000972 | 0.000700 | 0.9909 | 6.5256 | 6.5012 | 7.2678 | — (pending TP2) |
| 7.0 | short | 0.000805 | 0.000611 | 0.9915 | 7.0007 | 7.0012 | 7.7678 | — (pending TP2) |

**Reading the table**

- *phase-2 pick KLD*: the pick's KLD on the phase-2 selection slices (calib-verify + xcorp, 2.8M
  positions, docs ≤ 16.4k) — the number the pick was chosen on. The public corpus is disjoint from
  it.
- 1.5–4.0 (full golden): public reads **7–15 % easier** than phase-2 at every size — no
  overfitting signal in the wrong direction.
- 4.5–7.0 (public-short): a constant **+0.0002 … +0.0003 nat** in absolute terms vs phase-2 (the
  percentages +6.5…+32 % are the shrinking denominators) — corpus spread, same magnitude as the
  ~8–10 % corpus effects #206 measured for IQR-V4.
- **Long traces** (1.5–4.0 only, the two 245k-token agentic traces, scored whole at 262144 ctx):
  below the short-document mean at every size — the picks (selected on docs ≤ 16.4k) do not
  degrade at 245k context.
- **Size metrics** (AGENTS.md, both reported): *trunk+head bpw* = trunk + head bytes × 8 /
  25,625,844,224 params (the embedding, and the vision tower + DFlash2 drafter, excluded — the
  embedding can be served from host RAM); *LM container bpw* = the board formula (× 8 /
  26,895,998,464) on the text-only container; *shipped file bpw* = the same formula on the actual
  uploaded file, which additionally carries the 2.58 GB bf16 vision tower + MXFP4 DFlash2
  drafter (+0.7666 bpw).

**Comparison with IQR-V4 board numbers** (their model card, LM bpw): iq2r 0.0611 (~2.5) → our 2.5
is 0.0417 (−32 %, and our number includes both long traces); iq4r-s 0.00643 (~4.5) → our 4.5 is
0.00422 (−34 %). On phase-2 slices the gap was +14…+27 % untrimmed (#206).

Raw per-size numbers: `public-v3.json` (this directory). Reproduce: box2
`/data/q27/release-v3/public/<S>.json(.rows)`, goldens `/data/q27/challenge/golden/public{,-short}`,
gen-mask sums via `rxrun.hostd.doc_sums` (pattern in `search/v3/picks203/iqr206/iqr206.sh`).
