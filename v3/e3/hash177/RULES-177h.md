# #177 head bench rules: hash-codebook variants as R16 head donors, real head-only KL (w139, 2026-11-13 ~11:25 AM CST)

Written before any hash head donor exists. Committed alone before the first job. Lead notes 11-13 11:05 / 11:10 / 11:20 /
11:35 AM CST (`notes/w139.md`): the head is the hash study's test bench; this file replaces the colo proxy table as the
deciding evidence for codebook variants (the proxy, stage 2, stays a screen). It changes nothing in RULES-177.md
(trunk levels), which waits for this bench's winners (lead: trunk only after a 2-family transfer check, own amendment).

## Donors (ds8, own tree `/data/q27/w139h/code`)

- Tree = w119's donor tree on ds8 (`/data/q27/heads/w119/code`, = repo ourq.py) + `v3/e3/hash177/ourq-hash.patch`
  (`--codebook hash`) + `v3/e3/hash177/exl3hashm/` (exl3hash, w48, plus an optional 256-entry output map). Same SRC
  (`w119/exl3`), same inputs (R16.npy, hd-R16 H_out, template, bf16 checkpoint) and **w122's exact flags**: `--only-head
  --save-wq --head-perm R16.npy --qtf2 --frac-ext SRC --scaled-hessian --sigma-reg 0.025 --codebook C --hout hd-R16
  --hout-blocks group --sigma-reg-out 0.03`, f16 activations. Only the codebook changes.
- **Gate (before any hash donor is scored):** `--codebook i8` K4 on the patched tree must give tensors byte-identical to
  w122's `q/R16s003/K4` (`q-head.safetensors` and `wq/lm_head.safetensors`, every tensor; metadata aside) and a fit-gen
  KL vector equal to `R16s003-K4.npz`'s. Fail: stop, no hash data is read.
- **Arms at L16** (exllamav3's GPU trellis is 16-bit; L17 / L18 have no GPU quantizer, see below), value = q · 4 k_inv
  (mul1i8's int8 grid and step), S = bytesum(((s · A + B) mod 2³²) & 0x3F3F3F3F), A 0xd0474b, B 0x2545F491:
  - **I** (control): w122's mul1i8 donors `R16s003/K{2.5,4,6}` (existing, not rebuilt).
  - **H**: q = S − 126 (IQR's code with w27's A; = RULES-177's arm H).
  - **HG**: q = MAP[S], Gaussian quantiles of S's CDF, sd matched to H (`mkmap.py hg`; decode: a 256-entry int8 table).
  - **HC**: q = MAP[S], cubic shaping z + 0.02 z³, sd matched (`mkmap.py hc:0.02`; decode: 2 packed FMAs a pair).
  K = 2.5, 4, 6 each (9 donors). Stage-2 iid screen (L16, 512 tiles) for orientation, not a prediction: H ×1.009 /
  1.008 / 1.055, HG ×1.005 / 0.994 / 0.970, HC ×1.006 / 0.996 / 0.979 MSE vs mul1i8 at K2 / 4 / 6.

## Scoring

- score119.py (w119, unchanged) on w119's **fit-gen** dump (deciding) and **search-gen** dump (second slice), exact
  head-only KL of every gen row, `exl3:wq=…,perm=R16.npy`; ref.npz and the control arms' npz copied from w122's
  `scores/{fit,search}` (identical code, dump and reference; the gate checks one).
- **Statistic:** per (arm, K, slice) R = calib-weighted head-only KL(arm) / KL(I) (report119 `Set.ratio`: per-document
  calib-composition weights, 2,000-resample document bootstrap, seed 0, 95 % CI); unweighted ratio, top-1, rare-script
  and rare-token guard ratios (report119 `Set.guard`) reported beside it.
- **Equal-quality Δbpw:** the K at which the I curve (w122's 17 i8 donors on the same slice, KL log-linear in K between
  grid neighbours) reaches the arm's KL, minus the arm's K (negative = the arm saves bits).
- **Labels** (deciding slice; no automatic adoption: the user decides): "better" if the 95 % CI lies below 1 at a K,
  "costs" if above, "at par" if it contains 1. A K where fit-gen and search-gen point estimates differ in sign beyond ±1 %
  is "slice-dependent". Fast loop: an arm labelled "costs" at all three K is dropped from later variants.

## Later variants (each added here as a dated amendment before its donors exist)

L17 / L18 exact Viterbi (needs our own GPU trellis for L > 16), beam L20, tile CD on the head, per-K constants, libquant
TCQ cross-check, and the head's decode-step time (box2 e2e, R16 head packs, base vs hash builds). Trunk transfer
(ffn_down + attn_qg splice, real 27B KL) only for a winner here, as an amendment of RULES-177.md.

## Cost

9 donors × ~4 min on one ds8 card (locks: card + ram.lock shared, w122's), scoring ~10 s a donor. Runtime cost of what
it measures: decode −4 to −10 % (stage 1; HG / HC add a table / 2 FMAs a pair, to be timed); head-only rows, no ladder.
