# 27B swj (C10: gate / up output scales refit through the SwiGLU, codes fixed): swap-test pass rule

Written 2026-11-05 ~9:55 PM CST, before any 27B swj data. Tools, gate and tests: `v3/yaqa/swj27/README.md`.

**Question.** Does the 0.8B's swj gain transfer to the 27B?
- 0.8B, plain base: K3 −3.1%, K2 −8.6% (#85).
- 0.8B, on edy H_out (#85-03): K3 0.969 / 0.972 / 0.970, K2 0.910 / 0.914 / 0.912.
- It is part of the adopted stack (`edyswjasym_r0`).

swj changes only the f16 svh vectors of gate_proj and up_proj, on frozen codes, so this test needs no requantization.

## Arms

All arms are K2.5 (exl3i, mul1i8), V2 3.00's head and embedding, served by radiance 1.16.0 with our plugins (swap27.sh's
home, the same build for every arm). Scoring: `golden-search27` (search pool, 103 docs, 123,460 positions, 8k cap),
bf16 KV.

| pair | unedited arm (existing score) | swj arm (new) |
|---|---|---|
| seed 0 | `base` = `/data/v3ladder2/K2.5` (no H_out) | `baseswj` = base with swj scales |
| seed 1 | `s1` = `/data/swap27/q-s1/K2.5` (no H_out, rotation seed offset 1) | `s1swj` |
| report only | `bn`, `bns1` (bn H_out, seeds 0 / 1; bns1 from tonight's q1106) | `bnswj`, `bns1swj` |

- **Scales.** Each swj arm's scales are fitted on its own level (its own codes, decoded from the shards), with
  swiglufit.py's arithmetic:
  - grid c_g ∈ 0.6..1.4 by 0.01, c_u in closed form, both clamped to [0.5, 2];
  - target a = silu(W_g x)(W_u x) from the bf16 27B's MLP inputs.
- **Fit data: a qad-pool slice.** calib_slice.py, pools-v3, 1,048,576 tokens asked → **725 docs, 927,611 tokens**
  (category-proportional, documents cut to 4,096 tokens, seed 0).
  - *Why qad, though the 0.8B used fit (g08-fit1):* swj fits 2 × 17,408 scales a layer (2.2M parameters) to
    activations, which per AGENTS.md belongs on qad.
  - It also keeps the fit pool's later kfit2 sensitivity measurements on swj'd levels out of sample.
  - The scoring pool (search) is disjoint from both.
- **Token count, 9× the 0.8B's 100k.** The 27B's MLP activations are heavy-tailed: mkhin27 saw down_proj input
  diagonals still noise-dominated at 0.2M tokens. The held-out columns of `swiglu.jsonl` (fitted on even documents,
  scored on odd) show whether the fit generalizes.

## Rule

Let r0 = KLD(baseswj) / KLD(base) and r1 = KLD(s1swj) / KLD(s1), each pair paired per document (`tools/paired.py`).

- **PASS:** √(r0 · r1) ≤ 0.98 **and** r0 < 1 **and** r1 < 1. `rule.py` prints the verdict.
- **Why two seed pairs, although each pair shares its codes.**
  - Within a pair there is no quantization noise in the contrast: the same trellis, only svh differs. So the
    per-document interval is the right interval for *that* pair.
  - But the gain of a scale refit depends on the particular codes. H27 amendment 4 saw the 27B's seed-to-seed
    variation (base → s1 0.975) as large as the effects being tested.
  - Two seeds decide whether the setting transfers, not one quant.
- **FAIL:** either ratio ≥ 1, or the geometric mean > 0.98.
  - swj is then not adopted from this test. Log it in BACKLOG "proxy negatives to reconsider", with the mechanism to
    look at: the swiglu.jsonl held-out gains vs the 0.8B's, the c_g spread, and edge hits.
  - If both ratios are < 1 but the mean is > 0.98, log it as "real but below the bar". Revisit only when the step is
    nearly free (folded into a requant's own pass).
- **Worth it if PASS.**
  - Runtime cost: none (svh only; same format, kernels and bpw).
  - Cost: one fit pass per level (below) plus no requant.
  - Adopted as a per-level step after quantization for K ≤ 3.5 levels.
  - K ≥ 4 levels need their own pair first (`ARMS="base4 base4s1"`, V2 4.00, same tool). The 0.8B gain shrinks with K;
    the K4 / K5 effect of swj alone was never measured.
- **Report, not gating.**
  - The bn pairs say whether swj stacks on H_out on the 27B, as it did on the 0.8B (0.969–0.972 on edy).
  - The paired z of each pair.
  - swiglu.jsonl's in-sample vs held-out residual gains per layer, and the c_g mean / sd.
- **Verify:** once, after the decision, on a clean verify slice (`provenance.py`), never iterated on. Not part of this run.

## Commands (gpubox, card 0, after q1106 is done)

From the VM, copy the code (repo layout) to the scratch path the driver expects:

```
cd /workspace/rad-exl3-mix
tar -cf - v3/yaqa/swj27/*.py v3/yaqa/swj27/*.sh proxy2b/common.py proxy2b/exp/w02-fixedscale/fsutil.py \
  volunteer/checker/exl3dec.py mix/calib_slice.py | local/bin/ssh-gpubox.sh 'mkdir -p /data/q27/w36/code && tar -C /data/q27/w36/code -xf -'
```

On gpubox:

```
# the gating pairs (+ the H_out pair in the same fit pass if q1106 has made /data/swap27/q-bns1/K2.5 by then)
cd /data && nohup env ARMS="base s1" bash /data/q27/w36/code/v3/yaqa/swj27/swapfit.sh > /data/swj27.out 2>&1 < /dev/null &
#   ARMS="base s1 bn bns1" for all four; adding arms later refits only the new ones (own scratch)
#   steps one at a time: ... swapfit.sh meta plan | fit | apply | arms | rule  (each idempotent; rerun resumes)
tail -f /data/swj27/result.txt          # the log; fit progress: /data/swj27/fit-base-s1.log
```

- **Locks.**
  - `fit` holds `/root/card0.lock` and `/root/ram.lock` shared for the whole pass.
  - `arms` takes conv.lock / ram.lock for each pack and card0.lock for each score, as swap27k.sh does.
- **Kill and resume.** A killed fit resumes from its last checkpoint (≤ 10 min lost).
- **Ledger.** After `meta`, add the slice as a calib use: add a `swj27-qad` role to `eval/pools/ledger-roles.json`
  (suggested: `train`, line `v3`; 2.2M scales fitted to its activations), then:
  `eval/pools/ledger_add.py eval/pools/ledger-roles.json eval/pools/pools-v3.json eval/pools/ledger.jsonl <copy of /data/swj27/swj27-qad>`
  (only kld.json is read).
- **Raw results.** Copy `result.txt`, `fit/*/swiglu.jsonl`, `fit/*/swjfit.json`, `fit-*.log`, `scr-*/timing.jsonl` and
  `*swj.json` to `data/swj27/`.

## Cost estimate (gpubox; "base s1" unless stated)

Measured inputs:
- the gpubox mkhin27 run tonight: 63–69 µs per position and layer for a bf16 forward plus 0.79 GFLOP of f32 hooks, so
  f32 GEMMs run at ~17.5 TFLOPS there;
- the 4070 bench of the fused grid: 6.4 µs at this shape (torch path: 203 µs).

Per position and layer: bf16 forward ~20 µs, plus the f32 target GEMMs (0.36 GFLOP, ~20 µs), plus per source its
f32 GEMMs (~20 µs) and the fused grid (~5 µs). The first stage's tok/s in `fit-<arms>.log` gives the real total:
≈ stages × 927,611 / tok/s.

| item | base s1 | base s1 bn bns1 | basis |
|---|---|---|---|
| fit (card 0) | **~1.5 h** (1.2–2 h) | ~2.3 h (1.9–3.2 h) | 91 / 142 µs × 64 layers × 927,611 positions; `FUSED=0` would take ~4–6× as long; `TOK=524288` halves it |
| stage plan | 6 stages, 21.2–23.4 GiB resident | 8–11 stages, ≤ 23 GiB | `swapfit.sh plan` (run: 2 arms 6 stages; 3 arms 8 × 8 layers at 22.6 GiB) |
| VRAM peak | ≤ ~26.5 GiB | ≤ ~26.5 GiB | resident + ~3 GiB at a 4,096-token document; VRAM_CAP_GB 30 |
| host RAM | ~3–4 GiB RSS | same | one layer's bf16 tensors on their way to the card (mkhin27: 4.2 GiB); apply ~1 GiB; packs as swap27k (memguard 2.5 GB) |
| apply (CPU) | ~2 min an arm | | 64 shards read + written (7.1 GB) |
| pack (CPU, conv.lock) | ~27 min an arm | | swap27k's packs: 1,607–1,706 s |
| score (card 0) | ~3 min an arm | | swap27 scores: 138–161 s |
| disk, peak | ~70 GB | ~110 GB | boundaries 2 × 9.5 GB, sums 45 MB a layer and arm, per arm: level 7.1 GB, pack 8.8 GB (transient), mix 8.8 GB; /data has 3.2 TB free |
| wall time | **~2.5 h** (card 0 ~1.6 h) | ~4.5 h | |

Clean up after the result is logged: `scr-*` (sums and boundaries), `lv/*` (edited levels; recreated by `apply` in
minutes), the mixes once no further pairing needs them.

## Runtime cost (user rule, 2026-11-03)

None. Only the f16 svh vectors of gate_proj and up_proj change. The format, kernels, bpw, prefill and decode are
unchanged.

## Amendment 1 (2026-11-06 ~10:15 AM CST, after the K2.5 pass, before these data): stacking with H_out bn, K4 transfer
One fit pass, `ARMS="bn bns1 base4 bn4"` (gpubox insert `/data/q1106/insert-20-swj-bn-k4.sh`, before the next q2 arm).
- **Stacks with H_out bn (K2.5):** pairs bn → bnswj, bns1 → bns1swj; the same rule (geo ≤ 0.98, both < 1). A pass means
  the swj gain survives on top of H_out; a fail with geo in (0.98, 1) means it overlaps H_out's gain (then the stack
  decision follows whichever of H_out / swj the pooled pairs favor).
- **K4 transfer:** pairs base4 → base4swj, bn4 → bn4swj (two different bases rather than two seeds; base4s1 isn't made
  yet). Pass at K4 = geo ≤ 0.98 and both < 1 → swj at every level ≥ K4 too; geo in (0.98, 1) with both < 1 → adopt at
  ≥ K4 only if the fit cost stays "one pass per level" (it does); any pair ≥ 1 → swj only ≤ K3.5 until a K4 seed pair.

## Amendment 2 (2026-11-06 ~6:30 PM CST, w59, before any verify data): K4 confirmation on a fresh verify slice
Tools, slice and run: `v3/yaqa/verify27/README.md` (gpubox insert `/data/q1106/insert-40-verify27.sh`, driver `v27.sh`).

- **Confound found in amendment 1's K4 pairs (from the census files, no new data).**
  - base4 / bn4 were mixed into **V2 3.00's** head (exl3 K4, 0.636 GB) and embedding (0.497 GB). `swap27k.sh` sets
    `V2=$3`, then the `source` of swap27.sh's settings on its next line resets V2 to the 3.00 container.
  - base4swj / bn4swj (swapfit.sh's `V2of`) were mixed into **V2 4.00**: head K5 0.795 GB, embedding 0.656 GB.
  - So 0.7906 / 0.7667 measure swj **plus** a head / embedding upgrade.
  - The K2.5 pairs are unaffected (every K2.5 arm uses V2 3.00; equal sizes).
  - The same bug affects the base4s1 / bn4s1 arms in q2 (V2 3.00 head). Pairs among swap27k arms stay internally
    consistent.
- **Head-matched controls.** `base4v4` / `bn4v4` = base4's / bn4's 336 trunk weights cloned into V2 4.00 (radmix, no
  pack). Census: the same head, embedding and plain weights as base4swj, and the same bytes (13,702,406,144).
- **Slice `golden-verify27`.**
  - Recipe: calib_slice.py, pools-v3 verify, 250,000 tokens asked, cap 8192, **seed 20261107**; cut from the published
    calib golden with golden_pick.py, exactly as golden-search27 (search, 200,000 asked, seed 27; recipe reproduced
    bit for bit). No bf16 card pass is needed.
  - Result: **71 docs, 171,876 tokens, 127,261 scored positions.** The verify candidates run long, so ~100 docs and
    ~120k positions can't both be met; 250,000 is the smallest ask on a 25k grid that reaches ≥ 120k positions.
  - Excluded (`verify27/excl-kld.json`):
    - ver3 (V3 monitor), ver4 (V3 final) and their long traces;
    - the V4 reserve (mkver.py: not before V4);
    - docs trained on (@yaqa, @headsens);
    - g08-verify1 / g08-verify2 (the 0.8B swj verify checks, #93).
  - Every one of the 71 was read before, but only by V1 / V2-line checks and picks (calBbig 42, ver2 38, ver1 14,
    calB 11, long2 2). No V3 setting was judged on them, so a V3 swj contrast isn't biased by them.
- **Rule (K4, gating), on golden-verify27.** Let r0 = KLD(base4swj) / KLD(base4v4) and r1 = KLD(bn4swj) / KLD(bn4v4),
  each paired per document. geo = √(r0 · r1).
  - **Confirmed:** r0 < 1 and r1 < 1 and geo ≤ 0.90. Quote the K4 gain.
  - **Real, smaller than search said:** both < 1 and 0.90 < geo < 0.98. swj stays adopted at ≥ K4 (amendment 1's
    rule), but quote the verify number, not 0.79.
  - **Not confirmed:** a pair ≥ 1, or geo ≥ 0.98. swj at ≥ K4 falls back to amendment 1's "any pair ≥ 1" branch until a
    K4 seed pair settles it.
- **Report only.**
  - Verify: base4 → base4swj and bn4 → bn4swj (amendment 1's pairs as built, head confound included); base4 → base4v4
    and bn4 → bn4v4 (the head / embedding change alone); K2.5 base → baseswj and s1 → s1swj.
  - Search (golden-search27): base4v4 → base4swj and bn4v4 → bn4swj, which is amendment 1 corrected for the head.
  - Every arm: mean KLD, p99 KLD and top-1 agreement on the scored rows (tails rule); every pair: ratio, bootstrap
    95% CI, z, p99 ratio, top-1 difference.
- **Cost.** Card 0 ~35 min (12 scores at ~140–160 s each, the measured swap27 / swj27 scores). The golden and mixes
  took < 2 min of CPU in the dry run; disk ~0.7 GB (both mixes are reflink clones).
- **Runtime cost:** none (svh only, as above). The head-matched controls change nothing shipped.
