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
