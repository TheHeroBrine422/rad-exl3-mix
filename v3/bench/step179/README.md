# v3/bench/step179 — #179 step pass scripts (w152)

Run from a VM with `rad-local/bin/ssh-box2.sh`; box2 work dir `/data/q27/w152/` (tools copied to `tools/`). Every GPU unit
holds `/data/q27/locks/card<N>.lock` (+ `ram.lock` shared) and skips when its output exists.

| script | what |
|---|---|
| `mk.sh NAME REV [TCS]` | `git archive REV` of the worktree -> box2 `src-NAME`, the five plugins (plugb.sh) for ROCm 10.1 / 7.14, qwen35_exl3's host test (`SEDQ`: a sed on exl3_fuse.h, the negative control; `OVL=1`: build against an install overlay with the tree's `rad_block_dflash2.h`, needed from b83698d) |
| `e2.sh`, `e2sum.py` | w138's bench.py per pack and row (ctx 8k gen 64 + ctx 0 gen 256, warm 1 + 3), summary per round |
| `kld.sh`, `rowcmp.py` | whole-model KLD rows (def = score.sh's A8 chunks, c64 = 64-row steps), byte comparison |
| `gdec.sh`, `gdec.py` | greedy decode, 8 prompts × 96 tokens with top-20 logprobs; token / logprob comparison |
| `stamp.sh`, `stsum.py`, `smallops.py`, `gemvpos.py` | RADIANCE_AQL_STAMP decode timing (WATCHS: cardclock.py alongside; XARGS: more server args), per-kernel tables, the op-class split, GEMVs matched launch by launch |
| `dprof.sh`, `dprof2.py`, `profq.sh` | --profile-ops (synchronised: prefill tables) |
| `gt.sh`, `ctall.sh` | exl3i device tests of a build on 10.1 / 7.14 (w150's list) |
| `diag/` | the timing diagnostic (QWEN35_EXL3_DIAG_SKIP=nongemv; never a served build) |
| `q1.sh` … `q15.sh`, `qnext.sh` | the card-1 queues as run (order and re-plans in their headers) |
| `collect.sh` | the summaries -> `data/step179/` |
| `kctr-key.patch` | exl3i's KS counters keyed by (device, stream, output) — mad decode is NaN without it (PERF.md); applied by the lead as qc-exl3-v3 fdd480c |
| `e2b.sh`, `kldenv.sh`, `q13.sh`, `q14.sh` | the #177-02 re-measure after fdd480c (lead note 11-13 11:00 PM): w150's e2e rows and the decode-path KLD (EXL3I_MAD_SPLIT) per K, the race-free reference on the all-ordered build |
| `fbuilds.txt` | the stash commits the f* / mres* homes were built from (dangling objects: git may collect them). Rebuild instead from commits: mres3 = 1f4ecbc; fres2 = 1f4ecbc + kctr-key.patch; fstep = b73675c + the patch; mfix = 9f0a059 + the patch; mord = 9f0a059 with exl3i `rad_kernel_concurrent` returning 0; mstep = tag archive/w152-step-b02aa9a |
