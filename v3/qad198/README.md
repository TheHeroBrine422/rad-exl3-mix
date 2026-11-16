# v3/qad198 — QAD soup: averaging independent scale-QAD runs (#198, RULES198)

Eval-only soups of the #191 scale-QAD runs (mean of their trained log multipliers; sign consensus for C2), a fresh
seed of arm A (A-s1) and its clean same-recipe soup (S2s). Rules: `RULES198.md` (commit d2d27e53). Runtime: none (the
soup is an ordinary scale checkpoint; the served form is the same format). Result: `data/qad198/` (LOG 11-16 3:06 AM CST and 3:25 AM CST follow-up, #198;
report `out/w186.md`): trigger not met (S2s vs its best member −0.84 % [−2.15, +0.03]); S4 vs its best member −0.94 %
[−2.17, −0.12] with members that share a data order; agreement shrinkage (amendment 1) gives no gain (SH4 vs S4 +0.17 % n.s.) and is worse for the pair (SH2s vs S2s +0.85 % [+0.50, +1.23]).

## Files

| file | what |
|---|---|
| `soup198.py` | `build OUT {mean,consensus,shrink} CKPT... --start CKPT [--device auto\|cuda:0\|cpu]`: writes a trainer-format checkpoint (`scales/<key>.safetensors` = exp of the averaged log multipliers, `params.pt`, `mon.json`, `soup.json` manifest, `.done` last; atomic via `OUT.part`). `unit OUT CKPT`: a soup of one run = that run (state bitwise; scale files compared; `exact` flag). `unitshrink OUT CKPT N`: N identical members, shrink = plain mean, bitwise (amendment 1). `selftest`: synthetic checkpoints (formulas incl. a known sign split, refusals: shape / non-scale / corrupt scale file, skip-when-done). |
| `analyse198.py` | per-arm served KL (x181g, fe38, pooled by gen rows), contrasts with per-document bootstrap CIs (2,000 reps, seed 0, one resample for all arms), `--best X:M1,M2` = X vs its best member, sha256 of each served rows file. |
| `supp198.py` | supplementary: each soup vs the MEAN of its members, from `analysis/perdoc.json`; checks that the pre-registered S2s-vs-ref interval reproduces exactly. |
| `run198.sh` | one arm-A training job on box2 (`--seed 1` for A-s1; everything else = #196 arm A). Card lock held (flock -n, exit 75 when busy). |
| `p198sh.sh` | RULES198 amendment 1 (arm SH, eval only), restartable: unitshrink (N = 2, 4 identical) → SH4 = shrink(S4 members), SH2s = shrink(S2s pair), served → readout over all arms (`analysis-sh/`) + supp198. |
| `p198.sh` | the whole chain, restartable: unit check → REF2 (member re-served) and SO1 (soup of one, served; gate: byte-identical rows) → S2, S4, C2 → A-s1 training → A-s1 served → S2s → analysis. |
| `env198.sh` | box2 env (copied to `/data/q27/w186/env.sh`): w182's #196 env with W = `/data/q27/w186`. |

Served path: `v3/qad196/eval196.py` (w182's #196 code, copied to box2 at commit e22fcae8 with the trainer `qad196.py`).

## Decisions in the code (and why)

- **exp on the GPU.** The trainer writes scales as `exp(state)` on the GPU. On the CPU the float32 exp differs in ~4 % of
  values by 1 ulp, which flips float16 roundings in the served scales (SO1 served KL then moved by up to 0.17 % on fe38).
  On the GPU a soup of one reproduces the member's scale files bitwise (`unit/report.json`: 6,799,360 / 6,799,360).
- **Agreement shrinkage (SH, amendment 1).** Per log multiplier: m · m² / (m² + v/N), m the mean, v the unbiased variance over the N members, 0 where m = 0. Needs N ≥ 2. With identical members v = 0 and the value is m exactly (bitwise unit check).
- **Averaging in log space.** The state holds log multipliers (start 0 = multiplier 1, checked with the start state
  `w182/runs/ONE`), so the mean of logs is `mean(state)`; C2 = start + mean(d) where all members agree in sign of d, else start.
- **Non-scale parameters** must be identical across members (checked) and are copied from the first member.

## Arms

- Members (#191 runs on the 2.5 incumbent `vl/inc2.5m`, `/data/q27/w170/runs/<arm>/ckpt-1e+06`):
  L0B0T0Q0D0 (reference, seed 0), L0B0T0Q0D1 (other data mix), L0B0T0Q1D0 (a8 student, seed 0), L0B0T1Q0D0 (taps, seed 0).
  Note: D0, Q1D0 and T1 share seed 0's data order (sign agreement of moved scales 80–86 %); D1 does not.
- **SO1**: soup of L0B0T0Q0D0 alone (unit gate).
- **S2** = mean(D0, D1). **S4** = mean of the four members. **C2** = sign consensus of the S2 pair.
- **SH4** = shrink(the S4 members); **SH2s** = shrink(D0, A-s1). SHseed not built (no arm-A seeds 2 or 3 exist).
- **A-s1**: arm A (#196 = #191 arm A, lr 1e-3, #181 data mix) with `--seed 1`. **S2s** = mean(D0, A-s1).
- Reference scores: w170 served `L0B0T0Q0D0`, `L0B0T0Q0D1`, `L0B0T0Q1D0`, `L0B0T1Q0D0`, `ctl`; w182 served `A` (N0 = A vs D0).

## Running (box2, `/data/q27/w186`)

```
# box2 (once): copy of w182's code tree + these files; env198.sh -> /data/q27/w186/env.sh; chunks196 -> chunks198
CARD=2 setsid nohup bash ./code/v3/qad198/p198.sh > logs/p198.out 2>&1 < /dev/null &   # restart = rerun the same line
python3 code/v3/qad198/analyse198.py analysis ...   # the readout (p198.sh step 5 has the arguments)
python3 v3/qad198/supp198.py data/qad198/analysis   # supplementary (VM or box2)
python3 v3/qad198/soup198.py selftest                # on the VM or box2
```

Checkpoints are never copied off box2; the manifests (`soup.json`) record the sha256 of every member's `params.pt`.
