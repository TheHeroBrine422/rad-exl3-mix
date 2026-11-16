# v3/qad198 — QAD soup: averaging independent scale-QAD runs (#198, RULES198)

Eval-only soups of the #191 scale-QAD runs (mean of their trained log multipliers; sign consensus for C2), a fresh
seed of arm A (A-s1) and its clean same-recipe soup (S2s). Rules: `RULES198.md` (commit d2d27e53). Runtime: none (the
soup is an ordinary scale checkpoint; the served form is the same format).

## Files

| file | what |
|---|---|
| `soup198.py` | `build OUT {mean,consensus} CKPT... --start CKPT`: writes a trainer-format checkpoint (`scales/<key>.safetensors` = exp of the averaged log multipliers, `params.pt`, `mon.json`, `soup.json` manifest, `.done` last; atomic via `OUT.part`). `unit OUT CKPT`: a soup of one run = that run. `selftest`: synthetic checkpoints (formulas, refusals, skip-when-done). |
| `analyse198.py` | per-arm served KL (x181g, fe38, pooled by gen rows), contrasts with per-document bootstrap CIs (2,000 reps, seed 0, same resample for all arms), `--best X:M1,M2` = X vs its best member, sha256 of each served rows file. |
| `run198.sh` | one arm-A training job on box2 (`--seed 1` for A-s1; everything else = #196 arm A). Card lock held (flock -n, exit 75 when busy). |
| `p198.sh` | the whole chain, restartable: soups → served (SO1 unit gate first, then S2, S4, C2) → A-s1 training → served A-s1 → S2s → analysis. |
| `env198.sh` | box2 env (copied to `/data/q27/w186/env.sh`): w182's #196 env with W = `/data/q27/w186`. |

Served path: `v3/qad196/eval196.py` (w182's #196 code, copied to box2 at commit e22fcae8 with the trainer `qad196.py`).

## Arms

- Members (#191 runs on the 2.5 incumbent `vl/inc2.5m`, `/data/q27/w170/runs/<arm>/ckpt-1e+06`):
  L0B0T0Q0D0 (reference), L0B0T0Q0D1 (other data mix), L0B0T0Q1D0 (a8 student), L0B0T1Q0D0 (taps).
- **SO1**: soup of L0B0T0Q0D0 alone. Unit check (files; served gate: pooled KL within 0.5 % of L0B0T0Q0D0).
- **S2** = mean(L0B0T0Q0D0, L0B0T0Q0D1). **S4** = mean of the four members. **C2** = sign consensus of the S2 pair.
- **A-s1**: arm A (#196 = #191 arm A, lr 1e-3, #181 data mix) with `--seed 1`. **S2s** = mean(L0B0T0Q0D0, A-s1).
- Reference scores: w170 served `L0B0T0Q0D0`, `L0B0T0Q0D1`, `L0B0T0Q1D0`, `L0B0T1Q0D0`, `ctl`; w182 served `A` (N0 = A vs L0B0T0Q0D0).

## Running (box2, `/data/q27/w186`)

```
# box2 (once): copy of w182's code tree + these files; env198.sh -> /data/q27/w186/env.sh; chunks196 -> chunks198
CARD=2 setsid nohup bash ./code/v3/qad198/p198.sh > logs/p198.out 2>&1 < /dev/null &   # restart = rerun the same line
python3 code/v3/qad198/analyse198.py analysis ...   # the readout (p198.sh step 5 has the arguments)
python3 v3/qad198/soup198.py selftest                # on the VM or box2
```

Checkpoints are never copied off box2; the manifests (`soup.json`) record the sha256 of every member's `params.pt`.
