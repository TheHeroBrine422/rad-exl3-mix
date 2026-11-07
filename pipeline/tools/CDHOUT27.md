# Tile CD with H_out on attn_qg at K4: 27B "does no harm" pairs, rule

Written 2026-11-07 by w76, before any data. Prepared for the lead, not run. Code: `tools/cdhout27.sh` (one arm),
`tools/insert-cdhout27.sh` (both arms + verdict, a card-0 queue insert) and `tools/cdhout27verdict.py`. The pattern is
`v3/e3/settle/CD27.md`'s: the 16 attn_qg tensors of an existing swap-test level are replaced, the rest of the level is
kept.

## Question

Two settings are adopted and now meet in K ≥ 4 levels:
- **Tile CD on attn_qg** (#77). Its 27B no-harm check (CD27) passed at K2.5, without H_out.
- **H_out (bn) at K ≥ 4** (#33-28, H27 K4 pass, 11-07).

Before 11-07, tile CD refused H_out. rxq/tilecd.py now runs CD on the two-sided objective tr(Eᵀ H E M) after
ldlq_2hess. M is H_out in the quantizer's basis: block diagonal, with dense 128 × 128 blocks after the output Hadamard.
The CPU checks pass (`tests/check_tilecd.py`):
- with M = I it is today's CD bit for bit;
- the two-sided objective falls at every sweep;
- the stub run ends lower than the M-free CD on that objective.

This check asks one thing: does cd3 on the 16 attn_qg tensors of an H_out K4 level leave the 27B's KLD no worse? Like
CD27, it is a safety check. attn_qg is 3.7% of the weights, so the gain cannot show. It would catch a broken
implementation or a bad interaction (ldlq_2hess, the refit, rxq on ROCm).

## Arms

Two seed pairs at K4, both H_out levels from H27 (bn, 1,920 sequences, σ_out 0.01, ladder settings: qtf2, scaled H,
σ 0.025 / input 0.005, i8). Each level's settings are read from its own stamp.json by `rxq levelargs`, which refuses a
level that rxq's config cannot reproduce.

| arm | level (every tensor except attn_qg) | attn_qg | control (paired) |
|---|---|---|---|
| `bn4cd` | `/data/swap27/q-bn4/K4` | requantized by rxq: H_out + tile CD 3 sweeps (two-sided) | `bn4` (`/data/swap27/bn4.json.rows`) |
| `bn4s1cd` | `/data/swap27/q-bn4s1/K4` (seed offset 1) | the same | `bn4s1` (`/data/swap27/bn4s1.json.rows`) |

- **No-op gate.** The same 16 tensors are first requantized by rxq without tile CD. They must be bit-identical to the
  level's (`rxq.pairs same`). This also checks rxq against ourq on ROCm with H_out; the w52 check D did this on the
  4070. If the gate fails, the control becomes `<arm>n`: the level plus these no-CD tensors, packed and scored in the
  same run.
- **Hygiene (new, item 7).**
  - `rxq.pairs levels` checks that the assembled level equals the base level outside attn_qg.
  - `rxq.pairs mixes` checks that the packed arm equals the control mix outside attn_qg, weight by weight: planes,
    encoding, options, act mode. If either check fails, the arm is not scored.
  - If the control mix `/data/swap27/mix-<arm>.rad` is gone, the base level is packed again as `<arm>ctl`. That copy
    is scored, and it is the control.
- **Scoring:** as every swap arm (swap27's functions). Pack exl3i mul1i8 rate 4, mix into V2 4.00, census,
  `golden-search27` (search pool, 103 docs, 123,460 positions), bf16 KV, `tools/paired.py`.

## Rule (CD27.md's, unchanged)

Let r_i = KLD(arm_i) / KLD(control_i) for i = bn4, bn4s1, and g = √(r_bn4 · r_bn4s1).
- **PASS** if g ≤ 1.000 and both r_i ≤ 1.01. Tile CD then stays on at every level, with H_out where it applies (the
  settings file as of 11-07).
- **FAIL** otherwise. Tile CD is turned off where H_out applies: set `quant.tile_cd.k_max = 3.875`. A pair above 1.01 also triggers a look for an implementation error: decode the
  arm's attn_qg codes and compare them with the quantizer's returned weights.
- Single verdict, both pairs required. Verify only once after the decision.

**Report, not gating:**
- `q-cd/tilecd.jsonl`: two-sided rel after ldlq_2hess and after each sweep, accepted tile shares, seconds;
- the two-sided proxy_err ratio cd / noop after the refit (`q-*/log.jsonl`);
- quant time with H_out ± CD per tensor on the R9700.

**Runtime cost:** none. Codes only.

## Commands (gpubox, card 0; the lead schedules)

**Stage** the code from the VM: a private tree under `/data/cdhout27/code`. Nothing live is touched.

```
cd /workspace/rad-exl3-mix; I=local/bin/ssh-gpubox.sh; D=$(mktemp -d); mkdir -p $D/v3/e3 $D/store
cp -r pipeline $D/ && cp -r v3/e3/exl3frac2 v3/e3/exl3frac v3/e3/exl3cb $D/v3/e3/ && cp store/census.py $D/store/
find $D -name __pycache__ -prune -exec rm -rf {} +
tar -C $D -cf - . | $I 'mkdir -p /data/cdhout27/code && tar -C /data/cdhout27/code -xf -' && rm -rf $D
```

**Dry run** (prints every card / host command; runs levelargs and the pre checks):
`$I 'mkdir -p /data/cdhout27 && cd /data/cdhout27 && DRY=1 ARM=bn4 BASE=/data/swap27/q-bn4/K4 bash /data/cdhout27/code/pipeline/tools/cdhout27.sh'`

**Schedule** as a queue insert (bn4, then bn4s1, then the verdict):
`$I 'cp /data/cdhout27/code/pipeline/tools/insert-cdhout27.sh /data/q1106/insert-45-cdhout27.sh'`
Use whichever queue dir and number the lead's current card-0 queue uses.

- **Outputs:**
  - log per arm: `/data/cdhout27/<arm>/cdhout27.log`;
  - all arms: `/data/cdhout27/result.txt` (verdict line `CDHOUT27 VERDICT`);
  - scores: `/data/cdhout27/<arm>cd.json(.rows)`; pairs: `<arm>/pair.json`.
- **Raw results to the repo:** `result.txt`, `*/pair*.json`, `*/q-cd/tilecd.jsonl`, `*/q-{noop,cd}/log.jsonl`,
  `*/cdhout27.log` → `data/e3/tilecd/cdhout27/`.
- **Cleanup once logged:** `rm -rf /data/cdhout27/*/level-* /data/cdhout27/mix-*`.
- **Stop:** `touch /data/cdhout27/<arm>/STOP` (between steps).

## Budget (gpubox)

| step | card 0 | CPU / disk | basis |
|---|---|---|---|
| noop quant (16 q_proj, H_out) | ~25–45 min an arm | ~0.4 GB | H_out ≈ 5× a one-sided quant (PROFILE.md, H27: ~3.2 h a K4 level vs 37 min); q_proj ~1.5–2.5 min with H_out |
| cd quant (16 q_proj, H_out + 3 sweeps) | ~35–60 min an arm | ~0.4 GB | + 3 sweeps ≈ 3 one-sided LDLQ passes (w50) + the block-diagonal M terms (~3%) |
| splice + level hygiene | — | 8.8 GB an arm, minutes | one shard at a time |
| pack + mix + census | — | ~27 min an arm (conv.lock, ram.lock exclusive) | swap27's packs |
| mix hygiene | — | ~5 min (reads both mixes) | sha256 of every plane |
| score | ~3 min an arm | — | swap27 |
| **both arms** | **~2–3.5 h** | **~1.2 h** | + a control pack and score per arm if a no-op gate fails or the control mix is gone |
