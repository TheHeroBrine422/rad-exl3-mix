# v3/qad196 — QAD pilot: restricted factors, RMSNorm gains, un-shrink (#196, RULES196)

Eleven arms (2.5: A, R, U, UR, RN, AN, UA; 4.0: A4, R4, U4, UR4 — plus the untrained controls) on the S-only recipe of
#191 (`v3/qad191/`). Rules: `RULES196.md` (commit 22306e58). Pins: `plan196.json` (commit 117b09f8, before any run).
Tracker: INDEX / LOG #196. Runtime: none (scales and norm gains fold into the format).

## Files

| file | what |
|---|---|
| `qad196.py` | #191's trainer (copied) + `--factors all|attn_ssmout` (R: row scales only on attn_qg/k/v/output and ssm_out; the others frozen), `--norms` (N: RMSNorm gains, hook form as qmodel's arm C, on input/post-attention/q/k/GDN norms and the final norm; lr/10; exported to `norms.safetensors` in the checkpoint's own form), `--unshrink FILE` (U: start value of each listed output-side log multiplier; a frozen parameter where the output side is not trained), `--export-only` (U arms: the start-state checkpoint, no training). All switches off = #191 arm A (unit check: the first 4 steps identical, rows, lr and train_kl). |
| `unshrink196.py` | the closed form: c_r = sqrt(w_r H w_r^T / q_r H q_r^T), clamped to [0.9, 1.2], on attn_output, ssm_out, ffn_down, attn_v, ssm_qkv. `--selftest KEY`: the served-row reconstruction vs `QLinear.forward` (rel. err ~0.4 %, bf16 rounding); `--force-one`: c = 1 (the identity check). `--dev cuda:N` for the f64 products on a card (ffn_down). |
| `ushr196.sh` | the un-shrink per linear type (one process each) and the merged `U<TAG>.safetensors` (TAG 25 = 2.5 level, 40 = 4.0 level). |
| `patchnorms196.py` | the served base with the trained norms: a copy of the base (plain, no clone on box2's /data) with the 209 norm planes overwritten in place (bf16), read back. Container names: `blk.N.attn_norm` (input_layernorm), `ffn_norm` (post_attention), `attn_q_norm` / `attn_k_norm`, `ssm_norm` (GDN gated norm), `output_norm`. Identity norms → byte-identical base (`cmp`, rc 0, 11-15). |
| `eval196.py` | #191's served path (`eval191.py`) + the norm base: a checkpoint with `norms.safetensors` gets its patched base as the mix's and census's base. |
| `tscore196.py` | the torch path (A8, as served) of a checkpoint; replays the switches saved in the checkpoint's args (`apply196`) before loading its state. |
| `analyse196.py` | per arm served KL (x181g, fe38; pooled, per set); contrasts with per-document bootstrap CIs (2,000 reps, seed 0; both sets resampled jointly for all arms). |
| `run196.sh` | one training job on one box2 card (flock on the card lock; exit 75 when busy). `NOHEAD=1` (no torch head: the 4.0 student) and `EMBX=` (the 4.0 token table) override the 2.5 pieces. |
| `q196.sh`, `jobs-card{1,2,3}.txt` | the per-card queues (restartable; lines `NAME|LV|NOHEAD|EMBX|FLAGS|DEP`; `@cpu:` / `@gpu:` lines for the merges and the GPU un-shrink). |
| `s196.sh`, `jobs-served.txt` | the served queue on card 0 (the search's eval host shares card 0; flock). |
| `tcheck196.sh` | the torch path's rows on one judge set (for the served-vs-torch check). |
| `gpu3-ffn196.sh`, `ushr196.sh` | the GPU un-shrink on card 3 (ffn_down). |
| `RULES196.md`, `plan196.json` | the rules and the pins (students, hashes, arms, judge sets). |

## Students
- **2.5** (as #191): `vl/inc2.5m` (virtual level of `v3s183-2.5-inc-01334`), K2.875 torch head (R16 perm), t1.75 token table, T4 GDN a/b. Served: template `v3s183-2.5-inc-01334.mix`.
- **4.0** (`v3s183-4-inc-02362`, hash 481c35dc10881175, the last 4.0 incumbent event at 11-15 6:44 PM CST): `vl/inc4.0m` (vlevel.py, 336 trunk weights from 23 levels, 2 min); torch head = the **checkpoint's bf16 lm_head** (`NOHEAD=1`: the 4.0 head is the RAD donor `hd-u3.75.rad`, whose record plane is not the level's trellis layout; no torch head yet); token table = t3.375 (emb191.py oracle decode of `t-i8/K3.375/src`); GDN a/b as 2.5. Served template: 2.5's with the head line `hd-u3.75.rad` and the table line `emb-t-i8-K3.375.rad`.

## Running (box2, `/data/q27/w182`)
```
source env.sh                                              # w182 env (2.5 pieces from w170)
bash code/v3/qad196/run196.sh A 1                          # one arm on card 1 (locks held)
LV=$W/vl/inc4.0m NOHEAD=1 EMBX=$W/emb/t3.375.bf16 bash code/v3/qad196/run196.sh A4 2
code/v3/qad196/ushr196.sh $VL 25                           # un-shrink (CPU per type; TYPES=ffn_down DEV=cuda:0 on a card)
code/v3/qad196/q196.sh 1 jobs-card1.txt                    # card queues (restart by rerunning)
code/v3/qad196/s196.sh jobs-served.txt                     # served queue
python3 code/v3/qad196/analyse196.py served out/analysis   # readout
```
