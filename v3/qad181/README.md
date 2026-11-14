# v3/qad181 — QAD scale-up (#181): arm A at tens of millions of golden positions

#165's arm A (trunk suh / svh log multipliers on frozen trellis codes, the mix's quantized head frozen; `v3/qad/`) trained
for one epoch over the stored bf16 goldens of xcorp, own traces and the calib qad pool, data-parallel, on a searched mix,
with checkpoints at several budgets judged on the served path. **Runtime: none** (scales on fixed codes).

Rules: `RULES181.md` (a623415, amendment 1 359de69: the board's gen rows only). Tracker: INDEX / LOG #181.

## Files

| file | what |
|---|---|
| `qad181.py` | the trainer: `qmodel.Student` (from `v3/qad/`) one whole int8 student a card, torchrun data-parallel (gloo grad sum, identical Adam steps, checksums at checkpoints); stream plan from `data181`; segments carry the student's KV / GDN / conv state (`SegCache`, lower-right causal sdpa), a missing state is rebuilt by a no-grad prefix forward; WSD schedule with decay branches (ckpt-T = a one-epoch run of budget T); per-step running training KL per corpus (EMA), monitor reads; `--genmask DIR`; `--gate` (carry exactness, timing, memory), `--profile` (torch.profiler); OUT/STOP = save and exit |
| `data181.py` | goldens of any top-K (`Gold`: K from kld.json, optional gen-mask file), `Corpus` (chunks + the calib pool, items cached), `plan()` (V streams × Q positions a step, documents to CAP tokens, independent of the card count) |
| `genmask181.py` | the challenge's gen mask rebuilt from tokens (README "The score"): `check` vs calib's official gen.u8 (0 of 10,254,542 rows differ), `make GOLDEN OUT.u8 TOKENIZER` |
| `vlevel.py` | a searched mix's trunk as one virtual level (each tensor copied from the level the search picked) + its per-tensor rad-convert recipe; `--uniform LEVEL` |
| `eval181.py` | served-path judgement: scales → `rxq.swj apply_level` → pack (per-tensor recipe) → the incumbent's served form (hostd .mix + a8g128 + bf16 embedding + its R16 head) → census → radiance scores; `cmp` |
| `evals181.sh` | a host's judgement queue on one card (`MIX:TAG[:RUN/ckpt-T]`, `gate:MIX`) |
| `curve181.py` | readout: every checkpoint vs control and vs the previous one (paired CIs, the rule), position bins, training-curve summary |
| `run181.sh` | one training run on a host's cards (card locks held, restartable, movable between hosts) |
| `mkx181.sh` | the fresh xcorp judging part `golden-x181` (ds8) |
| `mksub.py`, `tscore181.py`, `bf16ref.py` | checks: sub-goldens, the student's per-row KL on a golden, an independent bf16 reference (layer-streamed torch forward; one job at a time per host: ~52 GB file-backed RSS) |

## Running (box2 / ds8, `/data/q27/w154`)

```
source env.sh                                           # fla 0.6.0 pylib, staged rxq, model / calib paths
python code/v3/qad181/vlevel.py vl/inc2.5 --ladder /data/q27/ladder-src --assign assign/v3s171-2.5-block-00626.json
python code/v3/qad181/genmask181.py make gold/xq-c00 genmask/xq-c00.u8 $MODEL      # every chunk + golden-x181
bash code/v3/qad181/run181.sh inc2.5g vl/inc2.5 heads/K2.75 0,2,3 --genmask genmask --budgets 1e6,4e6,16e6,40e6
bash code/v3/qad181/evals181.sh 0 inc2.5:ctl inc2.5:a1m:inc2.5g/ckpt-1e+06 ...
python code/v3/qad181/curve181.py eval/inc2.5 ctl a1m a4m a16m a40m --run runs/inc2.5g
```

## Numbers (R9700, 210 W)
One 1,024-token segment fwd + bwd 3.9 s (264 positions/s); GPU-bound: GEMMs 46 % of device time (~96 TFLOP/s),
int8 → bf16 / f32 ↔ bf16 copies 21 %, f32 scale multiplies 13 %, GDN 8 %. Peak 29.4 GiB at an 8k prefix. Data parallel:
3 cards 664 tokens/s (~590 gen rows/s).

## Results (LOG #181, 11-13 → 11-14)
- **Gen rows only.** The xcorp / trace goldens record every row; the board scores only the tokens the model writes
  (`gen.u8`). Prompt-side rows are arbitrary in any bf16 implementation and hold 73–83 % of the search slices' x-part KL.
  `genmask181.py` rebuilds the board's mask from tokens (exact on calib's 10.25 M rows); training and judging use it.
- **2.5 incumbent curve** (served path, healed / control): verify 0.937 / 0.933 / 0.943 at 1 / 4 / 16 M gen rows,
  fresh xcorp 0.884 / 0.835 / 0.808. Calib-like text saturates by 1–4 M; xcorp keeps gaining (the mix is 80 % xq).
- **Screen at 1 M** (held-out qad eval): lr 1e-3 + mixed data best (0.947 vs base); calib-only data, lr 3e-4 / 1e-4 /
  3e-3 all worse (CI > 1 vs the best). **4.0 / 6.0:** no lr gains at 1 M (best 0.993 / 1.009 vs base).
- **Throughput:** ~225 tokens/s a card in data-parallel (264 alone; ~1.18 tokens a gen row); a budget of 1 M gen rows ≈ 1.5 R9700-h, 4 M ≈ 6,
  16 M ≈ 23 (training only; a served-path judgement ≈ 0.2 card-h + ~10 min CPU).
