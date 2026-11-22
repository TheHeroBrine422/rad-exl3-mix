# v3/qad218 — code-QAD feasibility probe (#218, w218)

Question: on the released EXL3-V3 2.50 model, does healing that changes the trellis **codes** (PV-tuning style: a KL gradient step on
the codebook values, then a discrete re-encode with our own encoder at fixed scales) move whole-model KL more than scale-only QAD
on the same tensors and data? Report: `rad-local/workers/out/w218.md`. Raw results: `../../data/qad218/`. Runtime cost: none (same format).

| file | what |
|---|---|
| `mkdata218.py` | document split of the calib qad pool: judge `qad218-heldout` (role check, >= 200k gen rows, documents never read by the release's swj / head fits nor by another QAD read slice), tune `qad218-tune` (role select: the eta line search), train `qad218` (the rest, seeded order) |
| `probe218.py` | the torch student (w214's `rad2level214.py` output of the released 2.50 container, no A8 in the loop): `grad` (KL gradient dL/dW_hat of the probe tensors over a round's documents, + per-step samples for the SNR), `eval` (per-document KL of code variants on judge / tune), `scale` (scale-only QAD of the probe tensors, w214's arm A recipe) |
| `encode218.py` | the discrete step: W_t = W_hat - eta / rms(G) G, re-encoded with `exl3v3_tool reencode` (radiance branch `qad218`: `reencode_fixed`, LDLQ in the stored codes' metric Hb S H S Hb + the fp16 tile search, released suh / svh kept); stats per tensor (tiles changed, proxy, record decode == values) |
| `gpu218.sh` | iceman card 0 (03:00.0) runner: card lock, RAM floor, log |
| `agg218.py` | readout: paired per-document ratios with bootstrap CIs, SNR, tiles changed |
| `qerr218.py`, `precond218.py` | the 11-21 redesign: each tensor's quantization step delta (the normalized step `encode218.py --step quant`), the preconditioned direction M_d^-1 G |
| `cos218.py`, `diag218.py` | diagnostics: chunk-to-chunk gradient cosine (the honest SNR); a code's error vs the bf16 weight in the calib metric (`--dump` = reencode inputs of the true weight) |
| `ls218.sh`, `ls2218.sh`, `final218.sh` | the round-0 line searches (normalized plain / preconditioned / from the true weight: `encode218.py --from-orig`) and the judge reads + scale-only |
| `post218.sh`, `swap218.py`, `score218.sh`, `radbuild218.sh` | the served read of a winning round (not run: no round beat rel) |

## Rerun (iceman `/data/w218`; code = this repo at `code/`, the radiance `qad218` worktree's `libquant`, `tests/exl3v3`, `plugin/exl3import` at `rsrc/`)
```
hf download Hero422/qwen3.8-27b-exl3-v3-rad host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad --local-dir rel
python code/v3/qad214/rad2level214.py rel/host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad lv/2.50 --model $MODEL --check 4
(cd rsrc && bash tests/exl3v3/build.sh /data/w218/bin)
python code/v3/qad218/mkdata218.py /data/challenge/golden/calib code/eval/pools/pools-v3.json ledger.jsonl data218.json
G=code/v3/qad218/gpu218.sh; P="/data/venv-exl3/bin/python code/v3/qad218"
# round r (r = 0, 1, 2; CUR = "" for r = 0, else the previous round's chosen eta dir)
$G logs/rRgrad.log $P/probe218.py grad runs/rR/grad --round R --codes CUR
$G logs/rRenc.log  $P/encode218.py runs/rR/enc --grad runs/rR/grad --base CUR --etas 0,0.02,0.05,0.1,0.2
$G logs/rReval.log $P/probe218.py eval runs/rR/tune --set tune --variants cur=CUR,e0.02=runs/rR/enc/eta-0.02,...
$G logs/rRjudge.log $P/probe218.py eval runs/judge --set judge --variants rel=rel,rR=runs/rR/enc/eta-BEST
# 11-21 redesign (round 0 only, after the first design's rounds kept the released codes): code/v3/qad218/ls218.sh; ls2218.sh; final218.sh
# scale-only comparison on round 0's documents
$G logs/scale.log $P/probe218.py scale runs/scale --chunks 0-2 --compile-scales
python code/v3/qad218/agg218.py /data/w218/runs > ../../data/qad218/summary.txt
```
