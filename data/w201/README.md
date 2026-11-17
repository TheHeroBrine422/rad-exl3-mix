# w201 on box2 — the build, the equivalence check, the mad embedding donors (rerun README)

Everything here is under `/data/q27/w201/` on box2. Sources and scripts are copied into the repo at
`rad-exl3-mix/data/w201/` (scripts in `data/w201/tools/`, results in `data/w201/`).

## 1. The build (radiance `qc-exl3-v3-eval`)

Fork branch `qc-exl3-v3-eval` (local, never pushed), created from `qc-exl3-v3` a6a4dcc0:

| commit | what |
|---|---|
| 2b8e5b09 | kld: `--kld-progress FILE` (cherry-pick of kld-progress 8a3c08fe, pruned from qc-exl3-v3 by bc01f7f1) |
| e6552381 | kld: pictures in the KL mode (kld-progress 43a04e00, part of the branch as specified) |
| f1d05aa9 | tests/kld_test: progress lines parsed with rad::Json (kld-progress 2162ce54) |
| 159c2fc0 | #185 RadFile::open takes a manifest (vload 719dae65; conflict in radfile.h, both sides kept) |
| 5156fc0b | #185 is_container + sweep_container through the file's direct readers (vload 64e4487c; conflict in engine_bringup.cpp, ported onto qc-exl3-v3's window pinning: all reads go through the file's reader under a mutex) |
| 8bed46f7 | #185 final/ trunk lines (vload e29c765b, clean) |
| 02bf0d0a | port fix: the reader mutex declared before `fill` |
| d3885b03 | main.cpp: a flag command line runs as the one-model file it is equivalent to (`yaml_from_args` -> `parse`) |
| aee0cdba | main.cpp: `--check` is not a setting of the flag command line |
| 290d30dd | main.cpp: soft `RLIMIT_NOFILE` raised to the hard limit (cap 65536) |

Not taken: vload 64735054 (`rad-info --weights-dump`, a dump tool; not in the score command).

Build (ROCm 10.1 shadow toolchain shared with rad10, `rad21build.sh` = `pipeline/hosts/rad10/rad10build.sh` with this
branch's four plugins exl3quant qwen35_exl3 exl3import exl3i; gfx1201, FFmpeg off):

```
# source: git archive qc-exl3-v3-eval (290d30dd) -> /data/q27/w201/rad21/src (COMMIT file written by hand)
J=16 bash /data/q27/w201/rad21/build.sh /data/q27/w201/rad21     # log: rad21/logs/rad21build.log, ~50 min cold
# after a main.cpp-only change: rad21/rebuild-main.sh (incremental radiance + install)
```
Result: `rad21/install/bin/radiance`, `rad21/home` (RADIANCE_HOME). Plugins in the home: qwen35_exl3.so,
exl3quant.so, exl3i.so (exl3import is a header/common library they compile in). Host tests: exl3quant 1 passed,
exl3i 5 passed, qwen35_exl3 has no host-labelled tests (ctest: "No tests were found").
CPU tests on box2 (GPUs hidden): config_test, fleet_config_test, format_test, kld_test all Passed.

## 2. Equivalence (step 3): the coordinator's score command on s195c2, card 3

```
bash /data/q27/w201/tools/eq.sh     # 3 mixes with score_cmd, then 1 progress run (score_cmd_progress)
#   each run: score-w201.sh BUILD MIX GOLDEN OUTBASE [PROGRESS]
#   = flock card3.lock + flock -s ram.lock; ROCR_VISIBLE_DEVICES=3; radiance --model MIX --tp 1 --kv-cache-dtype bf16
#     --max-model-len 16448 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref GOLDEN --kld-out OUT.part [--kld-progress P]
```
Golden: `/data/q27/w180/slices/golden-s195c2` (the coordinator's `ref_dir`). Old-build numbers are the coordinator's
own scores in `/data/q27/srch183/scores/` (rad10 = radiance 1.17 + vload 64735054; `--max-model-len 16384` in those units'
eval.log, 16448 in the current host.json).

Compare: `/data/q27/venv/bin/python -I /data/q27/w201/tools/compare.py GOLDEN REF_BASE NEW_BASE` (reproduces the
coordinator's slice KLD, `result.json` `kld`, exactly on the old 7.0 run: 0.0006139825390838309).
Progress check: `/data/q27/venv/bin/python -I /data/q27/w201/tools/progcheck.py GOLDEN PROG.progress RUN_BASE`.

## 3. Mad embedding donors (step 4)

Status: all 21 built and verified (`donors/summary.jsonl`, `donors/verify.txt`; `verify-donors.sh` re-checks them).

```
bash /data/q27/w201/tools/mad-donor.sh 3          # then the rest: 1.625 1.5 1.75 1.875 2.25 2.5 2.625 2.75 2.875 3.375 3.625 3.75 3.875 4.125 4.25 4.625 5 5.5 6 6.5
```
Per K: recipe = the nine trunk lines of `finals-mad/q27v3-635bac02-k<K>-mad/final` (imported as trellis, codebook=mad
rate=K; the template is mad-packs' K4 recipe) + `token_embd.weight exl3i src=/data/q27/emb/trellis/mad/K<K>/src
codebook=mad rate=<K>`; `rad-convert --no-error` (no --reuse), then `radmix` (w158) drops every weight but token_embd,
and `embcmp157.py` must report the slim token_embd identical to the full one's before the donor is moved into place.
Output `/data/q27/emb/trellis/emb-t-mad-K<K>.rad`, summary `/data/q27/w201/donors/summary.jsonl`.
Each K takes ~3 min (one convert at a time: conv.lock, ram.lock shared).

## 4. Smoke test (step 5)

`/data/q27/w201/mixes/smoke-2.0-madK1.625.mix` = the 2.0 mix (`p1-v3s183-2-manual-01645.mix`) with line 339 changed
from `emb-t-i8-K1.625.rad` to `emb-t-mad-K1.625.rad` (only difference; diff checked). Census: census_hybrid.py as in the
coordinator's eval.log, PASS (token_embd = exl3i mad K1.625). Scored with score-w201.sh (identical command).
