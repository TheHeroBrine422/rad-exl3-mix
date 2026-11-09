# w92 · #122 full-stack pass-1 level at 27B (rule `../w92.md`, data `../data/w92/`)

Two K2.5 i8 levels (`q27v3-ed97adb7-k2.5-i8`, seed offsets 0 / 1) built from `pipeline/config/v3.toml` the way the donors
will build them (`rxq.ladder.level_jobs` → the spec rqah's `rxq_quant.stage()` writes → `python -m rxq job`), then
`rxq assemble` → `rxq.swjfit` → `rxq.post apply` → `rxq.mix recipe` → rad-convert → V2 3.00. They were scored against
fresh ladder2 controls and the #104 bn+swj arms on the fresh two-part slice golden-search27c. The K1 i4 arm was
dropped (lead 11-08 5:20 PM CST: i4 retired); a sub-1 job check ran instead.

| file | what |
|---|---|
| `w92.sh` | the driver (gpubox): slice, sub1, fs0 / fs1 / k1 levels, swj, post, packs, hygiene, scores, verdict; every step resumable |
| `mkjobs92.py` | a level's rxq job specs from `rxq.ladder.level_jobs` (group split), gpubox inputs in place of the datapack's |
| `layer92.sh`, `clean92.py` | the job runner (one layer's jobs under card0.lock + ram.lock shared; rqah's unlogged-tensor cleanup) |
| `sub1chk92.py` | K0.875 / K0.125 job check: logged σ per group, H_out blocks, block order, tile CD, C9, finite proxy |
| `verdict92.py` | the gate (pair84.py statistics), `--k1` report (unused: K1 dropped) |
| `e2e92.sh`, `serve92.sh`, `gen92.py` | lead's e2e add-on: A8 score, generation sanity, prefill speed / VRAM vs V2 3.00 |
| `stampchk92.sh` | 3 fs0 jobs re-made with the current trees (rad-exl3-mix 5ce222f, fork b0052f7c): bit-identical? |
| `codestamp-hipify.patch` | proposed fix: rxq's code stamp hashed ROCm hipify outputs (platform-dependent stamps) |

Run (gpubox; stage `git archive <commit> pipeline store mix v3/e3/exl3frac2 v3/e3/exl3frac v3/e3/exl3cb v3/e3/settle/w84`
+ this dir into `/data/w92/code`, the fork's `exllamav3/` (with 0008) into `/data/w92/exl3`):

```
setsid nohup bash /data/w92/code/v3/e3/settle/w92/w92.sh [STEPS] > /data/w92/run.out 2>&1 < /dev/null &
bash /data/w92/code/v3/e3/settle/w92/e2e92.sh; bash /data/w92/code/v3/e3/settle/w92/stampchk92.sh
```

Notes: the bf16 checkpoint, calib H_in and bn1920 H_out dirs are gpubox paths (`/data/workspace/models/qwen38-27b-bf16`,
`/root/challenge/hessians/calib`, `/data/swap27/hout-bn`). PYTHONPATH must put a fork with patch 0008 before
`/data/exllamav3` (that tree lacks 0008; rxq refuses the blocks path without it). Kernels build into
`$A/build` (TORCH_EXTENSIONS_DIR), not the shared `/data/exl3frac/build`. The lead fixed a `local` bug in `post()`
on 11-08 (`local t=$1 F=...$t...` expanded the caller's stale `t`); the pre-fix copy is `w92.sh.pre-fix` on gpubox.
