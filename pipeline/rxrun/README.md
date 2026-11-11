# rxrun — the local ladder runner (w105, #143)

Runs the V3 trunk ladder (`rxq.ladder.levels`: 50 levels at stamp 87a7e4ab) on our own cards, with outputs identical
to `python -m rxq job` processes (the donors' / w92's path). Outside `rxq/` so it does not move rxq's code stamp.

| file | what |
|---|---|
| `worker.py` | persistent job process: `rxq job` semantics, torch / kernels / engine set up once, specs on stdin |
| `pool.py` | N jobs at once per device (VRAM admission from `rxq.ladder.tensor_vram`, host-RAM guard, OOM → rerun alone), card lock + ram.lock shared per session (`chunk` jobs, then the lock is released so other card-0 work can go), hold file, input prefetch, shared `JobQueue` for several devices |
| `run.py` | the ladder driver (below) |
| `cmp.py` | identity of job / level dirs: tensor bytes, dtypes, shapes, offsets + the metadata dict |
| `seedsd.py` | the proxy's seed SD per family and K band from the duplicate-seed jobs |
| `../tests/check_rxrun.py FORK` | CPU stand-in checks (pool 6, run 5) |

## Run (gpubox card 0)

Stage a tree (as w92: `pipeline store mix v3/e3/{exl3frac,exl3frac2,exl3cb}` + the fork with 0008 / 0009), point a
config at it (`v3/e3/settle/w105/ladder-card0.json`: N = 3 persistent workers, chunk 32, swj in pairs), then

```
setsid nohup env PYTHONPATH=$CODE/pipeline /data/venv-exl3/bin/python -m rxrun.run ladder-card0.json > run.out 2>&1 < /dev/null &
python -m rxrun.run ladder-card0.json --status     # the status line (also ROOT/status.txt)
python -m rxrun.run ladder-card0.json --plan       # levels in order, DONE / todo
python -m rxrun.run ladder-card0.json --estimate [--rate R]   # projected card-days
touch ROOT/HOLD                                    # pause after the running jobs (rm + restart to resume)
python -m rxrun.seedsd ROOT                        # seed SD of the proxy, per family x K
```
Restart = run the same command: DONE levels, finished jobs, assembled levels, swj `.done`, `final/.done` and recipes
are skipped; partial outputs are `.part` (or cleaned by the worker: tensors saved but not logged are redone).
Settings come from the staged settings file; a level's specs are written once (`ROOT/<level>/specs/`): if the settings
stamp changes, start a new ROOT (assemble refuses mixed stamps anyway).

## Per level
specs → jobs (+ duplicate-seed jobs, never assembled) → assemble (CPU lane) → swj level pass (card, `swj_batch`
levels a pass) → post apply (swj + the in-job C9 scales; CPU lane) → recipe → `DONE`. The CPU lane runs while the card
goes on with the next level's jobs.

## Hosts and job kinds (design; only `host: "local"` runs today)
- A device = {name, host, device, lock, ram_lock, workers, vram_gib, max_job_vram, kinds}. Devices of one run share a
  `JobQueue`; a device takes only jobs whose modeled VRAM ≤ its `max_job_vram` (the 9070 XT 16 GB: `max_job_vram` 14;
  down_proj at sub-1 levels models ~11.5 GiB, so every quant job fits; swj (peak ~25 GB) has kind `swj` only on card 0).
- Remote host (the dad's 9070 XT boxes, ~32 GB RAM each): the worker protocol is line JSON on stdin / stdout, so a
  remote slot is `ssh HOST python -m rxrun.worker --device N` with specs naming the host's own paths. The host needs
  the bf16 checkpoint, calib H_in, H_out bn1920 and the staged tree (sha256-checked against the local one: the code
  stamp in every tensor file proves it afterwards; assemble refuses a mismatch). Outputs come back by rsync of the job
  dirs (~1–2 GB a level). Not built: the remote slot, the rsync-back step, host setup.
- Mix evals (pack + radiance KLD on a slice) as a second job kind with the same device config: not built. Open: the
  pack's peak host RAM on a 32 GB box (~22 GB on gpubox) and the eval VRAM fit on 16 GB at our slice settings; until
  measured, packs and evals stay on gpubox.
