# rxrun — the ladder runner (w105, #143; multi-host w120)

Runs the V3 trunk ladder (`rxq.ladder.levels`: 50 levels at stamp 87a7e4ab) on our own cards, with outputs identical
to `python -m rxq job` processes (the donors' / w92's path). Outside `rxq/` so it does not move rxq's code stamp.

| file | what |
|---|---|
| `worker.py` | persistent job process: `rxq job` semantics, torch / kernels / engine set up once, specs on stdin |
| `pool.py` | N jobs at once per device (VRAM admission from `rxq.ladder.tensor_vram`, host-RAM guard, OOM → rerun alone), card lock + ram.lock shared per session (`chunk` jobs, then the lock is released so other card-0 work can go), hold file, input prefetch, shared `JobQueue` for several devices |
| `run.py` | the ladder driver (below) |
| `cmp.py` | identity of job / level dirs: tensor bytes, dtypes, shapes, offsets + the metadata dict |
| `seedsd.py` | the proxy's seed SD per family and K band from the duplicate-seed jobs |
| `multi.py`, `hostd.py` | the multi-host queue (below) |
| `../tests/check_rxrun.py FORK` | CPU stand-in checks (pool 6, run 5, multi, evalstall 3) |

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

## Multi-host (w120, 11-12): `multi.py` + `hostd.py`
One queue over several hosts (gpubox card 0 + box2's 4 cards). `python -m rxrun.multi CONF [--follow]` runs on the VM
(the only place that reaches box2: the user's reverse forward 127.0.0.1:2241) and starts `python -m rxrun.hostd
HOSTCONF` on each host through a plain ssh line (stdin must pass through: not `ssh-box2.sh`). Light on the VM: JSON
over ssh, a state file; all work happens on the hosts.

| unit | what | where |
|---|---|---|
| level | `unit_levels` consecutive ladder levels (host's `rxrun.run` conf restricted to them; one swj pass for the unit) | any host whose limits (`k_min` / `k_max`, `max_unit_est_h`) admit them; each level built on exactly one host |
| eval | pack (rad-convert of each level's `final.recipe`, kept) → mix (radmix over a base container) → radiance KLD on each slice part | the host that holds every level of the mix (level dirs stay where they were built; box2's uplink is not ours). Mixes over levels on two hosts are reported BLOCKED |

- **Same staged tree:** each hostd says hello with the settings stamp and a tree stamp (staged sources + the fork's
  kernel sources, hipify outputs skipped); the coordinator refuses a host whose stamps differ from the first host's /
  CONF `expect`. `python -m rxrun.hostd HOSTCONF --stamp` prints them (rollout step 2).
- **Survives a dropped ssh:** every unit runs in its own session (`hostd --unit DIR`); the next hostd reports it running
  or finished and the coordinator (restarted from `ROOT/state.jsonl`) picks the result up; an in-flight unit is never
  given to another host. A unit that died without a result is retried once on the next free host (`retries`).
- **Results** (small, copied back in the JSON reply): `ROOT/results.jsonl` one line per eval × slice (tag, bytes,
  bpw, positions, kld, p99, top1, slice, host, id, t; failures: tag, fail, err), `ROOT/docs/<tag>-<slice>.json`
  (per-part per-document KL sums: `eval/docstat/readout.py`'s inputs); rows files stay on the host.
- **Eval queue:** the search appends `{"tag", "mix": {"base", "trunk" | "parity" | "levels", "donors"}, "slices":
  [{"name", "parts": {"x": GOLDEN, "c": GOLDEN}}]}` lines to CONF `evals`; names in HOSTCONF `eval.aliases` resolve per host.
- Configs: `hosts/gpubox-card0.{run,host}.json`, `hosts/box2.{run,host}.json`, `hosts/ds8.{run,host}.json` (8 cards; w121) (host side), `hosts/multi-coord.example.json`
  (coordinator: copy to `local/`, it holds ssh lines). Checks: `tests/check_rxrun.py FORK WORK --part multi` (CPU
  stand-in: two local hostd hosts + one with another tree; stub pack / mix / score).

**Evals beside level work (w123, 11-12):** an eval's pack / mix used to hold ram.lock exclusive: every card worker of
the host then waits at ram.lock (shared) holding its card lock, and the CPU lane's assemble / post too -- box2 + ds8 froze
11-12 15:14–15:51 CST. HOSTCONF `eval.ram_mode` `"shared"` (box2, ds8) takes ram.lock shared after a RAM admission
(`pack_ram_gib` 16: the k3 pack's anonymous peak is ~4 GiB, its 30-40 GB RSS is the mmap'd bf16 source; `mix_ram_gib`
4); conv.lock keeps one pack / mix a host; default `"exclusive"` (gpubox). `eval.goldcat` scores a slice's parts in one
run (`goldcat.py`, rows byte-identical); `eval.cards` [{card, lock}] scores on the first free card. Check:
`check_rxrun.py --part evalstall` (the old rule reproduces the stall, the shared one does not).

**Pause one card (user 11-12):** `touch <run root>/HOLD-<device name>` on the host (e.g. `/data/q27/ladder/HOLD-c3`): that card finishes its running jobs, releases its card and ram locks (other work can take it by the lock) and idles; the host's other cards keep going; the swj pass uses an unpaused swj card. `rm` the file to rejoin. Whole host: `HOLD`; whole run: the coordinator's `ROOT/HOLD`. Check: `check_rxrun.py --part pool` (dev-hold).

**Rollout (lead):** (1) stage the frozen commit on both hosts (gpubox `/data/v3run/code` + the fork; box2
`/data/q27/stage/code`, restaged from 4156463 to HEAD); (2) `hostd --stamp` on both: settings and tree stamps equal;
(3) box2: build `radmix` (`mix/build.sh`) into `/data/q27/tools`, copy the search slices' goldens (x parts too: box2
counts as ours since 11-12), the V2 reference containers; (4) copy the host confs to `<root>/multi/{run,host}.json`;
(5) on the VM: `setsid nohup python -m rxrun.multi local/.../multi-coord.json --follow > multi.out 2>&1 &`
(`--status`, `touch ROOT/HOLD`). If box2's ssh is refused on 127.0.0.1:2241 the forward is down: the coordinator keeps
gpubox going and retries box2 every `reconnect_s`; the user restores the forward (AGENTS.md "box2").
ds8 (w121, 11-12): staged at a84ed23 with `/data/q27/multi/{run,host}.json` (rollout steps 1, 2, 4 done there; tree stamp 8557ed25d9bdc6a6; step 3: `tools/radmix` + V2 3.00 only, no search-slice goldens yet); `local/notes/ds8.md`.
Not built: moving level dirs between hosts (an eval whose levels are split stays BLOCKED until someone copies them);
a 9070 XT host (needs its own `max_job_vram` limits; the unit-level split works as is).
