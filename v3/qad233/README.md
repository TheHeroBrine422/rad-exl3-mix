# v3/qad233 — QAD trainer performance pass (w233): stage processes, host-side syncs, host memory

Report: `rad-local/workers/out/w233.md`. Design: `DESIGN-stages.md`. Raw data: `../../data/qad233/`.

## What changed in the trainer (`v3/qad220/qad220.py`, `v3/qad/qmodel.py`, `v3/qadq/qrun.sh`)

All of this is behind flags, except two exact host-side changes that are on by default:
- the token table sits in host RAM (one copy per machine in /dev/shm) and rows go to the card pinned and asynchronously;
- the KL targets go pinned and asynchronously.

| flag | what |
|---|---|
| `--pipe-procs [S]` | The pipeline as S processes, one card each. qrun.sh launches them for a card group `A,B` / `A,B+C,D` / `A,B,C,D` (QAD_STAGES=S, rank = S × group + stage). Stage 0 holds the token table and its layers; the last stage holds its layers, the head and the loss. Hidden states and gradients go over gloo point-to-point links on threads. Each stage carries, parks and budgets its own layers' states. It is bit for bit today's `--pipeline` (teststage233 on CPU; eqS / eqS4 on GPU, 20 steps). |
| `--stage-bounds B1,..` | The S − 1 layer boundaries. For S = 2 it defaults to `--split-layers K`. |
| `--state-early` | A segment's carried state is stored right after its forward, so a stream's next segment does not wait for the backward. Exact. |
| `--park-pool-gb G` | Parked states go into one exact-size, page-locked pinned pool per process. It avoids the caching host allocator's power-of-two rounding and keeps host memory fixed. A park that does not fit goes to pageable memory and is counted as a pool miss. |
| `--pipe-ahead N` | Extra in-flight segments per stage beyond S − stage (no effect measured). |
| `--budget-evict-first` | Park other streams' states before finishing in-flight segments (no effect measured). |
| `--kv-full` | KV buffers at the document's full length from its first segment (static sizes, no growth copies). Exact. |
| `--keep-act LAYERS` | Those layers keep their activations instead of being recomputed. Exact, but ~0.45 GiB a layer a segment, so only a few layers fit. |

- Telemetry with `--pipe-procs`: `pipe_s` a_* / b_* / c_* / d_* per stage, `mem` / `state_peak_gib` / `states_gib` per card (cuda:<stage>),
  `host_stages` (RSS anon / file / shmem, pinned, parked, pool peak / misses), `rank_wall`.
- `W233_TL=1` writes a per-segment GPU / host timeline (`tl233.py` reads it).
- Environment: `TORCHINDUCTOR_COMPILE_THREADS=1` (qenv.sh, run233.sh). Inductor's default pool is 32 compile-worker processes per trainer,
  ~16 GB of host RAM.

## Tests (run on box2's CPU: `cputests233.sh CODE OLD_QAD220 OUTDIR`)
- `test233.py [cuda:0|cpu]`: the RAM table / pinned targets = Student.embed / Gold.targets, bit for bit.
- `testhf233.py`: `--keep-act` = checkpointed, bit for bit.
- `teststage233.py`: `--pipe-procs` 2 and 4 = `--pipeline`, bit for bit. Cases:
  - plain, budget (+ evict-first), accum, depth test;
  - `--state-early`, `--kv-full`;
  - resume across layouts, fixed eval, checkpoint files;
  - 2 dp × 2 stages, branch training.
- `test-qrun233.sh`: qrun.sh process layouts.

## Tools
- `prof233.py`: a one-card segment profile.
- `pipe233.py`: qad220 with profiled steps (`W233_PROF_STEPS`).
- `mkstate233.py`: resume a plan mid-way (deep windows).
- `cmp233.py`: two runs' loss curve and parameters.
- `logstat233.py`: wall vs elapsed and pipe_s.
- `tl233.py`: timelines.
- `run233.sh`: a lock-waiting runner (stage groups).
- `chain233*.sh`: the GPU blocks, one card-lock holder each.
- `specs/`: plan231 specs (fin12 = one finals rank on a pair, fin24 = the finals' two ranks).
