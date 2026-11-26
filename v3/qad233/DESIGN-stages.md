# w233 design note: stage processes for the two-card pipeline (`--pipe-procs`; approved by the user 11-25 ~10:40 PM CT, building)

## Why
The two-card pipeline (`--split-layers K --pipeline`, w225) runs both cards from one Python process: the main thread (card A), a thread for
card B, and two autograd device threads whose Python callbacks (checkpoint recompute, compiled-function wrappers, Triton launches) share
one GIL. The venv's Python 3.14 has the GIL on.
- w225: GPU work 37 / 40 s a step against 50 s wall.
- w233 fin12 W0 (one finals rank on a pair, finals flags): each card is busy 6.4 / 6.7 s of a 12.7 s step (~50 %). Card A's thread spends
  8.3 s launching 6.4 s of GPU work, plus 4.1 s waiting.
- One process per card gives each card its own interpreter, as in the one-card trainer (GPU-bound, w215).

## Measured since the proposal (fin12 W1 / W2: deep traces at ~120k / ~200k, branch sessions, mixed lengths; `data/qad233/`)
| window | s / step | GPU busy card A / B | GIL held | card A's thread | card B's thread |
|---|---|---|---|---|---|
| W1 (~120k) | 29.8 | 13.3 / 13.9 s (45 %) | 29 % | 37 % waits for B, 23 % parks states (synchronous D2H), 16 % backward launch, 6 % embed | 35 % waits for A, 30 % C++ in side_b (the per-segment `float(kl)` sync, copies), 16 % backward |
| W2 (~200k) | 45.4 | 17.0 / 17.4 s (38 %) | 23 % | 32 % waits for B, 32 % parks states, 12 % backward, 5 % embed, 4 % places states | 44-57 % waits for A, 21-23 % side_b C++, 9 % `from_host` sync |

So the GIL is not saturated in the deep windows. The cards idle because both threads block on synchronous host work:
- the state parking (a pageable `.to("cpu")` of GiBs of KV, which drains the card's queue);
- the embed and target copies (pageable);
- B's per-segment `float(kl)`;
- each thread waiting for the other's results.

The stage processes remove the cross-thread coupling (and the suspected cross-thread race behind the deep300 crashes). Each stage process
must also avoid those host syncs, so the build includes:
- **(i)** the w233 pinned asynchronous embed / targets (done, 08e1cd32);
- **(ii)** the loss sums kept on the card until the step's end (no per-segment sync);
- **(iii)** asynchronous parking: D2H into pinned memory with an event, and the H2D back on the same card's stream after it (no host wait);
  a CPU-side read (branch clone) waits on the event first;
- **(iv)** sends / receives on per-process comm threads (D2H + event + gloo send off the compute thread).

## What changes (all behind `--pipe-procs`; default off, so today's pipeline stays as it is)
1. **Processes.** Each data-parallel rank r gets two processes: stage 0 on card A (token table, layers 0..K-1) and stage 1 on card B
   (layers K..end, final norm, head, loss). Global rank = 2r + stage, world = 2W (gloo).
   - Two stage groups (the same stage across DP ranks) for the gradient all-reduce.
   - The pair talks point to point: h forward, dh back, 10 MB a segment, CPU tensors.
   - `qrun.sh` launches 2W processes, each with `ROCR_VISIBLE_DEVICES` = its card. `A,B+C,D` = 4 processes. T4 / qcard are otherwise
     unchanged.
2. **Student build per stage.** `qmodel.Student(..., only=(lo, hi))` builds and places only layers lo..hi-1 (the others are `None`), the
   head only when hi = NL, and the token table only when lo = 0. Parameter names are unchanged, so `stu.P` holds the stage's own.
3. **Segment loop.** The same work list, in the same order, in both processes; each process carries its own layers' states.
   - **Stage 0:** `state_in` (own layers) → forward A → asynchronous D2H + send h → keep up to 2 segments in flight → receive dh →
     backward A → `state_out`.
   - **Stage 1:** receive h → `state_in` → forward B + KL → backward → send dh → `state_out`.
   - The parking budget (`--state-card-gb`, `--state-budget-gb`), evictions, branch snapshots and the depth-test prefix are today's
     functions, applied per card. They only decide where a state lives, never the arithmetic.
   - **Prefix rebuilds.** The `(item, offset)` bookkeeping is identical in both processes, so both detect a missing state the same way.
     The rebuild's no-grad pieces travel the same channel (h only).
4. **`run_group`.**
   - Each process sums its own gradient vector over its stage group.
   - A world all-reduce carries the loss sums (stage 1) and prefix tokens (stage 0) and the two stages' squared norms, so `gnorm`,
     `--clip-gnorm` and logging stay global.
   - The Adam step, masks and trust clamp run per process (elementwise, identical).
   - `checksum` runs per stage group.
5. **Saving.**
   - `state.pt` holds both stages: stage 1 sends its snapshot to global rank 0. The halves are disjoint by name; there are two optimizer
     states, `opt` / `opt_s1`.
   - `write_ckpt` merges into today's checkpoint format (scales/<key>.safetensors, params.pt, norms), so the reads are unchanged.
   - Monitor / fixed eval run through the same no-grad pipeline.
   - Telemetry: stage 1's per-step memory record goes to rank 0's `train.jsonl` (`peak_gib_b`, `mem`).
6. **Not supported with `--pipe-procs`** (asserted): taps / local MSE / prompt weight / LoRA / block stage (as today's pipeline), and
   `--pipe-xfer p2p`.

## Equivalence (CPU, before any GPU time)
`testhf233.py` uses testhf225 / testhf231's tiny real Qwen3.5 stack (8 layers, CPU, gloo processes). It compares `--pipe-procs` (2
processes, and 4 = 2 DP ranks) with today's `--pipeline` (1 process; 2 ranks), **bit for bit**, on:
- the KL sums and rows per step;
- the reduced gradient;
- the parameters after 3 steps;
- `train.jsonl` fields.

Cases:
- plain;
- `--state-budget-gb` with forced evictions (`BUDGET_CPU`);
- a branch plan (testbranch231's sessions);
- a resume mid-run (prefix rebuild + lost snapshot);
- `--depth-test`;
- `--accum 2`;
- a frozen first half (`--train-layers`).

A checkpoint written by both must be byte-identical. All existing tests must still pass (test220, testhf225, testhf231, testbranch231,
testplan231, test-t4).

On GPU: a 2-card bitwise check against `--pipeline` over 3 steps, then timing on fin12 windows W1 / W2 and a 4-card dp check.

## Coordination with w231
qad220.py's `run_pipe` / budget / branch code is w231's. Two options:
- **(a)** w231 lands its pending fixes, then freezes those parts while this is built (~1 day). I start from its last commit.
- **(b)** I build on a branch and rebase onto w231's later commits before the CPU tests are rerun.

Either way the new code sits in new functions (`run_stage0` / `run_stage1`, `Dist` stage groups, `qmodel.Student(only=)`). It calls
w231's `state_in` / `state_out` / `park` / `branch_*` unchanged, so later fixes there apply to both pipelines.

## Expected gain
- The bound is card B's GPU time plus the bubble: W0 1.9× at most; w225's regime (80 % busy) ~1.25×.
- The measured fin12 W1 / W2 windows (deep traces at 120k / 200k, branch sessions, mixed lengths) set the estimate (report table).
