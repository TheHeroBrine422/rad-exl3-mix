# #200 checks (w187, 11-16, box2; raw files next to this note)

| check | what was run | result |
|---|---|---|
| CPU unit tests (`v3/qad200/test200.py`) | functions taken from `qad200.py` by AST; schedule (N = 1 vs `lr_main`; N = 4 shape: 194 plan steps -> 49 optimizer steps, warmup 8, decay from group 39, last group lr/10); parser defaults and flags; AdamW decoupled decay (closed form, float64, 3 steps); zero-gradient decay (2 % at wd 20); AdamW(wd 0) = Adam bitwise; accumulated gradient == one backward over the step's rows (unequal micro-batches, 5e-17) and the equal-size 1/N case | 15 / 15 PASS (`checks/test200.out`) |
| W = 0 and N = 1 = #196 (GPU, bitwise) | 3 plan steps, same data (1 M rows, fixed eval 30k rows, seed 0): #196's `qad196.py` (old, card 1) vs `qad200.py --wd 0 --accum 1` (new, card 3). `cmp200.py`: the states (2,880 tensors: scales, params, Adam state, step counts) and the 3 train records (rows, lr, train_kl, per-corpus KL and rows) | identical (`checks/cmp-old-vs-new.json`); train_kl per step identical to every digit printed (0.05398681, 0.04075040 at steps 2-3) |
| comparer is not vacuous | negative control: the new state with one scale value changed by 1e-7 | flagged: 1 difference, rc 1 (`checks/neg/cmp-old-vs-flip.json`) |
| accumulation on the GPU (smoke) | `--wd 20 --accum 4 --max-steps 2`: one optimizer step over plan steps 0-3 (AdamW, 4 micro-batches on card 1) | ran to the end; 75,375 rows (4 micro-batches, ~19k each); lr 5e-4 = 1e-3 x 1/2 (warmup 2 of 13 optimizer steps); grad norm 0.0242 logged; 98 s per micro-batch (= the plan-step rate of #196); checkpoint and state written (`checks/smoke-acc4/`) |
| data stream for 4 M rows | `plan:` line of the 4 M run: 2,193 plan steps on the whole #181 order (42,061,932 rows), budget 4 M reached at plan step 193 (4,000,406 rows; `plan.json` last = 193, fork at 156); B4 = 49 optimizer groups (193 / 4, the last with 1 plan step), warmup 8 of 49 | the 1 M run is the prefix of the 4 M run (same order sha1 2c3a906fcba5); chunk list `chunks200.txt` (49 golden dirs, sha256 dca14ff5...) |
| warmup stretched | 4 M arms: P20-4M lr at steps 1, 2 = 1e-3 / 32, 2 / 32 (`--warmup-frac 8/49` of 193 plan steps = round(31.5) = 32) | as designed |

Not checked on the GPU: the wd > 0 decay on the trained scales beyond the smoke run (the arms are the check); the
bit-identity of a 4-micro-batch group against a hand-summed gradient (the CPU test covers the arithmetic).
