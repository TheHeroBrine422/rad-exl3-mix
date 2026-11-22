# v3/qad220 — QAD trainer upgrade: faster by default, every knob the next waves need (w220)

`qad220.py` = `../qad200/qad200.py` (w217's trainer: `--eps`, `--compile-ew`, `--gradnoise`, copied unchanged at commit `fe2e1183`) + the switches below.
**Every w220 flag defaults to qad200's behaviour** (checked: `test220.py` on CPU, bit-for-bit 2-step comparison on a card, see the report
`rad-local/workers/out/w220.md`). Same student (`../qad/qmodel.py`), data (`../qad181/data181.py`), checkpoint format (`scales/<key>.safetensors`
per-channel `r` / `c`, `params.pt`, `norms.safetensors`), runners of w214 / w217 (`../qad214`, `../qad217`; `run220.sh` etc. here are theirs with `qad220.py`).
Runtime: none (training tool only). Tests: `python3 test220.py` (CPU, no student; the shipped functions via AST extraction). Raw results: `../../data/qad220/`.

## Speed (automatic or by flag)
- **Frozen prefix (automatic):** when no trainable parameter lies in layers `0..k-1` (e.g. `--train-layers 32-63`, or a mask that zeroes whole
  vectors there), those layers run once under `torch.no_grad()` (no activations kept, no checkpoint recompute, no backward) and checkpointing starts at
  layer `k`. Forward values are those of the full path; the trainable layers' gradients are bitwise those of the full-backward run on the toy stack
  (`test220.py`) and rounding-level on the real model (report). `--no-frozen-prefix` turns it off (equivalence check, speed table).
- `--compile-scales --compile-ew` (w215, 1.24x) — use both for new runs (the compiled-default check is in the report).

## w220 flags
One line each; all off by default.

| flag | what |
|---|---|
| `--train-layers A-B` | train only the parameters of decoder layers A..B (inclusive); the final norm / head are frozen unless they fall in the range (they sit after the last layer) |
| `--train-tensors REGEX` | train only parameters whose name (`<linear key>:r`, `:c`, `:g` norm gains) matches (`re.search`); intersects with `--train-layers` and `--mask` |
| `--mask FILE.npz` | per-scale 0 / 1 masks (key = parameter name, value uint8 `[numel]`): masked scales get gradient 0 and stay at their start value (also under `--wd`); a vector that is all zeros is frozen outright (counts for the frozen prefix); vectors not listed are not masked. Build one with `mkmask220.py` |
| `--tie-groups G` | one shared log multiplier per G consecutive channels on every trainable `r` / `c` (G must divide the channel counts; the tied parameter's gradient is the sum of its members'); export expands back to per-channel scales (format unchanged) |
| `--tie-layers A-B` | tie only in layers A..B (default all) |
| `--clip-gnorm X` | clip the global gradient norm at X x the median of the last 256 steps' raw norms (from step 6 on); clipped steps are logged (`CLIP step ...`, `"clip"` in `train.jsonl`) |
| `--beta2 B` | Adam beta2 (default 0.99, the trainer's value; also the adapters' Adam) |
| `--lr-steps "lr1:rows1,lr2:rows2,..."` | piecewise-constant lr: lr1 until `rows1` cumulative gen rows, lr2 until `rows2`, ...; warmup ramps to lr1; after the last end the final decay (`--decay-shape`) runs from the last lr to ~0 at the budget (`--decay-frac` is not used; one budget) |
| `--decay-shape {linear,sqrt,cosine-all,linear-all}` | the final decay's shape: `linear` (WSD, today), `sqrt` (1 - sqrt(progress), Hagele et al.), `cosine-all` / `linear-all` (decay over the whole run after warmup, no constant phase) |
| `--save-at R1,R2,...` | write `OUT/state-<R>.pt` (parameters, Adam state, step, data-plan checksum) when the run first passes R gen rows: the fork points |
| `--fork-from DIR\|FILE` | start from a saved state (`DIR/state.pt` or a `state-<R>.pt`) in a fresh OUT and run on from its step with this run's schedule flags (a cooldown); the data plan up to the fork step must be this run's (checksum, from `--save-at` states) |
| `--decay-data {mix,calib}` | data of the decay phase (plan steps from the decay start): `mix` = the plan as is; `calib` = a seeded plan over calib-pool documents the earlier steps did not use (V x Q streams as usual) |
| `--prompt-weight W` | add the golden's prompt-side rows (outside its gen mask) to the loss at weight W per row; the loss is still divided by the step's gen rows, the logged KL stays gen-only; W = 0: gen rows only, today's code path. Not with the T arm (`--taps`) |
| `--local-mse LAYERS,λ[,local\|joint]` + `--local-taps DIR` | add λ x the hidden-state MSE vs the bf16 model (per gen row, mean over the listed layers of `|s-t|^2/|t|^2`) to the KL. LAYERS: `0-31`, `5+19+33`, `0-15+20`. **local** (default): a listed layer's input and output are detached from the KL's graph, so it trains on its own MSE alone and layer i's MSE reaches only layer i (the student's own states in, the bf16 trajectory's out: arm B's block-wise loss; every listed layer sees no KL, the layers after them keep theirs); **joint**: gradients flow through the whole chain (the old arm B / T form) |
| `--make-taps DIR --tap-layers L,L,...` | store the bf16 teacher's hidden states at those layers (default the DFlash2 taps 5,19,33,47,61) at the plan's gen rows (2 cards; `--local-taps` reads them). **Size: 10 KB a row a layer** (0.5 M rows x 32 layers = 164 GB): use few layers (stage ends) or fewer rows |
| `--no-frozen-prefix` | test / benchmark switch: keep the full backward path for the layers before the first trainable one |

`--a8`, `--compile-scales`, `--compile-ew`, `--wd`, `--accum`, `--eps`, `--lora`, `--norms`, ... compose with all of the above (the tied scales and the masks live
in the parameters; the A8 / compiled forwards read `r` / `c` through the same attributes).

**Mask from w217's gradient-noise pass** (`--gradnoise 40` writes `gradnoise.npz`, keys `<name>|mean`, `<name>|std`):
```
python3 mkmask220.py /data/q27/w217/runs/gn-2.50/gradnoise.npz mask-2.50-z3.npz --n 40 --z 3 --json mask-2.50-z3.json
```
keeps a scale iff `|mean| > Z * std / sqrt(40)`; prints the kept share per layer.

## Examples (box2; `run220.sh ARM CARD SIZE BUDGETS [flags]`, w214 / w217's runner with qad220.py)
```
# w214's 25A recipe, compiled (the new default)
run220.sh 25C 0 2.50 2.5e5,5e5 --lr 1e-3 --compile-scales --compile-ew
# back half only (layers 0-31 run once under no_grad: ~1.5x), lr 3e-4
run220.sh 25B 0 2.50 5e5 --lr 3e-4 --train-layers 32-63 --compile-scales --compile-ew
# signal-masked step
run220.sh 25M 0 2.50 5e5 --lr 1e-3 --mask mask-2.50-z3.npz --compile-scales --compile-ew
# tied scales in the front half, per channel in the back half
run220.sh 25T 0 2.50 5e5 --lr 1e-3 --tie-groups 128 --tie-layers 0-31 --compile-scales --compile-ew
# step decay: 1e-3 for 0.25 M rows, 3e-4 to 0.9 M, then a short decay to 1 M
run220.sh 25S 0 2.50 1e6 --lr-steps "1e-3:2.5e5,3e-4:9e5" --compile-scales --compile-ew
# long cosine to zero; clipped spikes
run220.sh 25L 0 2.50 1e6 --lr 1e-3 --decay-shape cosine-all --clip-gnorm 4 --compile-scales --compile-ew
# cooldown fork: constant phase to 60 % saved, then two cooldowns from it
run220.sh 25P 0 2.50 1e6 --lr 1e-3 --decay-frac 0.4 --save-at 6e5 --max-steps 0 ...      # (stop at the fork point with a STOP file or --max-steps N)
run220.sh 25Q-sqrt 0 2.50 1e6 --lr 1e-3 --decay-frac 0.4 --fork-from runs/25P/state-600000.pt --decay-shape sqrt --decay-data calib
# prompt-side rows at 0.2, local MSE on the front half (taps of the quarter ends)
run220.sh 25N 0 2.50 5e5 --prompt-weight 0.2
python3 qad220.py runs/taps --make-taps runs/taps --tap-layers 15,31,47 <data flags>   # then:
run220.sh 25F 0 2.50 5e5 --lr 3e-4 --local-mse 15+31,0.1 --local-taps runs/taps
```
