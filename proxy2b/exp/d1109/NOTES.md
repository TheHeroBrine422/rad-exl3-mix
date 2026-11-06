# d1109 · REAL-Q implementation notes (worker w17, 2026-11-05)

Experiment and rule: `README.md` (#92). This file: what the code does, how close it is to the paper
(arXiv 2609.00049 v2), the choices the paper leaves open, and how to run it. Runtime cost of the result: none
(quantizer only; same EXL3 format and kernels).

## Files

| file | what |
|---|---|
| `rqdata.py` | calibration windows: qad-pool documents of `/data/challenge/calib-meta` in 2,048-token windows, shuffled like `v3/yaqa/collect.py` (seed 0 = c18 / c23's windows), plus each window's gen mask |
| `fisher.py` | stage 0: one end-to-end backward per window of the bf16 0.8B with labels sampled from its own distribution; F_l = mean_t g_t g_tᵀ at every block output (24 × 1024², fp32). Resumable (`state.pt`, `--max-sec`) |
| `realq.py` | REAL-Q around `quantize/ourq.py`. ourq runs unchanged, exec'd in-process with the same arguments, code stamp and shards. Wrapped from outside: exllamav3's `quantize_exl3`, `regularize`, `ldlq`, `ldlq_2hess` and `refit_scales`. **No shared code edited, no patch needed** |
| `rq.sh` | `q2b.sh`'s exact ourq settings, through realq.py |
| `run.sh` | the arm set for the lead (fisher → rq → rqnoh → summ), idempotent, GPU lock per unit |
| `smoke1.sh` | Fisher speed / memory, identity gate, GD timing on 2 layers |

## How it works

- **Streams.** The calibration windows (`--ncal`, default 256) live on the host as three bf16 tensors:
  - X̂_l: block l's input through the quantized prefix;
  - X\*_{l+1}, X\*_{l+2}: the bf16 model's outputs of blocks l and l+1.

  They advance one layer at each layer boundary.
- **Within a block,** linears are quantized in ourq's order (qkv/z or q/k/v, o/out, gate, up, down). A linear quantized earlier in the same block enters the forward at its quantized value; later ones stay bf16.
- **The REAL-Q step.** After every 128-row LDLQ span (exllamav3's (in, out) layout: 128 input features = the paper's "128 columns"), the current weight is [quantized rows | unquantized rows + LDLQ's accumulated error feedback], i.e. GPTQ's updated trailing columns. One Adam step runs on the unquantized rows:
  - batch of 32 windows, micro-batches of `--micro`;
  - loss α L_l + (1−α) L_{l+1}, where L = ½ mean_t Δyᵀ F Δy and α is the paper's sliding window over the block's column blocks;
  - block l+1 runs at bf16;
  - the last block uses the true KL against the head instead.

  The update is added to the unquantized rows, so the rest of the LDLQ sweep sees it.
- **Basis.** exllamav3 quantizes W_r = H_l(H_r(W / sv) / su) (block-diagonal 128-Hadamards, su/sv with g_scale). Rotated rows 0:i map exactly to input features 0:i (checked: `dequant` reproduces exllamav3's reconstruction to 0.0, and rows ≥ i are untouched). Adam runs in this basis (the analog of the paper's QuaRot-rotated weights), with lr × rms(W_r)/rms(W), so a step is lr-sized in weight units. The rms ratio was ~77 on a test tensor.
- **H_out arm.** `--hout` uses the two-sided sweep (anti-diagonals of 16×16 tiles), which has no column blocks. The steps come at every 1/(k/128) of the tiles quantized, which keeps the paper's step count per linear. The pending set is the not-yet-quantized tiles, with W + F (the two-sided feedback).
- **Scale refit** (our polish after LDLQ) fits against W + ΣΔ, the GD-updated target (`--refit-target gd`), not the original W. Otherwise the refit would pull the scales back toward the pre-GD weights.
- **Writers step of the stack** (asym ρ0, 48 tensors, `--only --hin --wpre`) can run through REAL-Q too: `--base-wq` supplies the other 102 tensors in the prefix and the block. With `--wpre` the target is W F, and the GD moves W F.

GD steps per full 0.8B quant: GDN layer 7+7+15+7+7+27 = 70, attention layer 3·7+15+7+7+27 = 77, total
18·70 + 6·77 = **1,722**; writers-only run 24·42 = 1,008.

## Faithfulness to the paper

Same as the paper:
- Fisher: aggregated full d×d true Fisher at block outputs from one backward per sample.
- Steps: one Adam step after every 128-column block on all still-unquantized weights, with the gradient recomputed on the current partially-quantized weight.
- Adam 0.9 / 0.999 / 1e-8, bias-corrected, state per linear across its block steps, lr fixed within a block.
- Loss sliding window α = 1 − (b−1)/(B−1) over all column blocks of the block's linears.
- Last block on the true KL with its own lr; reverse-cosine layer schedule from 0.01·η_final (eq. 29).
- Batch 32, sequence 2,048, Δy clip at 0.95 (the small-Qwen3 setting).
- Sequential: quantized-prefix inputs.

Different, by necessity or by our stack:
1. **Analytic part:** our LDLQ over the trellis (scaled Hessian, i8, qtf2, damping as q2b.sh) on the published calib H_in (bf16 inputs, 10M tokens), plus the stack's Σdy² H_out for the rq arm. The paper uses GuidedQuant's saliency-weighted, 4-group Hessian. The H_in is not recomputed on quantized-prefix inputs (that is its own queued 0.8B test).
2. **Adam basis:** rotated incoherence basis with lr rescaled, as above.
3. **Two-sided sweep:** GD at tile fractions instead of column blocks.
4. **Scale refit** to the GD target (the paper has none).
5. **Gen mask:** the Fisher's CE terms and the last-block KL count scored positions only (as c23 / our scoring). The paper uses every WikiText token. `fisher.py --no-genmask` and `realq.py --no-genmask` switch it off.
6. **Calibration size:** GD draws from 256 windows (the paper used 2,048 at W4 and 256 for its low-bit runs). The Fisher uses 1,024 windows (c18's). Both are qad pool.

## Choices the paper leaves open (or that I had to make)

- **Learning rates.** The paper tuned per model on a coarse log grid at W4/W3/W2. I took its Qwen3-0.6B W4A16 values: η_final 3e-4, last block 1e-5. For W2 on 8B it used 30× its W4 lr, so K2/K3 may want more. **This is the main open hyperparameter.** If it is screened, do it on fit (g08-fit1), not search.
- **Δy clip:** per token, over the 1,024 channels (the paper's sentence reads either way); entries above the quantile are scaled by a detached τ/|Δy|.
- **Fisher scale:** g from the per-window sum of CE, F averaged over all positions. Adam is nearly scale-free, so only the L_l : L_{l+1} balance depends on it, and that comes from one backward.
- **Fisher precision:** bf16 backward (the default). `--fp32` exists, and smoke1 measures both.
- **Writers in the rq chain:** REAL-Q by default (`D1109_WRITERS=plain` = plain ourq as in the stack, which would overwrite REAL-Q on 48 of 150 tensors, the down/out projections that take 60% of the steps).
- **Seeds:** `--seed-offset` changes ourq's rotations and REAL-Q's minibatch draws. The Fisher and the calibration windows are shared by all arms.

## Run

    # smoke (each step < 15 min; hold the lock):  flock /data/yb/gpu.lock timeout 880 proxy2b/exp/d1109/smoke1.sh fisher|gate|gd
    # the arm set (lead):  nohup proxy2b/exp/d1109/run.sh > /data/yb/d1109/run.log 2>&1 < /dev/null &
    # one quant by hand:   FISHER=/data/yb/d1109/fisher/fisher.pt proxy2b/exp/d1109/rq.sh 3 i8 OUT [--lr ..] -- --hout DIR --sigma-reg-out 0.01

Disk: ~2 GB per (arm, K, seed) (1.2 GB full wq + swj + writers); 12 of them ≈ 25 GB in `/data/yb/d1109`.

## Amendments 1 and 2: `--prefix bf16` and `run2.sh` (worker w30, 2026-11-05)

- **`realq.py --prefix quant|bf16`** (default quant = unchanged). bf16: quantized weights never enter the prefix model
  (`Streams.init` / `advance` run every earlier layer at bf16), so each layer is quantized on bf16 inputs. In-layer
  behaviour is unchanged (earlier linears of the same block at their quantized values, `--base-wq` as before). Layers
  are independent, so `--layers L-L` per layer gives the same shards as one run. The 11 added lines carry the tag
  `# +prefix`. The code hash skips tagged lines, and `--prefix quant` is left out of the stamp's args. Without the
  option, `realq-stamp.json` is byte-identical to the pre-edit one, so `run.sh`'s quants kept resuming across the
  edit (k3rqnoh_s1 did, 17:12 UTC).
- **Gate** (`gate-prefix.sh`, logs `/data/yb/d1109/gate-prefix*.log`):
  - REAL-Q's GD is not bit-reproducible run to run. Layer l+1's loss term backpropagates through flash attention
    (layer 3), and the trellis turns any difference into ~20% different codes. So the byte checks run in deterministic
    torch mode.
  - Old code (git HEAD) vs new, layer 1 on k3rq's own layer-0 shard: q-L01, the 6 layer-1 wq files and all 70 GD
    losses are byte-identical; realq-stamp.json old == new == k3rq's. **PASS.**
  - bf16, `warn_only=False` (deterministic flash backward): layer 2 of a layers 1-2 run == layer 2 alone, byte for
    byte. **PASS.** Layer-1 wq differ from quant's (the option acts).
  - Side note: a resumed `--prefix quant` run is not bit-equal to a continuous one either (same nondeterminism), so
    seeds, not reruns, measure noise.
- **`run2.sh`** (after `run.sh` exits; writers plain as run.sh; log `/data/yb/d1109/run2.log`, results appended to
  `result.txt`):
  - `healrq`: d1105's sp healing on k{K}rqswjasym_r0, seeds 0/1, vs d1105's `k{K}edyswjasym_r0_sp`.
    heal.py trains in `timeout 840` holds and resumes from `state.pt`: a full sp train is ~15.5 min, over the
    15-min unit.
  - `rqp`: run.sh's chain with `--prefix bf16`, K3/K2 x seeds 0/1.
  - `healrqp`: sp healing on rqp.
  - `summ2`: summ.py lines plus `rules.py`, which evaluates the base rule, amendment 1 and amendment 2 (75% of
    rq's mean-log gain, unhealed and healed; or, rq failing, rqp alone against both rules; seeds 0/1).
