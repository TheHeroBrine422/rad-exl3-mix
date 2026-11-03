# w04-blockgain: A6 gain predictor, D6 row-slab ceiling, B6 MLP permutations (ranked.md A6, D6, B6)

Worker w04, 2026-11-02. CPU analysis done: everything under **Results (CPU)** below is measured. The GPU arms in `run.sh` are not
run yet, and their pass rules are drafts for the lead.

## Mechanism (blk.py)

`prepare_H_out` (exllamav3 quantize.py) builds the output factor as `P diag(h·sv²) P`, where:
- `h` is the diagonal (`ourq --hout` hout_diag);
- `sv` is regularize's output scales, i.e. bf16 row RMS / mean (signs drop out);
- `P` is the blockwise 128-point Sylvester Hadamard.

It then normalises, adds σ_out = 0.01 and takes the block-16 LDL. Because `P` is block-diagonal, **every 128-row block of output channels is its own problem**: the output-side feedback of `ldlq_2hess` never crosses a block.

With white rounding noise (equal for one- and two-sided), the expected cost in the metric `G` is `s² tr(L⁻¹ G L⁻ᵀ)`. Without feedback it is `s² tr(G)`. So, per tensor, `R_pred = Σ_blocks two-sided / one-sided` needs only `h` and the bf16 weight. No quantization is involved.

Since the factor is diagonal, the loss-weighted error of a real quant is exactly `Σ_c h_c e_c`, where `e_c = (w_c − q_c)ᵀ H_in (w_c − q_c)`. This is the "h-proxy" below.

To compare tensors on the Fisher's scale, it is divided by `tr(H_in)`: `h_c` already sums the Fisher diagonal over a row's inputs (`stats.py`).

`selftest.txt` covers:
- the transform is block-diagonal;
- the closed form matches a Monte-Carlo run of `ldlq()`'s recursion with a dithered quantizer to 0.1%;
- the GM/AM bound;
- the permutations.

## Results (CPU, 0.8B, existing quants; `data/`, reproduce with `run-cpu.sh`)

**A6: the predictor works.** Measured errors are recomputed from bf16 + `wq` for 14 quants: K3/K2 × base/bdiag × 3 seeds, plus the identity arm.

| | K3 | K2 |
|---|---|---|
| Spearman(R_pred, R_meas) over 150 tensors | **0.998** | **0.997** |
| measured seed vs seed (ceiling) | 0.998–0.999 | 0.999 |
| log-log slope | 0.994 | 0.917 |
| L-weighted h-proxy gain: predicted / measured | 12.4% / 12.3% | 12.4% / 11.4% |
| measured KL gain (#33-16, 3 seeds) | −8.1% | −5.4% |
| **proxy → KL factor** | **0.66** | **0.47** |
| identity H_out: predicted / measured h-proxy | +1.6% / +1.6% | +1.6% / +1.7% |
| one-sided proxy of bdiag vs base (what knap4 reads, F1) | **+5.9%** | +5.8% |

- **By family:** the GDN/attention input projections gain 20–26% (R 0.74–0.81), gate/up 11–12%, and the residual writers 4–5%.
- **Loss shares (K3):** down 21%, up 20%, in_proj_qkv 15%, gate 13%, out_proj 12%.
- **Quant-free loss weight:** `L_est = proxy_err · Σ wᵀHw · Σh·sv²/Σsv² / tr H` reproduces the measured weights (Spearman 1.000) when `proxy_err` comes from a one-sided `proxy.jsonl`. Without it (ε = 1) the Spearman is 0.77.
- **Consequence for the 27B:** the 27B prediction needs only the bf16 checkpoint, the calib Hessians, the ladder K3 `proxy.jsonl` and the diagonal.

**Dose and estimator in proxy terms** (`hcompare.py`, scored in the 8K Sketch-B metric; c18 is calib-qad windows). This feeds A5/A2, and the KL versions are #33-18.

| design h | gain in the 8K metric | share of 8K's | self-predicted |
|---|---|---|---|
| 128 seqs | 9.3% | 75% | 13.9% |
| 256 | 11.4% | 92% | 13.3% |
| 512 | 11.6% | 94% | 12.7% |
| 1024 | 11.8% | 95% | 12.5% |
| 1024, Σ‖x‖²dy² (xw) | 11.9% | 96% | 12.0% |
| 1024, Σdy² (dy) | 11.9% | 96% | 12.0% |

- **Bias at small n:** a small-n diagonal over-predicts its own gain.
- **Split-half cross-fit** (`crossfit08.txt`): design on one half, score in the other. Halves of 128 give 9.5%, below the true 12.5% at 1024. Self-prediction at 256 gives 13.3%. **Cross-fit is a lower bound and self-prediction an upper bound.** `predict.py` reports both when `stats.py --state hdiag-state.pt` provides the even/odd halves; validated on c18's state: 12.5% self vs 11.7% cross.

**D6: across-block ceiling** (½log₂ AM/GM of the predicted two-sided per-128-block costs; measured block costs agree).

| family | bits (pred) | measured K3 |
|---|---|---|
| GDN in_proj_qkv | **0.192** | 0.176 |
| attn q_proj | **0.125** | 0.103 |
| GDN in_proj_z | 0.056 | 0.051 |
| mlp up / gate | 0.017 / 0.004 | 0.029 / 0.009 |
| k / v / residual writers | ≤ 0.02 | ≤ 0.02 |
| L-weighted all | 0.069 | |

- **gate/up fails ranked.md's bar** (≥ 0.1 bit; median 0.012). The spread sits in the head-structured input projections.
- **K per natural section** (qkv = q|k|v; q_proj = query|gate) captures only a median 0.053 / 0.043 bit, so slabs are needed, not sections.
- **Donor splice** (K3/K2 bdiag rows, half the blocks each, blocks picked by predicted cost): sorted vs random −13%, but sorted vs uniform K2.5 (log-interpolated) +9% overall. The split only pays on qkv (0.84×), z (0.97×) and q_proj (0.96×).
- **Restricted to qkv + q_proj in a K3 mix:** −3.8% h-proxy vs uniform K2.5, −10% vs a random split.

**B6: MLP neuron permutations.**
- **Output side (gate/up two-sided cost):** a balanced snake deal of the neurons gains **+0.018 bit** L-weighted; random gains +0.012. It is concentrated in layers 14/15/22/23 (layer 23: 0.15 bit), and is below ranked.md's 0.02 bar.
- **Input side of down_proj (`downperm.py`), the larger effect:** on the 0.8B, any re-ordering cuts down's input-side LDLQ cost (scaled Hessian, 128-Hadamard, block-16 LDL):
  - median ×0.88, layer 14 ×0.66, layer 23 ×0.52;
  - ordering by gate/up importance costs the same as the natural order (≈ 1.0), which suggests the 0.8B's neurons are importance-ordered.
- **Predicted 0.8B total:** −3.0% h-proxy (down −2.25%, up −0.6%, gate −0.2%), i.e. ≈ −2% KL at K3.
- **On the 27B the down side is small** (`data/downperm27-partial.txt`): ×0.93 at layer 0, ×0.98–0.99 at layers 62/63, ≥ ×0.99 in the middle. **The 0.8B number does not transfer.** The 27B gate/up side needs the real diagonal (`run27.sh predict`).

**Found on the way: `ourq --save-wq`'s `<key>.rowerr.npz` is wrong** (`rowerrcheck.txt`).
- **Cause:** `quantize_exl3` regularizes the weight in place, and `ourq` computes the file's err/ref from that mutated `w`. The `sigma_in_sweep` path clones and is unaffected.
- **Size of the error on k3base:** file err/ref = 70–1000× `proxy_err`, and per-row Spearman vs the true errors is 0.02 / −0.02 / 0.06.
- **Who is affected:** `v3/yaqa/headslab.py` builds the slab-head plans (#34, knap4 `KN_HEADSLAB`, ranked E1) from `lm_head.rowerr.npz`. Those plans were optimised on noise. headsim's measured KLs of those plans are still real, but better plans likely exist.
- **Fix:** `rowerr.patch`. For existing head quants, the correct row errors can be recomputed offline from `wq` + checkpoint + `lm_head.input` (as `stats.py` does) without requantizing.

## Arms (GPU) and draft pass rules (lead: commit before data)

`run.sh` runs on the 4070 with VRAM_CAP_GB 7.2. It is idempotent and writes `/data/yb/w04-blockgain/result.txt`. Expected time is ~35 min (0.8B quants ≈ 2 min, evals ≈ 25 s). Controls are the existing `k{2,3}bdiag[_s1,_s2]`.

**B6 (mechanism check):** MLP neurons permuted by `data/perms08.npz` "balanced". gate/up/down are requantized at base settings + 8K diagonal (`--only` MLP; every other tensor is the bdiag donor of the same seed, identical by ourq's per-position seeds), un-permuted (`mkperm.py`) and evaluated. 6 quants + 6 evals at K3/K2 × 3 seeds. `b6check.py` compares the measured MLP h-proxy with the prediction.
- **Pass (the predictor's input-side model holds on real trellis quants):** the measured MLP h-proxy ratio perm/natural is ≤ 1 − ½(1 − predicted) at both K (predicted ≈ 0.94 of the MLP total), and the per-layer Spearman of the down ratios vs downperm's is ≥ 0.7.
- **KL (reported; the expected effect is about the seed noise):** the mean over 3 seeds is ≤ 0.99× bdiag at K3, with every perm seed below its seed-paired bdiag.
- **Decision:** B6 enters the V3 requant only if it passes AND `run27.sh predict` puts the 27B's combined gate/up + down gain at ≥ 1% L-weighted h-proxy. Then apply it only to the layers predicted ≥ 1%. It is free: no format or runtime change, only the checkpoint's neuron order. Otherwise B6 is closed.

**D6 (slab split on GDN in_proj_qkv and attn q_proj; everything else k3bdiag):** three arms per seed:
- `d6u`: uniform K2.5 (new `--only` input-group quants, 3 seeds);
- `d6s`: half the 128-blocks at K3 by predicted cost, half at K2 (spliced from k3bdiag/k2bdiag, same bytes);
- `d6r`: the same with random blocks.

`splice.py`: output blocks are independent on both sides, so a splice is what a per-slab K produces, except each donor keeps its own refit `suh`. 3 quants + 9 evals.
- **Pass:** `d6s` beats `d6u` at every seed, with mean ratio ≤ 0.985 (predicted ≈ 0.975 = −3.8% proxy × 0.66), and `d6s` beats `d6r` at z ≤ −3 at every seed (sanity).
- **Then:** D6 is worth a format (per-128-slab K + per-slab `suh`) for these two families, if the 27B pilot diagonal shows ≥ 0.1 bit on them. The 27B GDN qkv is 2048 | 2048 | 6144, and its section ceiling is computed by `predict.py`.
- **Fail:** D6 is closed. gate/up are closed already by the CPU ceiling.

**A6 on the 27B (`run27.sh`):**
- `pilot` on ICEMAN, both cards: `hdiag27.sh 256`, ~2.4 h. **256 sequences, not ranked.md's 64**: at 64 the cross-fit/self bracket is too wide (see the table). The pilot is not lost, because the full run resumes from its state.
- `timing` on ICEMAN card 0, ~20 min, needs `LADDER_ARGS`: `ldlq_2hess` vs one-sided on layer 40's gate/up.
- `predict` on COLO CPU, ~30–40 min. Copy the pilot's `hdiag-snap256/`, `hdiag-state.pt` and the ladder K3 `proxy.jsonl` to `/data/work/w04/27b/` first.

- **Go rule for the full collection (draft):** the 27B L_est-weighted split-half cross-fit gain (halves of 128) is ≥ 4.75%, i.e. ≥ ½ of the 0.8B's 9.5% at the same n; AND halves median cos ≥ 0.98.
- **Then:** the expected 27B KL gain ≈ 0.5–0.66 × the self/cross bracket.
- **Below the bar:** stop the collection; the 27B diagonal buys less than half of what the 0.8B showed.

**Patch:** `rowerr.patch` (ourq.py: reload the original weight for the rowerr computation). The quant outputs are unchanged. `rowerr-test.sh` checks this on the 4070 (~2 min GPU): the layer-5 shard and `wq` must be identical old vs new, and rowerrcheck must go from WRONG to OK. Merging changes ourq's code stamp, so do it between runs (a resume across it needs OURQ_MIXED=1).

## What changes what

- **A6 passes on the 0.8B,** so the 27B diagonal can be judged by CPU before any 27B requant. This is the go/no-go ranked.md #8 asked for. In proxy terms it also supports A5: 256 sequences keep 92% of the 8K gain.
- **F1 is confirmed with numbers:** bdiag quants look +6% worse on the one-sided proxy knap4 prices with (in_proj_z +20%) while their true h-proxy is −12%. knap4 needs the h-proxy (`stats.py` + `predict.py` compute it per tensor) or its own family.
- **D6:** gate/up dead (0.012 bit); GDN qkv / attn q alive on the 0.8B (0.19 / 0.13 bit). The GPU arm costs only evals plus 3 small quants.
- **B6:** low value at 27B on the down side (≤ 1% except layer 0). The 0.8B's MLP neuron order is unusually bad, which is also a caveat for any 0.8B result that depends on the order of the MLP intermediate neurons.
- **E1 / #34:** re-run headslab after the rowerr fix.

## Files

| file | what |
|---|---|
| `blk.py` | the algebra: block costs, GM/AM, permutations |
| `stats.py` | per linear: h, sv², ref, e/TAG, tr H; `--state` gives the even/odd halves |
| `predict.py` | A6 / D6 / B6 report; `--perms-out` |
| `hcompare.py` | an estimate's h (dose / estimator / data) scored in a reference metric |
| `downperm.py` | B6 input side of down_proj |
| `mkperm.py` | builds the permuted model / Hessians / h; un-permutes quants |
| `splice.py` | D6 arms |
| `b6check.py` | measured vs predicted h-proxy for the permuted quants |
| `rowerrcheck.py` | checks `rowerr.npz` |
| `run.sh` | 4070 GPU arms |
| `run-cpu.sh` | regenerates `data/` from the VM |
| `run27.sh` | iceman pilot / timing, colo predict |
| `rowerr.patch`, `rowerr-test.sh` | the ourq fix and its test |
| `selftest*.py` / `.txt` | CPU self-tests (PASS) |

`data/*.npz` are regenerable (~118 MB; only `perms08.npz`, 0.5 MB, is needed by `run.sh`).
