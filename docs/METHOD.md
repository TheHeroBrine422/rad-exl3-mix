# How EXL3-V2 was made

EXL3-V2 is Qwen3.8-27B in EXL3 trellis codes with **a bit width per tensor**, chosen by a fitted sensitivity model and
an exact knapsack solver, then refined by measuring whole mixes. This page is the method in short; the experiment log
(`EXPERIMENTS.md`, incl. what did not work) and the design notes have the detail.

## 1. Options per tensor

- **Our quants:** every linear quantised at K = 1…8 bits a weight with exllamav3's EXL3 quantiser (LDLQ over the
  trellis, `mul1` codebook, output-channel scales; exllamav3 1.6.3 + `quantize/exllamav3-quantize-head-memory.patch`)
  on the challenge's published calibration Hessians (sum of x xᵀ over every calib token). Driver: `quantize/ourq.py`,
  one container per K (`quantize/ourchain2.sh`, `ourconv.sh`). Head: K2 from the full-model K2 container, K3–K8
  quantised separately.
- **turboderp's published tensors** (the EXL3-V1 containers), allowed where ≥ 5 bits a weight. The final picks use
  them at 3.5 (6 tensors), 5.5 (104), 6.0 (52), 6.5 (11), 7.0 (4), 8.0 (42) and 9.0 (2); the other 9 sizes are all ours.
- **Embedding:** a Lloyd–Max codebook under a 128-point Hadamard rotation, 1–8 bits (as in EXL3-V1), decoded by the
  `embed_tq` gather.
- **bf16**: 33 tensors at 9.0 and the embedding at 8.0 and 9.0 (the bf16 head is excluded: it broke the model in
  our tests).

## 2. Proxy error per option

`mix/hproxy.cpp`: for every (tensor, option), err = tr(ΔW H ΔWᵀ) / rows with W decoded by the plugin's own host decoder
(bit for bit what the kernel serves) and H the calibration Hessian. ~8,100 (tensor, option) rows, `data/v2/hproxy-*.jsonl`.

## 3. Sensitivity model (`search/kfit2.py`)

Predicted KLD of a mix = a run over the 64 layers, P ← P·(1 + κ_l·X_l) + X_l, where X_l sums the layer's tensor costs
s_w·err_w^q. log s_w is piecewise linear in depth per tensor group (attention in/out, delta-net in/out, FFN gate_up /
down; knots every 8 layers) plus the head's own term; κ (error compounding) on depth knots; a factor per quantiser
family; an offset per embedding level and per calibration slice. Fitted by least squares on log KLD to ~1,200
measurements: whole mixes, single-group swaps on a 4 bpw base, embedding / head swaps, single-layer probes.

## 4. Solver

Exact dynamic programming over 1 MiB byte buckets, layer by layer: the lowest predicted KLD at exactly the target size
(the compounding term makes it a DP over layers, not a knapsack over independent items).

## 5. Search

Streaming (`search/auto/`): after every ~6 new measurements the model is refitted and the queue rewritten with
- the mean model's mix per target,
- **Thompson** mixes: the DP under refits of a per-document bootstrap of the measurements,
- **local search**: mutations of the best measured mix, accepted by the model,
- **exploration**: the DP under deliberately tilted sensitivities (one group ×0.6/×1.6, a depth slope) and random
  directions, well outside the model's own uncertainty,
- targeted swaps for options the fit had no direct data on.
Shares between kinds adapt to how often each beats the producing fit's prediction. Evaluations on two GPUs, ~1.5–5 min
each (calibration slices of ~90–165k scored positions, documents ≤ 8k tokens, bf16 KV). ~1,450 evaluations in all.

## 6. Data discipline

The calibration corpus (10.26M tokens, 4,789 documents) is split into fixed pools that keep its category mix
(`eval/pools/`): **fit** (model fitting), **search**, **qad**, **verify**. The model was fitted on slices of the fit
pool. A first verify slice (ver1) tracked overfitting during the search (held-out score of the best-so-far picks);
the final choice per size was made on a second, untouched verify slice (ver2: 244 documents, 316k positions) by a
paired per-document bootstrap among the top 2–3 candidates plus the best ver1 pick, ties going to the mix built only
from our quants. The challenge's public corpus was not used for anything. Long-context check: the two ~250k-token
agent traces of the verify pool, whole.

## 7. What mattered, what did not

- Mattered: our own quants on the challenge Hessians (proxy error 0.75–0.80× turboderp's at K1–K4); the
  compounding term in the model; fixing a misfit of K1/K2 with targeted swaps; local search; picking finals on a fresh
  held-out slice (best-of-many fit-slice picks were ~3% optimistic).
- Did not: per-layer edge offsets, a learned residual re-ranker (ridge / XGBoost / GP), a vocabulary-tiered
  embedding (hurts multilingual text), more search past ~step 500 in the 2.75–4.5 bpw band (held-out score flat).
