# rad-exl3-mix

Mixed-K EXL3 quantization for [radiance](https://codeberg.org/StillDeadcode/radiance): our own EXL3 quants at
K1–8 per tensor, a fitted sensitivity model plus an exact knapsack solver choosing the per-tensor mix, and the
KLD evaluation and search pipeline around it. Built for the radiance quantization challenge (Qwen3.8-27B).

**Released:** [EXL3-V2 containers](https://huggingface.co/Hero422/qwen3.8-27b-exl3-v2-rad), 16 sizes from 1.5 to
9.0 bpw. Method: [`docs/METHOD.md`](docs/METHOD.md). Next: EXL3-V3 (integer compute, finer K;
[`docs/DESIGN-v3.md`](docs/DESIGN-v3.md), [`docs/ROADMAP.md`](docs/ROADMAP.md)).

## EXL3-V2 results

Scores from the challenge leaderboard (private corpus, published 2026-11-01). Real bpw = file bytes × 8 /
26,895,998,464. turboderp's EXL3 ladder (EXL3-V1) is interpolated to the same size along its own leaderboard entries.
EXL3-V2 is 10–31% below it, level with IQR-V3 from 2.5 to 4.5 bpw and 11–37% below IQR-V3 from 5.5 bpw up (model
card: comparison table).

| size | real bpw | KLD | top-1 | EXL3-V1 at the same size |
|---|---|---|---|---|
| 1.50 | 1.494 | **0.51233** | 73.4% | — |
| 1.75 | 1.738 | **0.29688** | 79.4% | 0.41 (**-28%**) |
| 2.00 | 1.988 | **0.16439** | 84.9% | 0.226 (**-27%**) |
| 2.50 | 2.494 | **0.06660** | 90.2% | 0.0902 (**-26%**) |
| 3.00 | 2.999 | **0.03063** | 93.6% | 0.0444 (**-31%**) |
| 3.50 | 3.498 | **0.01712** | 95.4% | 0.0247 (**-31%**) |
| 4.00 | 3.998 | **0.00828** | 96.7% | 0.0109 (**-24%**) |
| 4.50 | 4.500 | **0.00480** | 97.6% | 0.00549 (**-13%**) |
| 5.00 | 5.000 | **0.00256** | 98.2% | 0.00286 (**-10%**) |
| 5.50 | 5.492 | **0.00153** | 98.6% | 0.00173 (**-11%**) |
| 6.00 | 5.981 | **0.000914** | 99.0% | 0.00108 (**-15%**) |
| 6.50 | 6.495 | **0.000602** | 99.2% | — |
| 7.00 | 7.001 | **0.000450** | 99.3% | — |
| 7.50 | 7.316 | **0.000393** | 99.3% | — |
| 8.00 | 8.001 | **0.000341** | 99.4% | — |
| 9.00 | 9.001 | **0.000283** | 99.4% | — |

Data behind it: `data/v2/` (every measurement, proxy tables, the final fit, the picks), mixes: `v2/mixes/`.

| dir | contents |
|---|---|
| `quantize/` | our EXL3 quantizer (`ourq.py`, on exllamav3 and the challenge calib Hessians), embedding quant, level fits |
| `mix/` | `hproxy` (per-tensor Hessian proxy error), `radmix` (reflink container mixer), calib slices, paired bootstrap |
| `eval/` | KLD scoring helpers, calib pools (`pools/pools.json`), radiance KLD-path patches |
| `search/` | `kfit2.py` sensitivity model + DP, the streaming search pipeline (`auto/`), leaderboard tools (`lb/`) |
| `v2/` | EXL3-V2: the released mix files (`v2/mixes/`), HF upload and model card (`v2/hf/`) |
| `v1/` | EXL3-V1: the turboderp-ladder port's run scripts, recipes and results, as run (absolute paths kept); HF upload and model card (`v1/hf/`) |
| `v3/` | EXL3-V3 (integer activations) experiments: activation-quant hook, int8 kernels, trellis sweeps, GEMM benches |
| `data/` | small result tables (proxy errors, measured mixes) |
| `docs/` | experiment logs (incl. what did not work), design notes, research surveys |

## How a released container is assembled

The released containers are on Hugging Face; this documents how they were built. Every V2 container is a weight-by-weight mix (`mix/radmix.cpp`, reflinks: no copy) of a base container and donors:
- **base:** EXL3-V1's 4.00 container, `q38-27b-sc4.00-e4.rad` (= `qwen3.8-27b-exl3-4.00bpw.rad` in the
  [EXL3-V1 repo](https://huggingface.co/Hero422/qwen3.8-27b-exl3-turboderp-ladder)); tensors a mix does not name
  (3 of the 16 mixes keep its embedding) come from it;
- **`${RXM_MODELS}/q38-27b-*.rad`:** other EXL3-V1 containers (sc/plain = the repo's SC / `-plain` files; e3–e6 =
  3–6-bit embedding) and `q38-27b-bf16.rad` (`rad-convert` of the bf16 checkpoint, no recipe);
- **`${RXM_OURS}/ours-K<k>.rad`, `hd-K<k>.rad`:** our quants, made by step 1 below (not published: ~200 GB).

```
mix/build.sh                       # builds mix/radmix (and hproxy, headdec); needs RXM_RADIANCE and RXM_PLUGIN
export RXM_OURS=... RXM_MODELS=...
envsubst < v2/mixes/qwen3.8-27b-exl3-v2-4.00bpw.mix > /tmp/4.00.mix
mix/radmix out-4.00.rad $RXM_MODELS/q38-27b-sc4.00-e4.rad /tmp/4.00.mix
```

Byte-identical output needs the exact base container we used (our local `q38-27b-sc4.00-e4.rad`, 13,693,493,248
bytes; the EXL3-V1 4.00 file on the leaderboard is 13,679,634,145 bytes; we have not checked which tensors differ)
and our original `ours-K*` containers; re-quantizing them is not guaranteed to be bit-reproducible.

## Run the pipeline

Needs: AMD RDNA4 card(s) with ROCm; radiance 1.13.0 + the `qc-exl3-ladder` plugins
([PR #292](https://codeberg.org/StillDeadcode/radiance/pulls/292)) and a server build with the KLD recorder
(`eval/radiance-patches/`); exllamav3 1.6.3 in a torch-ROCm venv with `quantize/exllamav3-quantize-head-memory.patch`
(`quantize/exllamav3-environment.txt`); the challenge dataset `StillDeadcode/qwen3.8-27b-quant-challenge` (calib
Hessians, corpus, bf16 golden); the bf16 checkpoint `Qwen/Qwen3.8-27B`.

1. **Quantize** every linear at K1–8: `quantize/ourq.py -K k -o OUT` (resumable per layer), container with
   `quantize/ourconv.sh` (both driven by `quantize/ourchain2.sh`). Heads: `--only-head`. ~40–80 min per K per card.
2. **Proxy table:** `mix/build.sh`, then `mix/hproxy CKPT HESS out.jsonl tag=container.rad ...`.
3. **Slices:** pools in `eval/pools/pools.json`; `mix/calib_slice.py` picks documents of a pool,
   `eval/golden_pick.py` cuts the matching golden and corpus.
4. **Evaluate a mix:** `mix/radmix` + `eval/gold2.sh` (KLD + top-1 + per-row scores), or the queue workers
   `search/auto/evalq2.sh`.
5. **Fit + solve:** `search/kfit2.py KNAPDIR TARGETS` (env `KFIT_*`, see its header) writes the mixes and a report.
6. **Stream:** `search/auto/auto.sh` with `state/stream` (fit → queue → evaluate, unattended); final:
   `search/auto/finalsel.py` (candidates on the held-out slice) + `search/auto/vpick.py` (paired-bootstrap picks).

The scripts were written for one GPU box and still assume its layout (next section); expect to adapt paths.

## Paths

Machine paths are environment variables. The defaults are the layout of the GPU box everything ran on, so nothing
needs setting there.

| variable | default | what |
|---|---|---|
| `RXM_WORK` | `/root` | work dir on the GPU box: `auto/`, `knap/`, `evq/`, `golden-*/`, `corpus/`, `challenge/`, card locks |
| `RXM_MODELS` | `$RXM_WORK/models/final` | EXL3-V1 containers (`q38-27b-<tag>.rad`) and their recipes (`<tag>.recipe`, as in `v1/results/*.recipe`) |
| `RXM_OURS` | `/data/ours` | our K1–8 containers (`ours-K<k>.rad`, `hd-K<k>.rad`) |
| `RXM_MIX` | `/data/mix` | scratch for mixed containers (`radmix` output) |
| `RXM_RADIANCE` | `$RXM_WORK/install-card` | radiance install (`bin/rad-convert`, `share/radiance`) |
| `RXM_PLUGIN` | `$RXM_WORK/radiance-hero/plugin/exl3import` | the exl3import plugin of the qc-exl3-ladder plugins (EXL3 host decoder for `mix/build.sh`) |
| `RXM_KLDBIN0` | `$RXM_WORK/radiance-kldgold` | the KLD-recorder build used by `eval/gold2.sh` when `BIN` is not set |
| `RXM_KLDBIN` | `$RXM_WORK/radiance-kldgold-fast` | radiance server built with the KLD recorder (`eval/radiance-patches/`) |
| `RXM_CKPT` | `/data/workspace/models/qwen38-27b-bf16` | bf16 checkpoint |
| `RXM_CONVDIR` | `/workspace` | rad-convert's working dir (reads `models/qwen38-27b-bf16` from it) |
| `RXM_VENV` | `/data/venv-exl3` | Python venv with torch (ROCm) + exllamav3 |
| `RXM_TEMPLATE` | turboderp's SC 4.00 `config.json` | template config for our exports |
| `RXM_LOGS` | `/workspace/logs` | logs of the CPU sweeps (`v3/e3`) |

Scripts also expect the GPU box's flat layout under `$RXM_WORK`: `hproxy/` (= `mix/`), `ourq/` (= `quantize/`),
`knap/` (`search/kfit2.py` + its data), `auto/` (= `search/auto/`), `gold2.sh`, `golden_pick.py`, `aggscore.py`.
`v1/` keeps its original absolute paths: it is the record of what ran.

Names: EXL3-V1 = the turboderp port; EXL3-V2 = mixed-K EXL3 on the f16-activation kernel; EXL3-V3 = integer compute.

License: Apache-2.0.
