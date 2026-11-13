# v3/qad — scale-only end-to-end QAD (#165, V3.1)

Train only the continuous parameters of a finished V3 mix (every trunk tensor's `suh` / `svh`, optionally a few norm
gains) end to end against the bf16 model's **stored** top-512 golden; the trellis codes, the container layout and the
kernels stay as they are. **Runtime: none** (scales on fixed codes; same files, same kernels).

Rules of the pilot: `RULES.md` (committed alone before data, 163ef05). Tracker: INDEX / LOG #165.

## Files

| file | what |
|---|---|
| `qmodel.py` | the student: Qwen3.5 decoder layers (transformers' own modules) with every quantized trunk linear replaced by `QLinear` (codes decoded once by `rxq.deq` to the codebook's integers, int8 on the card, bit-exact for i8 / i8c / i4; bf16 for mul1), y = had(had(x·suh·e^r) @ m)·s·svh·e^c; trainable r / c (gauge of #89: down_proj r, v_proj c fixed); arm C: q_norm / k_norm / final-norm log gains. Layers in contiguous ranges over the given cards; optional bf16 teacher layers alongside (arm B) |
| `qad.py` | the trainer: document-start windows (the stored rows saw the whole prefix), scorer KL (top-512 + tail cell, `proxy2b/common.py score_rows`' formula) on gen rows, Adam on the logs, warmup + cosine, trust clamp, monitor eval / early stop, `state.pt` resume, outputs `params.pt`, `scales/<key>.safetensors` (r / c multipliers in `rxq.swj apply_level`'s format), `norms.safetensors`; `--gate` = decode check + one window timed |
| `healeval.py` | judgement path: apply the scales to the level → pack with the level's recipe → mix over a base (V2 3.00) → radiance score per slice; `cmp` = paired per-document ratio with a bootstrap CI and doc sign counts |
| `d0jobs.py` | arm D0's writer requant in one process (the ladder level's own job specs + hin / wpre; `noop` = the no-op gate) |
| `run165.sh` | the ds8 driver of the pilot (lanes, locks, restartable steps) |
| `headexport.py` | a quantized head with a run's healed suh / svh → `q-head.safetensors` + `wq/lm_head.safetensors` (rxq.deq W, f16, stored rows) for `v3/yaqa/headslab.py build` (bf16 rows in vocab order over a container's output.weight); `--ref` compares with ourq's own `wq` (unhealed: 67.9% bit-equal, rel 4.6e-4) |
| `posci.py` | the long-context check: KL by position bin (0–2k / 2–4k / 4–8k / 8k+), healed / unhealed with a document-bootstrap CI, flag when a 4k+ bin's CI lies above 1; radiance rows or `tscore-*.npz` |
| `lcmp.py` | arm L's readout on the torch rows: L / X with CI, adapter bytes (bf16, int8 + row scales), extra bpw, size-adjusted ratio ×e^(b·Δbpw) |
| `asym-w125.patch` | for the lead: `pipeline/rxq/asym.py` + w78's blocked Cholesky solve on the lean path (`mkdenoise-w78.patch`, never ported to rxq) + per-group resume. Without it `rxq.asym denoise` takes > 15 min a down group on an R9700 (rocBLAS's degraded f64 TRSM at k 17408, ~20 h a level); with it ~1.3 min, residual 4e-15. D0 runs the patched copy (`$W/asymfix/asym.py`) |
| `tests/` | `convcheck.py` (shift_conv1d vs F.conv1d: forward bit-identical, gradients at bf16 rounding), `mkfake.py` (a golden from a truncated bf16 model, so the loop runs on a 16 GB card), `bench_block.py` (per-block time, f32 decode identity), `gdncheck.py` (flash-linear-attention vs transformers' torch GDN rule: KL and gradient cosine), `exportcheck.py` (apply_level round trip), `push.sh` |

## Running

```
# dev (one 16 GB card, 4-layer truncated model + fake golden)
python3 tests/mkfake.py BF16 LEVEL SRC_GOLDEN fake4 --layers 4 --docs 300
python3 qad.py runs/miniA --model BF16 --level LEVEL --golden fake4 --pool-override fake4/pools.json --devs cuda:0 --layers 4 --ntrain 96 --accum 4 --eval-every 4 --monitor 8
python3 tests/exportcheck.py runs/miniA BF16 LEVEL fake4 fake4/pools.json scratch --layers 4

# 27B (ds8): the pilot's lanes (2 + 1 + 1 cards, then 4 for arm B)
bash run165.sh all-lanes          # or single steps: qadA qadC d0 qadD qadB ev:control ev:A ... cmp
# amendment 2 (box2): A16 / H (R16 head), W (8k windows, 2 cards), L8 / L16 (adapters), long-context checks
bash run165.sh lanes2             # or: qadA16 qadH qadW qadL8 qadL16 h16ctl h16H ev:control16 ev:A16 ev:H ev:W
                                  #     evlong:control evlong:A evlong:W lbase tscore:base tscore:L8 tscore:L16 cmp2 pos lcmp
```
Student options (qad.py): `--head DIR --head-perm R16.npy [--head-train]` (a quantized head as the student's lm_head,
targets mapped to its stored rows), `--lora R --lr-lora` (adapters), `--score NAME=GOLDEN --init PARAMS` (torch-path
scoring, every document whole), `--conv shift` (GDN causal conv without MIOpen: its per-shape solver search stalled the
8k windows).
GDN core: put flash-linear-attention 0.6.0 (+ fla-core, einops; `pip install --no-deps --target DIR`) on PYTHONPATH;
transformers then binds its Triton chunk rule (2× faster than the torch rule on gfx1201 for fwd + bwd).

## Numbers (dad1 9070 XT, 2,048 tokens)
bf16 GEMM 135 TFLOPS; checkpointed fwd + bwd per block: GDN 107 ms (fla; 205 ms torch rule), attention 85 ms →
≈ 300 tokens/s for the whole 27B with one card active at a time (layers split over 2 cards, model-parallel only).
