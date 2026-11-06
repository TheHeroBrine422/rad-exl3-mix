# rad-exl3-quantizer

The V3 quantizer pipeline for EXL3 trellis codes (Qwen3.8-27B first; model-agnostic in time): input statistics,
per-tensor trellis quantization with LDLQ, post-quantization scale fits, and the level assembly that the knapsack and
packer consume. One Python package (`rxq`), one settings file (`config/v3.toml`), one job unit (a tensor at a level)
that a compute-donation queue (rad-quant-at-home) can hand out.

- **DESIGN.md** — stages, the job unit, the settings and their evidence, open items.
- **PORTING.md** — what was ported from the research scripts (rad-exl3-mix) and how each port was checked.
- **PROFILE.md** — where the quantizer's time goes, per stage and family, and the ranked optimization targets.

## Run

Needs: python ≥ 3.11, torch with CUDA or ROCm, the exllamav3 fork (rad-exl3-mix `quantize/exl3-patches/`) on
`PYTHONPATH`, a rad-exl3-mix checkout for the run-time kernels.

```
export PYTHONPATH=/path/to/rad-exl3-quantizer:/path/to/exllamav3-fork
export RXQ_RXM=/path/to/rad-exl3-mix RXQ_EXL3_SRC=/path/to/exllamav3-fork/exllamav3/exllamav3_ext
export TORCH_EXTENSIONS_DIR=/path/to/torch_ext     # kernel build cache
# one layer at K2.5, the settings file's defaults (= the ladder's quantizer)
python -m rxq quant --layers 10-10 -K 2.5 --ckpt MODEL --hess HESSIANS -o out/K2.5/L10
# several levels in one process; a settings override
python -m rxq quant --keys model.language_model.layers.39.self_attn.q_proj -K 2,4,6 --ckpt MODEL --hess HESSIANS \
    -o 'out/K{K}' --set quant.tile_cd.enable=true
python -m rxq job spec.json -o out/job17        # a queue job (DESIGN.md "Job unit")
python -m rxq assemble level/K2.5 out/K2.5/*    # job outputs -> an ourq-layout level (rad-convert exl3i src=...)
python -m rxq stamp                             # the settings stamp of config/v3.toml
python -m rxq.stats blend-hin CALIB XCORP 0.5 OUT       # S1: xcorp H_in blend
python -m rxq.stats hout SNAPROOT OUT dy                # S1: block-normalized H_out (OUT/all)
python -m rxq.asym denoise CALIB MOMENTS OUT --lam 1 --rho 0 --ckpt MODEL   # S5: writers' G / F
python -m rxq.swj apply LEVEL SCALES OUT                # S4: swj scales into a level
```

`VRAM_CAP_GB` caps the CUDA allocator (shared cards); `RXQ_REFIT_CPU_FALLBACK=1` lets an out-of-memory scale refit
(27B ffn_down on ≤ 12 GB cards) run on the CPU.

## Tests

- `tests/check_quant.py REF_LEVEL JOB_OUT` — per-tensor jobs vs an ourq.py level with the same settings (GPU run first).
- `tests/check_cpu.py OUT` — the CPU ports vs the originals (blend, H_out bn, swj math / apply, mkdenoise both paths).
- `tools/run4070.sh A|B|C` — the GPU checks and the profile on the RTX 4070 box.

Licence: Apache-2.0 (LICENSE, NOTICE); exllamav3 is MIT (LICENSES/).
