# w209: native EXL3-V3 quantizer (radiance libquant `exl3v3`), stages 1 + 2 — raw data (2026-11-20, box2 card 3, R9700)

Report: `rad-local/workers/out/w209.md`. Code: radiance fork branch `exl3-v3` (`libquant/lq_exl3v3*`, `tests/exl3v3/`).
- `stage1-tiles.jsonl`, `stage1-tiles-tb16.jsonl`: `exl3v3_tool tiles` (card vs host definition, viterbi3.c, the exl3i oracle).
- `stage1-qtf2.jsonl`, `stage1-qtf2-speed.jsonl`: the same tiles through the old GPU encoder (exllamav3 fork + exl3hash mad).
- `harness-tb1.jsonl`, `harness.jsonl` (TB16): `exl3v3_tool harness` on the 22 C3 tensors; `cpuref.jsonl`: `cptc/tensor_check.py`
  (mad only, E3_V2=3, TB16, 16 threads) rerun on three of them. Reference: `../../cptc/rerun-1120/c3v-*.jsonl` (TB1).
- `prod-new.*.jsonl` / `prod-newsh.*.jsonl`: `exl3v3_tool tensor`, layer 3, K 2 / 4 / 6 / 4.25 (plain) and K4 with the V3 ladder's
  scaled Hessian + sigma_in 0.005; `old-K*/proxy.jsonl`, `oldsh-K4/`: `quantize/ourq.py --codebook mad --qtf2` on the same layer.
- `agg.py` (tables), `extrap.py` (speed extrapolations), `pull.sh` (copy from box2).
No KLD data was read (proxy errors on calib Hessians only): no overfit audit needed.

## kernel-1121/ — the fp16 kernel pass (2026-11-21, user decision: fp16 tile search)
qtf2 carried into libquant as the default tile search (radiance `exl3-v3` 68f082f9 + the K8 follow-up). Same box, card and
tiles: `stage1-tiles.jsonl` (`exl3v3_tool tiles`: f32 reference + fp16 search; the K8 row there is the removed fp16 K8 kernel,
`stage1-tiles-k8.jsonl` the final build), `stage1-qtf2*.jsonl` (the old encoder on the dumped tiles: codes identical on all 16
rates), `prod-new.*.jsonl` (layer 3, K 2 / 4 / 6 / 4.25), `old-K2`, `old-K4` (ourq re-timed in the same session), `agg.py`.
