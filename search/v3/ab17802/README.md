# #178-02 — GDN a/b served as one trellis K4 weight (T4) + one launch a layer — w147, 2026-11-13

Source: `rad-local/workers/instr3/w147.md` (lead, 11-13: T4 adopted from #178, G not served). Results: `data/ab17802/`
(README there). Radiance: `qc-exl3-v3` 179220f (branch `qc-exl3-v3-ab147`; measured on 3055e75 + the same commit).

## What is served

- **Container form.** A GDN layer may hold `blk.N.ssm_ab.weight`: exl3i [128, 5120], rows a (48), b (48), then a's
  first 32 rows again. This is w142's padded `in_proj_ab` exactly. The arch maps it with `map_rows` (in_proj_a 0–47,
  in_proj_b 0–47, in_proj_a 0–31), so a convert reads the checkpoint as the quantizer did. Such a layer has no
  `ssm_a` / `ssm_b`.
- **When a layer is fused.** Only when its `ssm_ab` is a trellis weight: the recipe line at convert, the container's
  entry at load. Every other layer keeps its plain a / b and two `gemm_nt`, byte for byte as before. B's whole-model
  rows are identical to w132's. The two forms can be mixed layer by layer.
- **Serve path** (`qwen35_exl3` `rad_block_gdn_exl3.h`):
  - One `exl3i_dense` (N 128) in the qkv / z input group.
  - It is the third consumer of the GDN norm's AQ site, `exl3i_actq` with planes 3: the same launch, no extra actq.
  - Its plane is its own. rxq's scale refit makes its `suh` differ from qkv's, so it cannot read qkv's plane.
  - With a fused ab the site no longer writes `h`, because nothing reads it.
  - `gdn_conv_prep` / `gdn_recurrent_update` read a and b as 48-column views (`bcol`, pitch 128) of the [max_tok, 128]
    output, as stock's fp8 GDN block does. The r4d kernels already take a strided a / b.
  - Tensor parallelism 1 only (exl3i serves whole anyway).
  - The act mode must equal qkv / z's (`exl3_act_same`).
- **Recipe line** (exl3quant maps `ssm_ab` → `linear_attn.in_proj_ab`):
  `blk.*.ssm_ab.weight exl3i src=<dir with the 48 in_proj_ab tensors> codebook=mul1i8 rate=4`
- **Mix lines.** The new `radmix` lines `add` / `drop` and the matching `census` change are both **patches for the lead**
  (see the table below); `mix147.sh` writes these lines:
  `add blk.N.ssm_ab.weight <donor>`, `drop blk.N.ssm_a.weight`, `drop blk.N.ssm_b.weight`, `act blk.N.ssm_ab.weight a8g128`.

## Files

| file | what |
|---|---|
| `build147.sh` | box2 / ds8: the plugins (exl3i, exl3dense, exl3quant, qwen35_exl3) from a `git archive` against rad10-perf's install, every non-gpu test, then `W/home` = rad10-perf/home + these .so (w144's build144.sh) |
| `absrc147.py` | w142's rxq K4 job output (`/data/q27/w142/tq/K4/t-*.safetensors`) → an exl3quant source (`model.safetensors` with the 48 layers' `in_proj_ab.{trellis,suh,svh,mul1}` as stored, `config.json`, `stamp.json` K4 i8) |
| `donor147.sh` | ds8: donor `W/donor-ab-K4.rad`: the K0.125 level pack re-converted with its recipe + the ssm_ab line (`--reuse`, w144's donor144.sh) |
| `radmix.cpp`, `radmix-add-drop.patch` | `mix/radmix.cpp` + `add <w> <donor>` (a weight BASE lacks, placed after its layer's last weight) and `drop <w>` (with its act key). **Patch for the lead**: on the B mixfile the patched binary writes a byte-identical container |
| `census.py`, `census-add-drop.patch` | `store/census.py` + the same lines: an added weight is byte-checked against its donor; a dropped one must be absent with no act key; every BASE weight not dropped must be present. **Patch for the lead** |
| `mix147.sh` | ds8: B (the incumbent `.mix` + `act a8g128` on every trunk weight, stock radmix), Bx (the same through radmix147: cmp = identical), T4 (B + add / drop / act per GDN layer) |
| `abcheck147.py` | the container's ssm_ab planes vs the source (byte-equal), the oracle's decode vs w142's simulated values; `OUT` writes the oracle's bf16 a / b for a splice |
| `run147.sh` | box2, one card: KLD of B and T4 on golden-s171d, then w144's perf protocol (8k prefill-only ×4, bench ctx 8k gen 32 / ctx 0 gen 256, `--profile-ops` 8k ×2), 2 rounds, order swapped |
| `run147b.sh` | box2, after run147: the isolation scores. T4aq0: `QWEN35_EXL3_AQ=0`, each consumer quantises h itself. T4o: B with the oracle's bf16 a / b (w142's abpatch.py). T4s: B with w142's own ab-T4 |
| `run147c.sh` | box2, after run147b: `--profile-ops` over a ctx-0 256-token decode, B and T4 (the decode op table) |
| `cmp147.py` | rows → KLD per part, pairs (identical?, KLD diff, pmean log ratio with a part-stratified paired bootstrap) |
| `perfsum147.py` | perf → static VRAM, 8k prefill, TTFT, decode ms / step, the profile's a / b ops and AQ sites |

Run order: (ds8) `build147.sh exl3quant qwen35_exl3`, `absrc147.py`, `donor147.sh 4`, `mix147.sh 2.5`, `census.py`, then
copy `2.5-{B,T4}.rad` + the golden to box2; (box2) `build147.sh exl3i exl3dense exl3quant qwen35_exl3`, `run147.sh`,
`run147b.sh`, `run147c.sh`, `cmp147.py`, `perfsum147.py`.
