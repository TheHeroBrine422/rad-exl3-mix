# Bias / side-car fields for exl3i: design note (w91 item 3, 2026-11-08; no format or code change made)

Trigger: `../w91.md` item 2 was POSITIVE, narrowly. On plain 0.8B levels a per-output-channel bias β = (W − Ŵ) μ on
every linear cuts KL 1.1% at K3 (3 seeds, each z ≈ −10). It is seed-dependent at K2 (1.000 / 0.975 / 0.992).
Item 1 was DROP: restoring the massive channel's writer rows exactly at L0–L3 does nothing at 27B, so sparse exact
rows have no evidence.

## Recommendation

1. **Do not reserve a field in the exl3i encoding.** A bias plane in exl3i would need:
   - `EXL3I_ENCODING_VERSION` 3;
   - `exl3i_encoding_read` to accept a 6th plane (today it refuses any encoding that is not exactly
     `exl3i_encoding_make`'s 4 or 5 planes);
   - every exl3i kernel path (f16, A8, A4, two-pass) to know about it.

   It would buy nothing that a separate weight cannot do.
2. **If a bias ships, make it an optional architecture-level weight:** `blk.N.<linear>.bias`, f32 (or bf16) `[rows]`,
   declared by the qwen35_exl3 plugin only when the container holds it. Added after the linear: an `add` op, or fused
   into the existing residual `add_rmsnorm` for writers and into the GLU for gate / up. This is the form radiance
   already uses for biased linears elsewhere (`gemm_nt_bias` in the vision towers).
   - Absent bias = today's graph and today's container, byte for byte.
   - Nothing in the ladder changes. β is computed from fixed codes after quantization (the oracle did exactly that),
     so it never forces a requant. **It is not a pre-freeze item.**
3. **Not reserved: sparse exact rows / columns (side-car planes).** Item 1's ceiling for the massive channel's rows is
   0.0% (R = 1.0000). Column extraction for readers is a different oracle and was not tested; it would also be a
   separate weight pair (indices + bf16 rows) plus a scatter-GEMV, not an exl3i field.

## What ships if the 27B check passes (spec)

- **Container:** per linear l with a bias, a weight `blk.<L>.<name>.bias` (name as the linear's weight: `attn_qg`,
  `attn_k`, `attn_v`, `attn_output`, `ssm_qkv`, `ssm_z`, `ssm_out`, `ffn_gate_up` (gate rows then up rows), `ffn_down`),
  dtype f32, shape `[rows]`. Plus a metadata key `exl3.bias = 1` that an engine without bias support refuses. That
  makes it a loud refusal, never a silent "loaded without the bias".
  - First check whether radiance refuses undeclared container weights. If it silently ignores them, the key is
    mandatory.
- **Producer:** `rxq` stage `bias` (post-level). For each linear: β = (W − Ŵ) μ, with μ the linear's mean input on
  fit-pool calib tokens. Ŵ is decoded with `rxq/deq.py`, the same arithmetic as `w91/drift91.py`'s oracle.
  - Open choice: μ from the bf16 model (one collection, mix-independent, storable per level) or from the quantized
    mix (per mix; what the oracle used). This must be measured (0.8B, below) before building.
  - `radmix` carries the bias weights like any other weight. The census lists them. `rxq.pairs mixes` treats them as
    part of their linear's arm.
- **Engine:** the qwen35_exl3 block files (`rad_block_mlp_exl3.h`, `rad_block_gdn_exl3.h`,
  `rad_block_attn_gated_exl3.h`) declare the optional bias and add it.
  - Fused path, two options:
    - writers: the residual `add_rmsnorm` takes one more vector;
    - readers: q / k / v, qkv / z and gate_up add it before RoPE / conv / GLU (an `add` with a broadcast row), or
      inside each exl3i epilogue (4 kernels: f16, A8, A4, two-pass).
  - Unfused path: one `add` per linear.

## Byte-identity selftest (what "zero-length = today" means, automated)

1. **Container identity:** a level converted with the bias-aware converter and **no** bias data is sha256-identical
   to today's converter output from the same recipe (whole file, and `census.py` / `rxq.pairs mixes` PAIR OK with 0
   changed). Same check for a `radmix` of it into V2 3.00.
2. **Graph identity:** loading a no-bias container with the bias-aware plugin gives the same declared op list and
   plan fingerprint as today's plugin; `--kld-out` rows bit-identical on golden-search27b-c (one score, card 0).
3. **Zero bias = no bias:** a container carrying all-zero bias weights scores bit-identical rows (adding +0.0 in f32;
   a −0.0 output edge case is reported if it ever differs).
4. **Refusals:** wrong shape, wrong dtype, a bias without its linear, `exl3.bias = 1` with no bias weights, and an
   old engine on a bias container (must refuse).
5. **Oracle match:** on one 27B tensor pair, the engine's output with the bias equals torch's `y + β` to the f16
   tolerance of the existing exl3i selftest.

## Cost

- **Engineering:** ~1 day.
  - plugin declares + add ops: ~0.5 day;
  - `rxq` bias stage + census / pairs awareness: ~0.3 day;
  - selftests 1–5: ~0.2 day;
  - fused epilogues: +1–2 days, only if the unfused add's prefill cost matters.
- **Bytes:** f32 per output row. Per layer ≈ 10,240 + 6,144 + 5,120 + 2 × 17,408 + 5,120 ≈ 61k values (GDN layer;
  attention layers similar), × 64 layers × 4 B ≈ 15.7 MB = **0.005 bpw** (f16: 0.0023 bpw).
- **Runtime:**
  - **Prefill (scored):** unfused, one extra elementwise pass per linear, ~T × N × 4 B read + write: for gate_up at
    T 2048, ~0.3 GB ≈ 0.4 ms at ~640 GB/s. Over ~7 linears × 64 layers that is roughly 10–30 ms per 2048-token chunk.
    This is not negligible, so it must be measured with `rad-kbench` (or fused, ~0).
  - **Decode:** the same adds at T = 1 cost ~µs each, launch-bound: ~450 extra launches a token unless fused or
    graphed.
  - Must be timed before adoption (AGENTS: runtime cost of every experiment).

## Before anything is built: the checks that decide it (proposed; rules to be written before data)

1. **0.8B (4070, ~30 min):** the same oracle on the **adopted 0.8B stack** (swj + C9 + H_out levels, not plain), K2 /
   K3 × 3 seeds, measured on search + verify. Arms:
   - all linears;
   - writers only;
   - readers only;
   - μ from the bf16 model vs from the quantized model.

   Pass: ≥ 1% at K3 on the adopted stack with all seeds < 1. The plain-level 1.1% is an upper estimate, because swj
   already removes coherent gain error.
2. **27B (only if 1 passes):** one fixed-code pair at K2.5 on two seed pairs, with the bias applied in a dev build
   (selftests 2–3 first). Read on a fresh two-part search slice, rule as
   #117: R ≤ 0.99 and both parts < 1. Plus the measured prefill / decode cost.
3. Only then build the producer stage. It is post-freeze work: it adds weights to finished mixes and does not touch
   the ladder.
