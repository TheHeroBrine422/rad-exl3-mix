# IQR-V4 (deadcode, November 2026): what it means for us (read 2026-11-04)

Source: `radiance.stilldeadco.de/whitepapers/iqr-v4.pdf` (copy in `/workspace/references/whitepapers/`). IQR is the
main competitor: GPTQ on a rotated int8 grid, a computed trellis code since V3, int8 activations (A8) everywhere.

## Board standing (private split, mean KL)

| bpw | IQR-V4 | our EXL3-V2 | V4 vs V2 |
|---|---|---|---|
| 2.47 / 2.5 | 0.0611 | 0.0666 | −8% |
| 3.16 / 3.0 | 0.0224 | 0.0306 (3.0) | ahead (interpolated ~−10%) |
| 3.47 / 3.5 | 0.0150 | 0.0171 | −12% |
| 4.47 / 4.5 | 0.00444 | 0.00480 | −7% |
| 5.07 / 5.0 | 0.00223 | 0.00256 | −13% |
| 5.43 / 5.5 | 0.00154 | 0.00153 | level |
| 6.06 / 6.0 | 0.00100 | 0.000914 | +9% (we lead) |
| 7–8 | | | we lead by 35–50% (their int8 grid limits the top) |

**Speed:** trunk GEMM at 2048 tokens 3,800–4,000 µs a layer (1.02–1.08× FP8's 3,712). EXL3-V1/V2 ~8,030; our V3
target ≤ ~5,200 (1.4× FP8). They lead on prefill by a wide margin, and their decode beats MXFP4 at one stream.
Our projected V3 quality (−20…−33% vs V2 at 2–4 bpw, `local/TODO.md` estimate) would put us ahead again on KL if
the 0.8B gains transfer; speed stays behind unless A8 lands near their numbers.

## Their measured wins (candidates for us)

| IQR-V4 item | their gain | ours | test |
|---|---|---|---|
| **Act order** inside the trellis (columns coded by descending H diagonal, stored back in order) + longer tail-biting runs (256 → 512 / 1024 rows on FFN) | uniform 3–6 bit KL −0–2.3%; first entries −0–7% | not tried: exllamav3 LDLQ runs in stored order (after the 128-Hadamard) | CPU trellis proxy (`v3/e3/trellis2.py` with a column permutation), then 0.8B KL |
| **bf16 (row, 128) scales on the head-fed projections** (attn_output, ssm_out) from 5 bits, 0.010 bpw | board −1–6% at 4–6 bpw | #40 per-group scales were dead at low K on all tensors; head-fed at ≥ 5 bits untested (`v3.md` §2 "scale gain per K and family … larger at ≥ 5 bits") | CPU proxy on attn_output / ssm_out at K5–K6; kernel cost (an f16 scale per 128 in the K loop) |
| Measured unit prices, L63 ffn_gate_up one width up | −6–8% at 2.8–3.8 bpw | kfit2 / knap4 already measured-swap based; L63 spike known (#21) | check the V3 menu allocation gives L63 gate_up its extra width |
| **Vocabulary-weighted head**: 4 width classes per 256-row run sorted by token probability, a width floor | head ~1.5 bits cheaper at equal KL; entries −3–10% at 4–6 bpw | our slab head (#34, Fisher-sorted 128-row slabs, K3 floor) is the same idea | confirms the direction; their floor finding matches ours |
| int8 head GPTQ'd instead of rounded (top of the ladder) | −2–3% mean, −10% median | our heads are quantized with LDLQ already | — |
| Decode kernels (block per weight tile, 16-token barrier-free tile, quantisers folded into producers) | decode 1.15–1.7× vs V3 | our decode: exl3r4d GEMV | reference for V3 decode work (user cares about decode) |

## Their negatives that contradict our 0.8B results (read before the 27B swap test)

- "Every reweighting of the Hessian lost to the plain calibration Gram: Fisher row and layer allocation,
  **YAQA-style sketched Hessians**, end-to-end row factors and uncertainty weights all followed the heavy tail of
  per-token gradient energy (10 of 65K tokens carry 30–60% of it)." Our 0.8B: diagonal Σdy² H_out wins −8…−9% at K3
  (3+ seeds, paired). Possible reasons: their GPTQ on the int8 grid vs our LDLQ + trellis; their estimators (sketched,
  row factors) vs our diagonal with output damping 0.01 and calib H_in kept; or **the gain does not transfer to the
  27B**. The swap test (`v3/yaqa/H27.md`) decides; their heavy-tail remark suggests also checking the 27B Σdy²
  diagonal's concentration (share of the sum in the top tokens / rows) from the collector's snapshots.
- "A scale refit after GPTQ … 0–2% or not at all." Our #85 refits gave 0.8B K3 −3%, K2 −9%: the same transfer question.
- "Random-sign incoherence: rotated weights are already Gaussian (excess kurtosis −0.05)": consistent with faraway's
  headroom audit.
