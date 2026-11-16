# RULES196 — QAD pilot: restricted factors, un-shrink, norms; 2.5 and 4.0 (#196)

Written 2026-11-15 7:05 PM CST before any run (user: "are there any more qad pilots we could test while the search
finishes"). Base: #191's trainer and judging (`v3/qad191/`, RULES191: student = the 2.5 incumbent of plan191, codes
frozen, 1M gen rows, lr 1e-3 at 2.5, #181 data mix, judged on #181's held-out sets on the served path).
Ideas from a competitor's public build notes (Perire / Ashlyn, radiance PR #364; lead notes in
`rad-local/research/iqr-ashlyn-2026-11-15.md`).

Arms at 2.5 (student and data exactly as #191's S-only reference):
- **A**: #191 arm A (all trunk row scales) — rerun as the in-pilot reference (same seed as #191).
- **R**: restricted factors — row scales of attn_qg, attn_k, attn_v, attn_output and ssm_out only (no ffn gate/up /
  down, no ssm_qkv / ssm_z, no head).
- **RN**: R + RMSNorm weights trained at lr / 10.
- **AN**: A + RMSNorm weights at lr / 10.
- **U**: un-shrink only, no training: per row c_r = sqrt(w_r H w_rᵀ / q_r H q_rᵀ) (w bf16 row, q decoded row, H the
  calibration input Gram — the published calib Hessians), clamped to [0.9, 1.2], on ssm_out, attn_output, ffn_down,
  attn_v, ssm_qkv (their ssm_inz is our ssm_qkv/ssm_z input side: use ssm_qkv), folded into the row scales.
- **UA**: U then A. **UR**: U then R.
Arms at 4.0 (the 4.0 search incumbent at the start, assignment hash recorded; lr 1e-4): **A4**, **R4**, **U4**, **UR4**.
Response: served-path mean KL on the #181 held-out judge sets (gen rows), per-document bootstrap CI of each arm vs the
untrained student and vs A (A4). Norm training changes RMSNorm weights: check they fold into the container (radiance
reads norms as bf16 tensors) and that the served path reproduces the torch student's rows within #191's observed gap.
Verdict per arm: CI vs A wholly below 0 = better. Runtime: none (scales / norms fold into the format).
