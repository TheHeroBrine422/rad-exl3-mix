# Do large MoEs quantize better than dense? Public KLD data (gathered 2026-11-04)

Question (user, for a future QFN quant: ~120B in VRAM + ~60B SSD-streamed engrams): is mean KLD ~0.025 reachable at
2–3 bpw? bpw below = GGUF size × 8 / parameters (approximate). Each source scores on its own corpus, so
comparisons across sources are loose.

| model | params | quant | bpw | mean KLD |
|---|---|---|---|---|
| Qwen3.8-27B dense (board, UD) | 27B | UD-Q2_K_XL / UD-IQ3_XXS / UD-IQ3_S / UD-Q3_K_XL | 2.82 / 3.15 / 3.47 / 3.80 | 0.079 / 0.051 / 0.035 / 0.024 |
| same, our EXL3-V2 / IQR-V4 | | | 3.0 / 3.16 | 0.031 / 0.022 |
| Qwen3.5-35B-A3B (Unsloth) | 35B | Q2_K_XL / IQ3_XXS / Q3_K_XL / Q4_K_XL | 2.75 / 3.0 / 3.67 / 4.38 | 0.097 / 0.050 / 0.031 / 0.0137 |
| DeepSeek-V4.1-Flash-1017 (Unsloth; backbone bpw, engrams separate at IQ4_NL–Q6_K; ref = official MXFP4-expert model; wiki / long17k) | 552B + 196B engram | IQ2_XXS / Q2_K_XL / IQ3_XXS / IQ3_S / Q3_K_XL / Q8_K_XL | 2.40 / 2.89 / 3.38 / 3.80 / 4.09 / 4.43 | 0.20/0.14, 0.14/0.088, 0.063/0.043, 0.032/0.020, 0.020/0.011, 0.0048/0.0028 |
| GLM-5.3-Flash (Unsloth) | 320B / 18B active | Q2_K_XL / IQ3_XXS / Q3_K_XL / Q4_K_XL / Q5_K_XL / Q6_K_XL | 2.71 / 3.0 / 3.68 / 4.98 / 5.99 / 7.28 | 0.38 / 0.28 / 0.16 / 0.049 / 0.027 / 0.019 |
| GLM-5.3 (Unsloth) | 744B / 40B active | Q2_K_XL / IQ3_XXS / Q3_K_XL / Q4_K_XL / Q5_K_XL / Q6_K_XL | 2.73 / 3.03 / 3.69 / 5.02 / 6.05 / 7.36 | 0.37 / 0.27 / 0.14 / 0.037 / 0.020 / 0.013 |
| Mistral Large 4 (Unsloth; ref FP8 1.1 TB; 7-category mean) | 1.1T / 52B active | IQ1_M / IQ2_XXS / Q2_K_XL / IQ3_XXS / Q3_K_XL / Q4_K_XL / Q6_K_XL | 1.89 / 2.20 / 2.75 / 3.08 / 3.59 / 4.40 / 6.47 | 0.092 / 0.057 / 0.030 / 0.022 / 0.016 / 0.0071 / 0.0032 |
| Ling-3.1-flash (AtomicChat; eval_neutral 144 × 4k) | 564B / 24B active | IQ1_S / IQ2_XXS / IQ2_XS / IQ2_M / IQ3_XXS / IQ3_M / IQ4_XXS / Q8_0 | 2.01 / 2.47 / 2.81 / 3.09 / 3.64 / 3.93 / 4.42 / 8.52 | 0.20 / 0.11 / 0.080 / 0.059 / 0.034 / 0.026 / 0.014 / 0.0050 |
Sources: unsloth.ai/docs/models/{deepseek-v4.1-flash, glm-5.3-flash, glm-5.3, qwen3.5/gguf-benchmarks}; MoQE
(arXiv 2310.02410: experts more robust than dense FFN, 3-bit calibration-free, 2-bit needs QAT); arXiv 2604.06515
(Mixtral: uniform 2-bit experts degrade badly).

Reading: at equal bpw these MoEs are **not** better than the dense 27B in KLD (DeepSeek ≈ dense; GLM far worse, its
Q6 at 0.013–0.019 suggests a large eval floor or a sensitive architecture). Trellis quants (EXL3-V2 / IQR) run ~2× below
UD at 3 bpw on the 27B; applied to DeepSeek-V4.1-Flash, 0.025 lands near ~3.3–3.6 bpw, not 2–3.

Update (same day, more models): the spread across MoEs is wide. Mistral Large 4 is ~2.6× below the dense 27B at
~2.8 bpw (0.030 vs 0.079) and reaches 0.025 near 2.9 bpw even as a GGUF; Ling-3.1-flash and DeepSeek-V4.1-Flash track
the dense 27B; GLM-5.3 / Flash are far worse. No KLD tables found for MiMo-V2.6, Step 5, MiniMax-M3.1-Flash (chart
only), Muse Spark Open (chart only). Eval floors differ (Q6/Q8 KLD: dense board 0.0007, Mistral 0.003, DeepSeek
0.003–0.005, Ling 0.005, GLM 0.013–0.019), which matters at ≥ 4 bpw but not at 2–3. "MoE quantizes better" is
model-dependent; QFN has to be measured.
