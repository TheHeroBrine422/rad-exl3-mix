# Baseline: radiance #416 / #412 (Code 000, 2026) — what every "new vs #416" line refers to

Sources: radiance PR #416 "kv: per-layer KV quantisation, zero-bit transforms and rad-autoquant" and PR #412 "kv:
window residency for sparse attention" (codeberg StillDeadcode/radiance, both open on 2026-11-06), and the public paper
"Infinite KV Cache" (github.com/Code000-211/RadianceMemory, `docs/paper/Infinite_KV_Cache.pdf`, Nov 2026; "IKV" below).
All numbers: Qwen3.8-Flash-Next (48 layers, 12 QSA attention layers, MoE, fp8 weights / int4 experts), 2× R9700, `--tp 2`.

**Already in #416 (do not re-propose as new):**
- Per-layer layout recipe (`--kv-recipe`), one KV group per distinct layout; layouts bf16 (16 b), i8h (8.06), fp8 (8),
  fp8_v4 (6.25), k6_v4 (5.5), k5_v4 (5.0), k4h_v4 (4.5), k4h_v2h (3.5 bits/value). Integer groups = 32 values along
  head_dim of one token, one f16 scale; i8h one f16 scale per token and head.
- Zero-bit transforms: order-256 Hadamard on q, k, v after RoPE (undone on the output); static per-layer per-KV-head
  **V mean**; **vpred** (V stored as residual of an affine map of the same layer's quantised K; `P·K̂` from a second
  attention pass, 22 MiB calibration file).
- k4h_v2h's 2-bit V is "token-energy normalised" (thresholds at 0, ±α·rms; α = 1.0 built; IKV E.5).
- `rad-autoquant`: uniform screen → measured noise floor (bf16 at prefill chunk 64: **KL 0.0243**) → per-layer
  costs → knapsack → **verify on device and rescale** → second pass along the uniform hull → global knobs.
- QSA block keys in int8 (1.73 vs 3.25 KiB per 4-token block), split across TP ranks (exact); f16 GDN state.
- #412: window residency (sink 512 tokens + tail on card, unread middle in pinned host memory, `qsa_page` pager,
  byte-identical output; decode cost 3–15 %, 32 × 128K sessions with follow-ups 58/96 s vs 1 843/3 330 s).

**Key numbers (PR #416 table; IKV Table E.1):** KL above floor / top-1 / ΔNLL on 4 docs × ~34K, 16 380 positions:
i8h +0.0016 / 94.77 % / −0.0024; fp8 +0.0027 / 94.35 % / +0.0010; k6_v4 rot+vmean +0.0064; k5_v4 rot+vmean +0.0084;
k4h_v4 vpred +0.0126; k4h_v4 vmean +0.0162; k4h_v2h vpred +0.0296 (ΔNLL +0.0083, significant); k4h_v2h vmean
+0.0567 (+0.0132). Presets: near-lossless 9.56 KiB/token (KL 0.0065), fp8-equivalent 5.28 (−22 % vs fp8),
floor+0.005 4.79, compact 3.76 (−44 %, ΔNLL +0.0008, MTP acceptance 0.581).

**Measured negatives (IKV Table 9, proxy = attention-output error):** KLT/PCA of V (needs 129–183 of 256 comps for
90 % variance; 2× int4's error at 4.5 bpw); VQ codebooks for V (−30 % vs scalar at 3.5 bpw, still worse than int4 at
4.5: host tier only); static key channel scaling after Hadamard (no gain); QJL/TurboQuant-style unbiased K ("K is not
the bottleneck below 6 bits"); KIVI per-channel 2-bit (proxy 0.201 at 3 bpw vs 0.104 for k4h_v2h+vm at 3.5);
clipping (best c = 0.95, −1 %); top-1–2 % outlier retention (−9 to −11 % on k4h_v2h only, +0.3–0.5 bpw); WUSH-style
data-adaptive transform and best-of-16 random rotations (both worse than Hadamard); one norm per vector + Lloyd–Max
(TurboQuant-style; worse than int5 groups at equal bits); RoPE dims at higher precision; E2M1 FP4 (emulated, no FP4 on
gfx1201); cross-layer KV sharing (CKA 0.52 V / 0.80 K); cross-layer selection reuse and hierarchical indexer
(ΔNLL +0.010 to +0.026, significant); fp8 / PCA-64 block keys; token eviction ("unbounded error, conflicts with
positions and the prefix cache"); importance-driven demotion (no better than a position hash).

**Lessons the autoresearch judge must keep:** costs are not additive across layers (one early layer quantised ≈ all
of them), so every recipe is verified on device; the MTP drafter shares the last attention layer's cache: 2-bit V on
layer 47 dropped acceptance 0.63 → 0.41 and decode 178 → 132 tok/s while trunk KL was the best (IKV Table 11);
any 3-bit V was poor (k6_v3 0.0658, k4h_v3 0.0715 proxy, IKV E.5).

**What #416 did not cover (gaps this corpus targets):** a dense-attention model (Qwen3.8-27B: 16 full-attention
layers, gqa 6, head 256, 16-token pages; KV read in full every decode step, so bytes cost speed at ≥ 128K, unlike
QSA); tails (p99 / worst-document / by-position KL) and RULER-style probes (#416 reports mean KL, ΔNLL, top-1 on 3
four-document corpora); a fused vpred kernel; per-head (not per-layer) recipes; anything that changes which tokens
are stored (eviction, merging) or their dimensionality (low rank); trained or learned transforms beyond Hadamard;
tiered precision with #412 for the 27B (dense attention cannot page a cold middle).
