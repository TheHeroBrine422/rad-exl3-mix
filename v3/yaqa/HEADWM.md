# #34-02 · Slab head, whole model; #99 · 8.0 floor split (2026-11-05 ~12:30 AM CST, rules before data)

`headwm.sh`. Heads at equal bytes (all bf16-dequantized, swapped into V2 4.00, only the head differs), scored on
golden-search27 (search pool, 103 docs), paired per document.
- **Validity first:** `h4-bf16` (the true bf16 head) must score finite and below `v4` (V2 4.00's own 8-bit head);
  otherwise the plugin's bf16-head path is broken and nothing below counts.
- **#34-02 rule:** the slab head is built into the plugin (row-slab K per 128-row block) if `s5-3` vs `u5` **and**
  `s6-3` vs `u6` are both < 1 with 95% upper bounds < 1 (whole-model). If only one is, it is a size-dependent menu
  option. Before building, IQR-V4's vocabulary-weighted head is compared on head-only KL (separate run).
- **#99 floor split (8.0):** report `v8`, `h8-bf16` (V2 8.00 + bf16 head) and, if both cards are free, `bf16` (the bf16
  model on TP2 vs the published golden: the engine + reference floor). The 8.0 rule (TODO): a V3 8.0 is built only if
  predicted ≥ 10% below 0.00034 *with* the floor fixes; this run says how much of V2 8.00's KLD the head and the
  engine floor account for. f32 output stores need a plugin option: not in this run.
