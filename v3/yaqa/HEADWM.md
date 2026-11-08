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

## Amendment 1 (2026-11-08 1:20 PM CST / 19:20 UTC, lead note 11-08 (w90); before any data on the new slice)

State at the time: v4 and h4-u5 scored on golden-search27 (11-08 11:14 / 11:33 UTC). The h4-s5-3 score was stopped at
18:37 UTC before it wrote rows. Nothing has been scored on golden-search27b for any head arm.
- **Plans:** `ourq --save-wq`'s head `rowerr.npz` was broken (err ≈ ref, LOG 11-08 12:50 PM CST #34-02), so headwm's
  plans were not error-driven. The plans are rebuilt from the corrected rowerr (`local/workers/w90/rowerr90.py`:
  saved wq + bf16 head + calib H; sum err / ref = the logged proxy within 0.13%). The broken files and the old
  s5-3 mix move to `/data/headslab3/broken-rowerr-1108/`.
- **Slice:** golden-search27 was retired for decisions (overfit audit, 11-08 6:35 AM CST). **Every arm is scored on
  both parts of golden-search27b** (w84 amendment 1: `-x` 65 xcorp pieces, `-c` 36 calib search-pool docs; gpubox
  copies in `/data/w90/`, kld.json identical to `v3/e3/settle/data/w84/`). The golden-search27 rows (v4, h4-u5) are
  report-only.
- **Per pair** (u5 → s5-3, u6 → s6-3; v4 → h4-bf16): pooled r = Σ KL(arm) / Σ KL(control) over the scored positions
  of both parts, its 95% document-bootstrap interval resampled within each part (`v3/e3/settle/w84/pair84.py`), and
  the part ratios r_x, r_c.
- **Validity:** h4-bf16 finite on both parts and pooled r(v4 → h4-bf16) < 1.
- **#34-02:** a pair passes if pooled r < 1, its 95% upper bound < 1, **and r_x < 1 and r_c < 1** (same sign on both
  parts). A pooled pass with one part ≥ 1 is flagged, not a pass. Both pairs pass → slab head built into the plugin;
  one → a size-dependent menu option; none → not built. (IQR-V4 head comparison as before, separate.)
- **#99:** v8 and h8-bf16 scored on both parts, reported (pooled and per part). The bf16-model TP2 arm is not run
  (card 1 is not ours).
