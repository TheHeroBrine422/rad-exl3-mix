# Infinite KV Cache (Code 000, Nov 2026) · github.com/Code000-211/RadianceMemory

`docs/paper/Infinite_KV_Cache.pdf`, 23 pp., built 2026-11-06 06:10 UTC (sha256 `b09c0f66…`). Setting:
Qwen3.8-Flash-Next (12 QSA layers, quantised MoE), 2× R9700 at TP2. This is the paper behind #416 / #412. Its
layouts, presets, Table 9 negatives and drafter lesson are in [radiance-416.md](radiance-416.md); this page covers
the rest.

**Idea.** QSA reads only the top 512 four-token blocks. So the sink and tail stay in VRAM, and an on-device pager
fetches the selected cold blocks from pinned host memory inside the step, with byte-identical output. Per-layer and
per-tier quantisation then runs under a measured KL limit (rad-autoquant).

**Method beyond #416.** *Tiered precision*: a recipe line `demote LAYOUT` narrows the blocks moved to host. An
on-device transcode does it with no re-rotation, rounding each block once. *Floor anatomy*: two bf16-cache runs that
differ only in prefill chunk, compared layer by layer.

**Numbers.**
- *Tiered, i8h resident (Table 8):*
  - k4h_v4+vm middle: 29.25 vs 52.5 KiB per block (−44 % host), +0.0049 above the floor, multi-round ΔNLL n.s.
  - k4h_v2h+vm middle: +0.0193, multi-round ΔNLL +0.0043 (significant).
- *Floor (§7.7):*
  - 0.0243 at chunk 64; 0.0456 with dense attention (1 800-token docs).
  - **A dense, unquantised Qwen3.5-2B in bf16 gives 0.00018, "about 130 times less".**
  - The top-10 expert set flips for ~60 % of tokens from layer 30 on.
  - An f32 router gives 0.0239. Perturbations saturate rather than add.
- *Importance-driven demotion:* no better than a position-hash control (all intervals contain zero).
- *Search cost (Table 3):*
  - 112–113 recipes per limit: the first limit takes 58 new runs in 4 222 s, later ones 16–17 runs in 1 120–1 219 s
    (~70 s a run).
  - The second pass removes 0.25–1.87 bpw.
- *Intervals:* with four documents, a t-interval over documents is 3–6× wider than the block bootstrap.

**Runtime cost.** Residency costs 3–15 % of decode, and its page-in cannot overlap the GDN layers. Tiered speed was
not measured.

**Failure modes.**
- Tiered layers + int8 block keys + MTP diverged from their emulation (acceptance ~0.68 → 0.10), and were not
  re-run after the fix.
- Every corpus has four documents.
- The bf16 checkpoint itself was never measured.

**New vs #416.**
- Tiered precision (in neither PR).
- The floor comes from this quantised MoE. The dense 27B should sit far lower, so anchor limits on fp8's KL
  rather than `--max-dkl`.
- The ModelOpt AutoQuant KV comparison (§5.9): every answer is measured, limits are floor-relative, and a second
  pass is added.
- Demotion by importance is a measured no-go.

**Radiance fit.** On the dense 27B every block is read every step, so a host tier only helps idle or prefix-cached
sessions. The search transfers as it is: see [AUTOQUANT.md](AUTOQUANT.md).
