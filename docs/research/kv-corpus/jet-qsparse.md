# Jet-QSparse (2026) · arXiv:2611.01022 — **Qwen3.8-27B results, incl. agentic**

**Idea.** Training-free: store KV in NVFP4, run a **dense 4-bit scan of q·k over the whole key history** (native FP4
MMA on Blackwell), select the top **N blocks of 128 tokens** per KV head, and aggregate V over those blocks in FP16
with FP32 accumulation. Calibrated key centering; BF16 copies of the first and the current block (≤ 2 BF16 pages
per sequence and layer); FP8 prefill in a temporary arena, persistent cache packed from BF16.

**Method.** NVFP4 cache = 4.625 bits/element at head 256 (4.75 at 128). Selection: fixed budget N ∈ {32, 64, 128}
(4K–16K tokens) or capped tail-mass. Evaluated on Qwen3.6-35B-A3B, **Qwen3.8-27B**, Nemotron 3 Nano / 3.5, and
Qwen3.8-2.4T-A95B (8 nodes).

**Numbers.** Qwen3.8-27B (Table, mean ± sd): GPQA-D / AIME25 / LCB v6 / Terminal-Bench 2.1 / SWE-bench Verified:
BF16 90.1 / 99.0 / 86.7 / 70.0 / 82.7; **Jet-QSparse N = 64: 90.9 / 98.3 / 88.5 / 71.9 / 83.5, decode 1.94×, KV
0.289× BF16**; **Quest: 89.1 / 98.1 / 81.5 / 27.7 / 4.7** (KV 1.055×); SparQ 89.2 / 98.7 / 87.6 / 72.3 / 83.7 (KV
1.501×, does not fit the batch); TriAttention AIME25 79.0, LCB 66.4. Decode speedup (GB300, 128K, batch 16, 1024
generated): 1.94× on the 27B; 0.95–1.14× at batch 1, 2.00× at 252K. Pool capacity 2.34M → 6.05M tokens (2.58×).
"128 blocks per KV head: no detected loss"; smaller budgets lose on some agentic tests (model-dependent).

**Runtime cost.** Dense scan still reads all K (4-bit) every step; V reads only selected blocks; FP4 MMA on
Blackwell. Calibration of key centres (no weight updates).

**Failure modes.** **Training-free page selection (Quest) collapses on agentic tasks on the 27B (SWE-V 4.7 vs 82.7)**
while reasoning benchmarks look intact — the strongest evidence in this corpus that sparse-decode methods must be
gated on agentic / multi-turn tasks, not RULER alone. Speedups vanish at batch 1.

**New vs #416.** "Score with cheap K, read V sparsely" on a dense hybrid; key centering (a K-side analogue of V
mean). Different from QSA (trained indexer, 4-token blocks).

**Radiance fit.** On gfx1201 (no FP4): int4 K (k4h layouts already exist) scanned with int8 q via
`iu8`/`iu4` WMMA, top-N over 128-token groups of pages, then the existing paged attention kernel on the selected
pages. Two BF16 pages = sink + current, as #412. The 27B's 16 layers × 4 KV heads make the scan cheap relative to
weights at batch ≥ 8.
