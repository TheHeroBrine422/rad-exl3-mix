# D-LSA in radiance: design and the two-worker split (V3 runtime option)

Status: draft brief, 2026-11-05. Nothing built yet. ROADMAP "V3 extras" has the scope (V3-only release, off by
default, fp32 sidecar map). Target use: Qwen3.8-27B at 256k–750k context with fp8 KV, one R9700.

## Facts (checked)

- **Map** (`d-lsa-weights-v1`, `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`, 6.3 MB, Apache-2.0): a torch
  zip with `{L}.W_Q_idx` and `{L}.W_K_idx` for the 16 full-attention layers L = 3, 7, …, 63, each **[768, 64] fp32**.
  768 = 4 KV heads × 192 position-free dims (head_dim 256 minus the 64 rotary dims), so there is **one index per layer,
  shared by all KV heads**; index dim 64. Query side: each GQA group's 6 queries summed, same 768 layout.
- **Flat design defaults** (`src/lsa/sglang/settings/design.py`): window 512, sink 4, top_k 4096, pool_size 4,
  bucket_size 8; a "cluster block" = window_size keys. A query reads sink + top_k + window ≈ 4.6k keys.
- **Clustering is local and incremental:** when 512 keys leave the window, that closed block's index keys are pooled
  by 4 and Ward-merged into buckets (`core/ward.py`, `core/reference.py` `build_buckets`); each bucket keeps a
  normalized centroid and its member positions. No global re-clustering. Scoring: query · centroids over all closed
  blocks, top-p buckets, expanded to top_k keys (`reference.py` `read`; fused scan in `core/kernels/scan_select.py`).
- **CPU oracle:** `src/lsa/core/reference.py` (plain torch, the reference their kernels are tested against).
- **Costs at 512k:** fp8 KV 32 KiB/token (16 layers × K+V × 4 heads × 256) = 16 GiB read per dense decode step;
  index keys 64 dims × 16 layers ≈ 2 KiB/token bf16 (+6% memory); centroids ≈ 64k × 64 × 2 B per layer ≈ 128 MB a
  step over 16 layers. LSA does not raise max context (all KV is still stored).
- **radiance today** (fork `qc-exl3-v3`, radiance 1.17.0): attention is the core op `attn_paged`
  (`libr4d/r4d_attn_paged_h256_gqa6.hip`, prefill `r4d_attn_prefill_h256_gqa6.hip`, operands in
  `libr4d/r4d_attn_args.h`: q, kv, block_table, seq_lens, k/v descale, cu_seqlens, out; page = 16 tokens; caches
  bf16 / fp8 / fp8_v4 / k6_v4). `r4d_attn_paged_window.hip` already does sliding-window paged attention. KV groups are
  declared by the arch (`plugin/qwen35_exl3/qwen35_exl3.cpp`, `RAD_KV_FULL` etc.; core `core/mem/kv.*`).
- **Prior art for the sparse kernel:** saber-engine (`GGZ14/saber-engine`, Apache-2.0, gfx1201):
  `crates/saber-kernels/hip/sattn_decode.h` (split-KV decode, in-kernel merge) and `sattn_prefill.h` (query-tiled
  prefill) take a compile-time key source "block table **or a selected slot list** + mask", with fp8 KV
  (`*_kv8`) and e4m3 QK/PV WMMA legs. Built for Qwen's QSA indexer; the list form is what LSA needs.

## Where it lives

Core radiance (new op + KV group + scheduler hook), written as a clean **upstream PR**. Arch hooks: the bf16 block
(`arch/common/rad_block_attn_gated.h`, `qwen35_bf16`), **the fp8/IQR block** (`rad_block_attn_gated_fp8.h`, used by
`qwen35_fp8`, which serves IQR-V4 checkpoints; its attention is fused with the gate quant, `op_attn_gq` = `attn_paged`
+ `gate_quant_fp8`, so LSA needs the list form of that fused op or unfuses it when on), and our `qwen35_exl3` plugin
(V3). V3's HF repos ship the map in `lsa/`. Added 2026-11-05 (user): the fp8/IQR hook is part of the PR, not left out;
worker A was briefed on the bf16 block only, so the lead adds the fp8 hook at integration if A did not.

## Interface contract (both workers code to this)

- `lsa_index_store` (worker A): after `kv_store`, writes the token's index key
  `normalize(k_free[768] @ W_K_idx)` (bf16, 64) into a new KV group `kv_lsa_idx` (same paging as `kv_attn`).
- `lsa_select` (worker A): per decode row and layer, outputs **`sel_slots` int32 [rows, S_max]** (token positions,
  ascending, padded with -1) and **`sel_count` int32 [rows]**, S_max = sink + top_k + window (4612). The list
  always includes the sink and the window; the attention kernel does no window logic of its own.
- `attn_paged_list` (worker B): `attn_paged`'s operands plus `sel_slots` / `sel_count`; positions are mapped to
  (page, offset) through the block table. Same q / out layout, same caches (bf16, fp8 first; fp8_v4 later). With a
  list of every position it must equal `attn_paged` (bit-exact for bf16 is not required; max abs diff ≤ 1e-3 and
  matching argmax on a test set).

## Worker A — index side (host + small kernels)

1. Load the map (sidecar `--lsa PATH`, fp32), wire it into the plugin; index-key store op; the `kv_lsa_idx` group.
2. Closed-block clustering: on each 512-key boundary, pool + Ward → buckets (centroid bf16/fp32, members, sizes).
   Start on the host from the CPU oracle's algorithm; a GPU kernel only if it shows in the profile.
3. `lsa_select` for decode: centroid scan + top-p + expansion to `sel_slots` (GPU kernel; the oracle checks it).
4. Tests: against `reference.py` on dumped keys/queries (same buckets, same selected sets).

## Worker B — attention side (kernel)

1. `attn_paged_list` decode for h256 / gqa6, bf16 and fp8 caches: split-KV over the list, merge, written from
   saber's `sattn_decode.h` design (credit in the file header) or by extending `r4d_attn_paged_window.hip`'s scalar
   decode path, whichever is faster to get right on our ABI.
2. Row-table registration, constraint, `R4DArgs` fields; unit test vs `attn_paged` with full lists and with random
   lists vs a torch reference.
3. Microbench: tok/s of the attention op alone at 128k / 512k, dense vs list (4.6k keys).

## Phase 2 (after decode gates): prefill

Block-sparse prefill (per query block a key list) from `sattn_prefill.h`'s list leg; incremental prefill first
(new tokens over cached keys), which is the agent workload.

## Gate (write the experiment row before running)

KLD of quant+LSA vs **the same quant with full attention** (no bf16 reference needed), fp8 KV, whole traces at
256k / 512k / 750k with YaRN, at 4.0 (V2 weights as the stand-in) then V3 2.0 / 4.0 / 7.0; decode tok/s at the
same lengths. Open: their Qwen runs use YaRN factor 40 (`lsa_flat_tp1.yaml`); match it.
