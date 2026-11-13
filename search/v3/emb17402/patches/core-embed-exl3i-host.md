# core-embed-exl3i-host.patch: why, and the dual-embedding loader (w144, #174-02, 2026-11-13)

Not applied: core and the ABI are shared code. Against radiance-hero `qc-exl3-v3-emb144` (core 5c9956c); not built or
served yet.

## What the patch does

1. **`is_embedding_table` accepts `embed_exl3i`** (`core/place/planner.h`). The planner pins only a model-level weight
   read solely by `embed_lookup` into the mapped host pool (`--embedding-placement ram`). Without this change an exl3i
   table always lands in VRAM whatever the flag says. With it, the record / suh / svh views go to the pinned pool, and
   `embed_exl3i` reads them in place through the pool's device mapping, exactly as `embed_lookup` reads a bf16 table.
   The kernel needs no change: `exl3dense_embed_exl3i_gpu_test --host` runs the same launch on mapped pinned memory
   (numbers in LOG #174-02).
2. **`RadBuildCtx.embedding_vram`** (appended; needs the ABI bump, 35 → 36, which the release assigns). It carries
   `--embedding-placement` as asked to `declare`, so a plugin can pick which of two token tables to declare.

## The loader (plugin side, ~10 lines in qwen35_exl3 `declare`, not written yet)

- A container may carry `token_embd.weight` (the in-VRAM table the knapsack picked for its size: Lloyd or trellis)
  **and** `token_embd_host.weight` (the host table).
- `declare` takes `token_embd_host.weight` when `!ctx->embedding_vram` and the container has it, else
  `token_embd.weight`. Then it declares the views / op exactly as today (embed_lookup, embed_tq or embed_exl3i by the
  chosen weight's encoding).
- **The unused table costs nothing**, by construction: radiance places, pins and uploads only declared weights. The
  container is mmapped, so the other table's pages are never read. *To verify when the plugin side lands:* serve one dual
  container both ways and compare static VRAM and RSS with the single-table containers.
- **Fallback:** today the default (`ram`, not asked) falls back to VRAM with a reason when the host cannot pin. With two
  tables that fallback would put the *host* table on the card (K8: 1.27 GB). So when the container has two tables and the
  pin fails, the server should stop and name `--embedding-placement vram`, rather than fall back. Alternatively the core
  could re-run `declare` with `embedding_vram = 1`.
- **Building the container:** radmix replaces only weights the base already has (`base has no weight`). The dual
  container needs either a radmix line that adds a weight under a new name (`token_embd_host.weight <donor>
  from=token_embd.weight`), or a rad-convert recipe line. That is shared code under `mix/`, so it is also left to the lead.

## Default

See LOG #174-02 for the measured costs. In short: every host candidate (bf16, int8 per row, K8 trellis) is ≈ lossless.
They differ only in host RAM and in the link traffic of the gather (a K8 trellis token reads its whole 128-row block).
