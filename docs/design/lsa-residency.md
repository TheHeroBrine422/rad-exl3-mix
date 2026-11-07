# D-LSA on top of window residency (radiance #412): long context on one card for the dense 27B

Status: design + host-tested prototype, 2026-11-07 (worker w65). Branch `lsa-residency` in
`/workspace/forks/radiance-lsa-res` = our `lsa` (0b86554) + Code000's #412 (a817b2e, cherry-picked)
+ two commits of ours. **No card run yet** (gpubox was off limits for this worker; the device plan
is in "What only a card can tell"). Priority as in [lsa.md](lsa.md): fills idle capacity only.
Code000's own LSA PR was not public when this was written (checked 11-07 ~12:20 AM CST on
codeberg.org/StillDeadcode/radiance and Code000/radiance); when it appears, link it here, don't
merge it, keep the interfaces below compatible where cheap (lead note 11-07).

## In one paragraph

#412 lets a KV group declare its consumer's reach (`residency_window`); blocks outside the sinks
and the reach's tail move to a pinned host-mapped store, the block-table entry becomes cold
(`-2 - slot`), and a per-layer pager (`qsa_page`) copies the cold blocks a decode step selects into
per-row scratch pages before the attention, byte-identically. It maps onto D-LSA with one bridge
and no attention-kernel change: our read is a list of *positions* that `attn_paged_list` maps
through the block table, while `qsa_page` rewrites a row of *table entries*. So per layer:
`lsa_page_gather` (the distinct pages of each read: resident entries into a shadow table, cold ones
into a page row) → `qsa_page` (unchanged) → `lsa_page_scatter` (page row into the shadow) →
`attn_paged_list` reading the shadow. Two limits matter more than the code. **(1) #412 sizes the
pager's scratch from the same `residency_window` that sets the resident tail**, so D-LSA, whose
read can touch 1025 distinct 16-token pages, must declare a 16,384-token tail (+0.5 GiB fp8 per
sequence over what it needs). **(2) Prefill rows are promoted whole** and the sweep never demotes a
sequence mid-prompt; our prefill is dense, so a long prompt's peak VRAM is its full KV cache.
Residency therefore frees ~75% of a 128k fp8 sequence's KV and ~93% at 512k *after* prefill (many long sessions on
one card, follow-up turns), but **does not raise the longest single prompt** one card can take
(~530k tokens fp8 / ~275k bf16 at 4.0 bpw weights) until prefill reads the middle from the store
(#412 §12's "one-layer stage", not built).

## The mapping

### KV groups (Qwen3.8-27B: 64 layers, 16 full-attention, 48 GDN)

| group | per token | who reads it | residency |
|---|---|---|---|
| `kv_attn` (16 layers × K,V × 4 heads × 256) | 32 KiB fp8 / 64 KiB bf16 | `attn_paged_list` (decode, the list), `attn_paged` (prefill, dense) | **window = (ceil(top_k/pool)) × 16 = 16,384 tokens** (see below); sinks via `--kv-residency-sink-tokens` (512 default; ours need 4) |
| `kv_lsa_idx` (64 bf16 a token a layer) | 2 KiB | `lsa_cluster`, only the block a row closes (inside the last `window` = 512 keys) | window 2 × 512; never paged (no pager op; a cold entry there is never read) |
| `kv_lsa_bkt` (64 buckets × (64 + 16) × 2 B per 512-token block per layer) | 320 B | `lsa_select` scores every closed block every step | none: never demoted (as QSA's block keys) |
| GDN recurrent + conv state | fixed ~147 MiB a sequence | every step | not a paged group; untouched |

Only full-attention layers own a growing cache; GDN layers have fixed state and nothing to page.

### Block size vs #412's paging unit

#412 pages whole pool blocks (one store slot = one block of every layer, the pager copies one
layer's fragment). Our `kv_attn` block is 16 tokens (the h256/gqa6 kernels require it); QSA's is 4.
D-LSA selects at token grain in 4-key pools (`pool` = 4, block-local, 4-aligned), so a pool lies in
one page and **a read touches at most ceil(top_k / pool) + 1 = 1,025 cold pages** (the +1 for a
read whose sink page went cold under `--kv-residency-sink-tokens 0`; the window's pages are always
in the tail). That is exactly `qsa_page`'s widest row (`kMaxW` = 1025, its LDS hash). Cost: a 4-key
pool drags its 16-key page, so the link moves up to 4× the selected bytes: worst case 1,024 pages ×
32 KiB (fp8, one layer) × 16 layers = **512 MiB a decode token** (1 GiB bf16), ≈ 10-21 ms at
25-50 GB/s host-mapped reads, against ~25 ms for the 4.0 bpw weights themselves. QSA's step reads
~48 MiB. The page cache (tags; previous step's pages reused, 56-62% hits on QSA) and the 16k
resident tail cut this; how much is only measurable on the card with real selections.

### What `residency_window` must be under #412's ABI

#412 uses one number for two things: the tail kept resident, `ceil(window / block)` blocks, and a
decode row's scratch pages, `window / block + 1` (`KVManager`, `scratch_pitch`). D-LSA needs a tail
of 33 blocks (its 512-key window, unaligned) but 1,025 scratch pages. Declaring 528 tokens would
leave 34 scratch pages (the pager would leave entries cold, which the attention must never see);
so the prototype declares `(ceil(top_k/pool)) × block` = 16,384 tokens: correct, and the larger
tail makes recent selections free, but it costs 1,024 resident pages (512 MiB fp8, 1 GiB bf16) a
sequence. **Ask:** a separate scratch-pages field (question 1 below).

### The attention kernel and cold entries

`attn_paged_list` resolves key `i` as `table[seq][pos / 16]`, `pos % 16`. Three ways to keep a cold
entry out of it:

1. **Bridge + shadow table (built).** `lsa_page_gather` → `qsa_page` → `lsa_page_scatter`; the
   attention reads a per-sequence shadow table (`[max_seqs, ceil(max_ctx/16) + 1]` i32, PERSIST)
   whose entries for every page the step's reads touch are pool blocks (scratch pages for cold
   ones). No kernel change; the split law is over `sel_max`, not the context, so the output is the
   resident run's bit for bit when the bytes are. Stale shadow entries are never read (only pages
   of listed positions are). Verify rows of one sequence page into their own scratch rows; two rows
   naming one block leave either page in the shadow, same bytes.
2. Zero-copy: the kernel reads a cold key straight from the host-mapped store. A kernel change, a
   PCIe round trip inside the tile loop, no cache. Not pursued.
3. Physical-slot list (saber's `KeyMap::List`): the pager writes a physical slot per key, the kernel
   skips the table. A small kernel change; it would allow **pool-granular paging** (copy 4-key pools,
   not 16-key pages: ¼ of the link bytes, ¼ of the scratch). Worth it if the card shows paging cost
   (question 4).

### Selection latency and page-in

Per layer, all on the compute stream, no host sync: `lsa_select` (~100 µs at 128k, ~135 µs at 512k
a decode row) → gather (one workgroup a row) → `qsa_page` plan + fill (bounded by the cold pages'
PCIe reads: ≤ 32 MiB fp8 a layer) → scatter → `attn_paged_list` (~31 µs fp8). The fetch cannot
start before the select (the selection is device data and data-dependent), and the next full-
attention layer's select depends on this layer's output, so there is no cross-layer prefetch.
The only prefetch is temporal: the page cache keeps last step's pages per (row, layer). A copy-engine
prefetch would need the selection on the host (#412 §4 rejects that for QSA for the same reason).

### Prefill, and the largest context on one card

Our prefill is dense (block-sparse prefill is LSA phase 2), and #412 promotes every prefill row's
cold middle before its step and skips mid-prompt sequences in the sweep. So during a prompt the
whole KV is resident, and the longest prompt one card takes is unchanged by residency:

| weights | fp8 KV | bf16 KV |
|---|---|---|
| 2.0 bpw (6.3 GiB) | ~720k | ~370k |
| 3.0 bpw (9.4 GiB) | ~625k | ~325k |
| 4.0 bpw (12.5 GiB) | ~530k | ~275k |
| 5.0 bpw (15.7 GiB) | ~435k | ~225k |

(32 GiB − weights − ~2 GiB runtime/activations (estimate, to measure) − GDN state, over 34.3 KiB
(fp8) / 66.3 KiB (bf16) a token: KV + index keys + buckets. The model's native window is 262,144;
past it needs YaRN.) What residency changes is the VRAM a sequence holds after its prefill
(decode phase, one sequence, computed):

| context | fp8 all resident | fp8 residency (#412 ABI) | fp8, scratch decoupled | host store | bf16 all resident | bf16 residency |
|---|---|---|---|---|---|---|
| 32k | 1.07 GiB | 1.03 GiB | 0.55 GiB | 0.55 GiB | 2.07 GiB | 2.05 GiB |
| 128k | 4.29 GiB | 1.06 GiB | 0.58 GiB | 3.73 GiB | 8.29 GiB | 2.08 GiB |
| 256k | 8.58 GiB | 1.10 GiB | 0.62 GiB | 7.98 GiB | 16.58 GiB | 2.12 GiB |
| 512k | 17.16 GiB | 1.18 GiB | 0.69 GiB | 16.48 GiB | 33.16 GiB | 2.19 GiB |

So: nothing at 32k (scratch + tail cost what the middle saves), ~3.2 GiB at 128k fp8, ~16 GiB at
512k. Two real constraints on iceman: the store is **pinned host memory** (16.5 GiB at 512k fp8)
and iceman has 30 GB of RAM, most of it taken by quantization jobs; and a follow-up turn's
prefill promotes the whole middle back. To raise the single-prompt limit we need prefill rows
served from the store: #412 §12's one-layer stage (each layer's history staged into a one-layer
buffer per chunk: 2 KiB a token fp8 of VRAM, ~N × chunks link bytes, e.g. 512 GiB total over a
512k prefill at 8k chunks ≈ 10-20 s, hidden under minutes of dense prefill) or LSA phase-2 sparse
prefill (a chunk's selection union is still most of the history, #412 §12). With a stage the
binding limit becomes host RAM (~34 KiB a token fp8).

## The prototype (branch `lsa-residency`)

| commit | what |
|---|---|
| `6ec7fea` | #412 a817b2e cherry-picked (author Code000). Conflicts: ABI only. Both PRs took ABI 38 at the same tail of `RadBuildCtx`; here **#412 keeps 38** (`kv_residency` at offset 208) and **D-LSA moves to 39** (`RadBuildCtx::lsa` at 216, `RadArchProbe::lsa_ok`); layout table, `abi_layout_test` pin, emulator test lists merged. (#416 is at 48 on its own branch; whichever lands first keeps its number.) |
| `ac91898` | `lsa_page_gather` / `lsa_page_scatter`: libref/ref_lsa.h `page_gather` / `page_scatter` (reference, refuses non-ascending lists, pages past the table, overflow), libr4d `r4d_lsa_page.hip` (gather: a workgroup a row, chunk scan; scatter: a thread an entry), schemas and rows in `r4d_rows.cpp`, docs/OPS.md, `tests/r4d_emu/test_lsapage.cpp`. |
| `936d3ee` | `rad_block_lsa.h` / `rad_block_attn_gated.h` / `qwen35_bf16.cpp`: with `--lsa` and `--kv-residency-mib`, where the decode rows attend over lists, the windows (kv_attn 16,384, kv_lsa_idx 1,024, kv_lsa_bkt none), the bridge ops per full-attention layer, issued between `lsa_select` and `attn_paged_list` when the batch has the store open; the attention then reads the shadow. Off (no store, or no list attention): nothing declared, pinned graphs unchanged. `arch_test` case. |

**Tests (host, colo, clang 23 for the emulator; `ctest` targets built individually):**
`abi_layout_test` (ABI 39 table) ok; `arch_test` 224 cases ok, incl. the new
`qwen35_bf16_lsa_under_window_residency_pages_between_select_and_attention` (windows, op counts,
issue order select < gather < qsa_page < scatter < attn_paged_list, the shadow as the attention's
table operand, lint clean, store closed = unchanged step, no list attention = nothing declared);
`mem_test` (243 cases, 110k checks), `sched_test`, `ref_test`, `config_test`, `lsa_ref_test`,
`r4d_emu_lsa` (30/30), `r4d_emu_qsapage` all ok; **`r4d_emu_lsapage` 1853/1853**: the kernels equal
the reference on 200 random tables (resident / empty / cold entries, verify rows), and the chain
(gather → emulated `qsa_page` with tags, 4 drifting steps × 2 layers, two sequences, a demoted
middle whose pool blocks were overwritten → scatter) reads every listed key's K and V bytes through
the shadow exactly as the all-resident run, bf16 and fp8 (page cache served 272/496 and 260/494
paged entries). `attn_paged_list` itself does not build in the emulator (`v8hf` WMMA,
`extern __shared__`), so output identity through the real attention is a card check. libr4d's HIP
build was not run (no ROCm off gpubox); `r4d_rows.cpp` passes a g++ syntax check and
`r4d_lsa_page.hip` compiles as C++ under the emulator.

Not done: the fp8/IQR block (`rad_block_attn_gated_fp8.h`) has no LSA hook yet, so no residency
there either; our `qwen35_exl3` plugin likewise (V3). TP > 1 is refused under `--lsa` anyway.

## What only a card can tell (device plan, for the lead to schedule)

`local/workers/w65/device-run.sh` (not run): build libr4d + the server into `/root/dev-home` from
`/data/lsa/res`, then on card 0 under `/root/card0.lock` (+ `/root/ram.lock` shared), ≤ 10 min a hold:

1. `r4d_selftest --case qsa_page_u8` and the existing `lsa_device_test` on the merged tree.
2. **Identity**: a 27B mix (the 0.8B can't: the published map is 27B-only, key_dim 768), `--lsa
   MAP --kv-cache-dtype fp8`, one ~32k-token prompt (past 16,384 + 512 so the middle goes cold),
   greedy 256 tokens, `--kv-residency-mib 0` vs `8192`: byte-identical text, and `/stats`
   `kv_residency.demoted > 0`, `paged_in > 0` (a run that never paged is a failure, not a pass).
3. **Cost**: decode tok/s on/off at 32k and 128k, fp8 and bf16, `paged_hits / paged_in`,
   `RADIANCE_QSA_PAGE_CACHE=0` as an A/B; VRAM from `/stats` `freed_bytes` + `rocm-smi`.
4. **Largest prompt** with and without residency (expected equal: prefill-bound), and the largest
   *pair* of sessions (where residency should win).

Host RAM on iceman (30 GB, quantization using most of it) limits the store: 128k fp8 needs ~4 GiB
pinned, 512k ~16.5 GiB. Run 4 only when the box is otherwise idle.

## Questions for Code000

1. **Scratch vs tail.** Can `RadKVGroupDecl` get a separate "pages a decode row may select" (e.g.
   `residency_scratch_blocks`, 0 = today's `window / block + 1`)? D-LSA reads a 512-token window but
   up to 1,025 cold 16-token pages; today it must declare a 16,384-token tail (+512 MiB fp8 / 1 GiB
   bf16 a sequence). ABI 38 is unreleased, so it could be the same step.
2. **Dense prefill rows from the store.** D-LSA's prefill is dense, so residency doesn't raise the
   longest single prompt. Would you take #412 §12's one-layer stage for prefill rows (and let the
   sweep demote between chunks of a long prompt), or should we build it on your design?
3. **`qsa_page` as a generic pager.** We use it unchanged on rows of table entries built from
   position lists. OK to depend on its contract as is (maybe a neutral name, `kv_page`)? Its plan
   is capped at 1,025 pages a row; top_k 8192 would need 2,049.
4. **Narrower paging.** A 4-key pool drags a 16-key page (up to 4× the bytes). Would a
   pool-granular fragment (your `KVGroupPlan::residency_frag()` note) with a physical-slot list
   for the attention fit your plans, or do you see a better shape?
5. **A discard class.** `kv_lsa_idx`'s middle is never read after its block is clustered; demoting
   it costs host RAM and link bytes for nothing. A "drop when out of window" flag, or is demoting
   fine?
6. **Several windows.** `res_dense_rows` ORs over groups with windows (here 16,384 and 1,024). Is
   that the intended semantics for more than one residency group?
7. **Verify rows** each own their scratch row, so two rows of one sequence can page the same block
   twice. A per-sequence scratch shared by its rows: worth it?
8. **Measured link rate**: what GB/s does the `qsa_page` fill reach from the host-mapped store on an
   R9700 (x16 PCIe 5)? It sets D-LSA's worst case (512 MiB a token fp8).
9. **Store memory**: pinned only, or could a slot live in pageable / file-backed memory? Our GPU box
   has 30 GB of RAM for 64 GB of VRAM.
10. **ABI numbering**: we put #412 at 38 and D-LSA (#414) at 39; please tell us if you'd rather order
    them otherwise. And: what selection unit and block size does your LSA use, so the two can share
    this bridge?
