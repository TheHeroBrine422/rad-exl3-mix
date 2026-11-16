# #197 · head / embedding swaps vs KL-IQR, and a host-RAM bf16 embedding (w185, box2)

Question 1: how good are KL-IQR's head (`output`) and embedding (`token_embd`) on our trunk, at the byte cost each
carries in its own container? Question 2: what does a bf16 embedding off the card (host RAM) cost or buy, quality and
VRAM, and what would the freed VRAM buy in trunk bits? Slice: t183 only (`/data/q27/w131/slices/golden-t183`, whole
slice, gen rows, bf16 KV). Headline tables: `table.md` (also in `out/w185.md` on the lead's side).

## Files

| path | what |
|---|---|
| `tools/dq197.cpp` | dequantise one 2-D weight of a KL-IQR `.rad` to bf16 (core `enc_decode_rows` + the quantiser plugin's decoder, `libquant.so` `gptq_decode`, registered as the core's plugin loader does). `--raw` dumps a plane's bytes (ids, vrow). |
| `tools/head197.sh`, `tools/assemble197.py` | KL-IQR's row-split head (`output.kB` classes + `ids`, `vrow`) → one `[248320, 5120]` bf16 head. Checks: ids partition the vocabulary; `vrow` agrees (`vrow >> 28` = the class width, `vrow & 0x0FFFFFFF` = the row). |
| `tools/bf16cmp.py` | a decoded bf16 tensor vs the true bf16 tensor of the checkpoint (chunked; rel. Frobenius error, row cosine, scale). |
| `tools/mkdonor197.cpp` | one-weight `.rad` with a plain bf16 weight (core `RadWriter`), a donor for radmix / the vload manifest. Round trip (dq197 back) is byte identical. |
| `tools/mkmanifest197.py` | one arm's serve manifest: `base`, 336 `final` trunk lines (one per level's `final/`), head / embedding donor lines, `act a8g128` lines. Same line order as `pipeline/rxrun/hostd.py`. |
| `tools/score197.sh`, `tools/score197x.sh`, `tools/queue197*.sh` | one t183 read per arm (card lock, ram.lock shared; `score197x` adds extra flags and a per-card VRAM sampler). |
| `tools/consensus197.py` | copy of `search/v3/inbox192/consensus192.py` plus `--force TENSOR=OPTION` and `--card-total BYTES` (the S' probes). Reproduces the lead's 5.5 probe exactly (338 of 338 assignments, same bytes). |
| `tools/table197.py` | the tables and the per-arm numbers (`arms.json`): mean KL over gen rows; paired per-document bootstrap of the difference to arm (a) (paired.py's statistic, 2000 resamples, seed 0). |
| `assigns/` | the three assigns (5.5 probe, 4.0 and 2.5 incumbents, copied from the lead's driver files) and the three S' probes (inbox format and bare dicts). |
| `manifests/` | the serve manifests of all arms (text; paths are box2's). |
| `raw/` | per-arm radiance summaries (`*.json`), sampler TSVs, the arms table. |

## Build (box2, scratch `/data/q27/w185`)

- Decoder / donor tools: `/data/q27/w177/build` (their KL-IQR build, read only) gives the core library
  `core/librad_core.a`. Compile with the flags of that build's `compile_commands.json` (`-I core -I abi -I rad_include
  -isystem /opt/rocm-10.1/include -std=gnu++20`), link like `rad-info` (whole-archive `librad_core.a`,
  `vendor/librad_vendor_llama.a`, `vendor/librad_vendor_httplib.a`, `libamdhip64.so`, `libhsa-runtime64.so`,
  `--export-dynamic`). The plugin is `/data/q27/w177/install/share/radiance/quantizers/libquant.so`.
- Vload radiance (manifest reader, `final/` trunk lines): fork `radiance-hero` commit `e29c765b` (w162, #185 (B)),
  built on box2 as `/data/q27/w185/rad-vload` (cmake Ninja, Release, gfx1201, ROCm 10.1; `RAD_HIP_RUNTIME` and
  `RAD_HSA_RUNTIME` set explicitly; targets `radiance rad-info`). Its own plugin home is not installed, so every read
  runs with the search's home `RADIANCE_HOME=/data/q27/rad10-e157b/home` (the same plugins as the live search).
  Startup report: radiance 1.17.0; `--embedding-placement` default resolves to `vram` (all_vram) here.

## Commands (in order)

```
dq197 PLUGIN IN.rad token_embd.weight OUT.bf16              # their embedding (tiny, small)
head197.sh CONTAINER TAG k3 k4 k5 k6                        # tiny head classes (small: k4 k5 k6 k7)
assemble197.py OUT.bf16 248320 5120 TAG-vrow.bin --cls-values 3,4,5,6 --cls ids:bf16 ...
bf16cmp.py /data/q27/models/qwen38-27b-bf16 lm_head.weight OUT.bf16   # vs the truth
mkdonor197 donors/TAG-head.rad output.weight 248320 5120 OUT.bf16      # plain donor
mkmanifest197.py assigns/A.json manifests/A-x.mix [--head base|PATH] [--emb base|PATH]
score197.sh NAME manifests/NAME.mix /data/q27/w131/slices/golden-t183   # or queue197.sh / score197x.sh (EXTRA)
table197.py scores/ /data/q27/w131/slices/golden-t183 out/
consensus197.py conf.json OUT --probe 5.5 [--force token_embd.weight=bf16 --card-total BYTES]
```

`conf.json` = `{menu, root, results, size_tol}`: `menu` = `menu173m-v2.json`; `root` holds the driver's `state.jsonl` and
`assign/`; `results` = one `{tag, kld, slice}` line per scored reselection (from the state's `result` events).

## Arms

| arm | change vs the base's own assign |
|---|---|
| a | none (ours, as the search serves it) |
| b | head = the base's own bf16 `output.weight` (no donor line; byte-identical to the checkpoint) |
| c | head = their dequantised head (bf16 donor, `donors/{small,tiny}-head.rad`) |
| d | embedding = the base's own bf16 `token_embd.weight` (no donor line; byte-identical to the checkpoint) |
| e | embedding = their dequantised embedding (bf16 donor, `donors/{small,tiny}-emb.rad`) |

Bases: 5.5 = the lead's 5.5 consensus probe (small donors for c / e); 4.0 = the 4.0 incumbent `v3s183-4-manual-02474`
(tiny donors); 2.5 = the 2.5 incumbent `v3s183-2.5-block-02394` (tiny donors; arm (c) and (e) are an extra here).

## Caveats

- Their head and embedding are checked against the true bf16 weights (rel. Frobenius: small head 0.059, tiny head 0.116,
  small embedding 0.036, tiny embedding 0.135); the GPU-logits check was not done (their radiance has no logits dump).
- `vrow` and the ids partition are verified (248,320 of 248,320 ids; no duplicates).
- Only t183 (defaults). Single-slice differences of a few per cent need a second slice before they are news.

## Results (t183 unless stated; full numbers in `table.md`, `raw/arms.json`, `raw/t186n.json`)

Q1, mean KL over gen rows, change vs arm (a) with a 95 % paired per-document bootstrap:

| base | b true bf16 head | c their head | d true bf16 emb | e their emb |
|---|---|---|---|---|
| 5.5 probe (a 0.001305) | −3.45 % ± 1.0 | +2.82 % ± 0.9 | +0.37 % ± 7.2 | −0.46 % ± 2.1 |
| 4.0 incumbent (a 0.007080) | −6.04 % ± 1.7 | −0.67 % ± 1.0 | −3.03 % ± 3.2 | +1.99 % ± 1.9 |
| 2.5 incumbent (a 0.050391) | −3.70 % ± 1.2 | −3.04 % ± 1.1 | −5.34 % ± 1.8 | −4.26 % ± 1.5 |

Second slice (t186n, 22 documents): 5.5 b −7.99 % ± 3.6; 4.0 b −13.51 % ± 5.9 (same sign as t183).

Q2 (card bytes = menu accounting; VRAM = engine settled free, agreeing with the sysfs sampler):

| S | card now | card, bf16 emb on host | t183 a | t183 d (bf16 emb at S) | S' (label, card bpw) | t183 at S' |
|---|---|---|---|---|---|---|
| 2.5 | 8,404,987,904 B; 11.40 GiB | 8,106,248,704 B; 11.12 GiB | 0.050391 | 0.047701 | 3.2563 (2.50) | 0.043685 (−13.3 %) |
| 4.0 | 13,447,983,104 B; 16.09 GiB | 12,910,856,704 B; 15.59 GiB | 0.007080 | 0.006865 | 4.7563 (4.00) | 0.005571 (−21.3 %) |
| 5.5 | 18,490,937,344 B; 20.79 GiB | 17,755,154,944 B; 20.11 GiB | 0.001305 | 0.001310 | 6.2563 (5.50) | 0.000968 (−25.8 %) |

## Provenance

- Donors (box2 `/data/q27/w185/donors/`, plain bf16 `.rad`, four files): sha256 in `raw/donor-sha256.txt`.
- Decoder checks: head/emb rel. Frobenius vs the true bf16 weights: small head 0.0593, tiny head 0.1157, small emb 0.0361,
  tiny emb 0.1352; vrow / ids partition ok (248,320 / 248,320).
- Rows sha256 (the 28 rows files, on box2): `raw/rows.sha256`. Repeated reads of the same manifest are byte-identical
  (5.5-a vs 5.5-a-vs, 4.0-a vs 4.0-a-vs, 2.5-a vs 2.5-a-vs).
- Failed attempts kept in `raw/logs/failed/`: the two `a-ram` reads (the vload build refuses `--embedding-placement ram`
  for the trellis table; the d-ram runs are the host-RAM variant).
