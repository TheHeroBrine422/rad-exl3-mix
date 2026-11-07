# eval/ruler — validation runs (2026-11-07, w66)

Set `ruler-qwen38-v1` (README), template `chat`, greedy. Scores on RULER's 0–100 scale; cells with fewer than 10 rows
are marked `*` (their numbers move by 10–50 points a row: read them as sanity checks, not measurements).

## Qwen3.5-0.8B bf16 on the RTX 4070 (transformers 5.22 + fla, `hf` backend; radiance has no CUDA path)

`dev08-v1` (12 tasks × 4k–32k × 5 rows, run `q08-bf16-a`, one-shot prefill) and 12 tasks × 128k × 2 rows (run
`q08-bf16-c2k`, prefill in 2k chunks: a one-shot 128k prefill does not fit the 7.2 GB VRAM cap):

| task | 4k | 8k | 16k | 32k | 128k (2 rows) |
|---|---|---|---|---|---|
| niah_single_1 / 2 / 3 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| niah_multikey_1 / 2 / 3 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |
| niah_multivalue / multiquery | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| vt | 92 | 88 | 84 | 84 | 30 |
| cwe | 86 | 74 | 82 | 78 | 55 |
| fwe | 100 | 93.3 | 93.3 | 100 | 100 |
| qa_squad | 60 | 80 | 80 | 60 | 100 |
| **avg** | **94.8** | **94.6** | **94.9** | **93.5** | **90.4** |

- dev08 overall **94.47** (95% CI 92.51–96.18, 240 rows). Time: 0.6 / 0.8 / 1.3 / 2.2 s a sample at 4k / 8k / 16k /
  32k (5 min for the 240), ~27 s at 128k with 2k chunks; VRAM peak 3.9 GB at 32k.
- **Determinism:** a second run of dev08-v1 (`q08-bf16-b`, a separate process) gave byte-identical replies on 240 of
  240 rows (`rreport.py same`: 0 different; `compare`: B−A +0.00, no flips).
- **NIAH sanity:** every niah task is 100% at every length 4k–128k.
- **256k (its trained context, 262144):** not run: with 2k chunks the 256k prefill needs > 7 GB (KV of the 6 attention
  layers 3.2 GB + the chunk×context mask + weights) and hit the 7.2 GB cap (OOM, nothing written). Needs either a
  free 4070 (no cap) or a mask-free chunked attention.
- Raw records: the 4070's `/data/yb/w66/runs/{q08-bf16-a,q08-bf16-b,q08-bf16-c2k}` (the 4070 went down at 2:17 AM
  CST before they were copied here). Not done for the same reason: a paired check of the 2k-chunked prefill against
  the one-shot run on the 60 rows at 32k (the 128k column uses chunking; the 4k–32k columns do not).
- Misses are the usual RULER ones: vt drops one hop of the 5-name chain (score 0.8), cwe lists 7–9 of 10 words, qa
  fails on exact-substring strictness ("Denmark, Iceland, and Norway." vs the reference without the Oxford comma).

## Qwen3.8-27B EXL3 V2 4.00 bpw on gpubox card 0 (radiance)

Config `rad-kvbase-v2-400-bf16`: the KV task's baseline build (`/data/ar/kv/base/kv-27b/…-adf4599c`, upstream main +
#416 + our plugins, harness patch 0003), `RADIANCE_RELAYOUT_POOL_MIB=768`, bf16 KV, chunked prefill 2048 (default).
The frozen `install-116` was tried first: its 4 GiB relayout pool took MemAvailable from 9.1 GB to under 2.0 GB within
6 s of the load (our guard killed it twice; the quantization's guard line is 2.5 GB), so it cannot load beside
anything on gpubox today. Server tokenization of the rendered prompt = our token ids (`tokenize-check.json`).

| | 8k | 32k |
|---|---|---|
| fast-v1 (12 tasks × 2 rows, 48 rows) | 100.0 | 100.0 |
| run `kvbase-bf16-a`, all 65 rows (12 tasks × 3 rows at 8k, 2–4 at 32k) | 97.2 (qa_squad 66.7: the Oxford-comma miss) | 100.0 |

- **fast-v1: 100.00 overall** (48/48 rows fully right; 95% CI [100, 100]). All 65 rows: 98.61 (CI 95.8–100), one miss
  (qa_squad 8k row 2, "Denmark, Iceland, and Norway" vs the reference without the comma — the same row the 0.8B
  misses). **NIAH sanity:** every niah row right at 8k and 32k.
- **Time** (one stream, chunked prefill 2048): 5.5 s a row at 8k (4.7–7.1), 21.0 s at 32k (18.7–22.8); engine start
  11–20 s. fast-v1 = ~11 min of card; `full-v1` ≈ 9–10 h (estimate; 64k / 128k not measured, see below).
- **Determinism:** a second run (`kvbase-bf16-b`, a new engine process in another hold) gave byte-identical replies
  on all 38 rows it reached (8k all, 32k for 7 tasks; `same`: 0 different; `compare`: +0.00, no flips).
- **Holds on card 0** (CST; each also held conv.lock so no pack loads beside the engine): 1:03–1:11 AM (47 rows),
  1:37–1:43 AM (18 rows), 1:56–2:03 AM (determinism, 38 rows), 2:15 AM (the 128k attempt, killed at load), plus the
  two install-116 attempts (seconds each). ~22 min of card. Cost to the card-0 queue: the bns2 pack started ~7 min late
  (it waited for hold 1's conv.lock), the bns2 score ~6 min late (hold 2), the bn4s1 quant ~7 min late (hold 3): ~20 min.
- **64k / 128k: not run.** The engine's load transiently takes ~6 GB of host RAM even with the 768 MiB relayout pool
  (MemAvailable 8.5–8.9 GB at the hold → 2.6–2.9 GB 6–8 s into the load at max-model-len 33k; at 131k it went under our
  2.5 GB kill line). gpubox has ~8.5 GB available while the card-0 quantization waits on the lock, so a 128k engine
  does not fit beside the queue today. Run it when the box is quiet (no quantization, no pack), or after the loader's
  transient is found and cut.
- Server tokenization of the rendered chat prompt = our token ids (32,650 of 32,650: `tokenize-check.json`), so a
  text-prompt client (the kv_judge patch) sends the same tokens.

## Compared with published RULER numbers

Qwen2.5-1M technical report (arXiv 2501.15383, Table 4; RULER's 13 tasks, 4k / 8k / 16k / 32k / 64k / 128k):
Qwen2.5-7B-Instruct 96.7 / 95.1 / 93.7 / 89.4 / 82.3 / 55.1 (YaRN past 32k), Qwen2.5-7B-Instruct-1M 96.8 / 95.3 /
93.0 / 91.1 / 90.4 / 84.4, Qwen2.5-14B-Instruct-1M 97.5 / 97.1 / 94.6 / 94.9 / 94.9 / 92.2, Qwen2.5-72B-Instruct
97.7 / 97.2 / 97.7 / 96.5 / 93.0 / 88.4. No published RULER exists for Qwen3.5-0.8B or Qwen3.8-27B that we know of.

- **0.8B: 94.8 / 94.6 / 94.9 / 93.5 (4k–32k), 90.4 at 128k** — level with the 7B-1M at 4k–16k and above it at
  32k–128k. Reasons it reads high: (1) no qa_hotpot (12 of 13 tasks; HotpotQA is usually the lowest task, ~10 points
  under the average, so it would take ~1 point off each length); (2) 5 rows a cell (2 at 128k), so a length's average
  is ±3–5 points; (3) Qwen3.5 is a 262k-native hybrid (delta net + 6 attention layers) — its NIAH is flat at 100 to
  128k, where its losses are vt and cwe, the tasks every small model loses first.
- **27B V2 4.00: 100 at 8k and 32k on fast-v1** — above every Qwen2.5 row at those lengths; with 2 rows a cell this
  says "no breakage", not "better than 97.7": fast-v1 saturates on this model, and only full-v1 / kv-long-v1 / the
  128k–768k cells can rank configs of it.
