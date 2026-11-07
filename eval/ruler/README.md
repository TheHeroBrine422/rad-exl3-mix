# eval/ruler — RULER for radiance (long-context gate for D-LSA, the KV judge's T2, long-context checks of quants)

NVIDIA's RULER task definitions (github.com/NVIDIA/RULER at `db43f92`, Apache-2.0), ported to one module, tokenized with the
Qwen3.8 tokenizer, seeded and written once to versioned files with checksums; a runner that drives radiance (its
OpenAI-style server) or transformers (the 0.8B proxy on the 4070), resumable per sample; RULER's exact metrics; reports and
**paired** comparisons of two configs on the same rows. Synthetic data: no calib / public corpus, no pool rules.

| file | what |
|---|---|
| `rgen.py` | the generator port (niah ×8, vt, cwe, fwe, qa_squad; qa_hotpot supported, not built) + `build` / `check` |
| `build_set.sh` | builds the set `ruler-qwen38-v1` (how it was made on colo) |
| `rlib.py` | stdlib library: sets, selections, prompt rendering, `score` (RULER metrics), `report`, `paired`, atomic writes |
| `rrun.py` | runner: one config × one selection → `RUNDIR/pred/<cell>/<row>.json`; backends `radiance` / `openai` / `hf` |
| `rreport.py` | `report` / `compare` (paired bootstrap) / `same` (determinism) |
| `run4070.sh`, `rungpubox.sh` | one unit under the 4070's `gpu.lock` (VRAM cap) / one hold of gpubox card 0 (+ conv.lock, ram.lock shared, a RAM check inside the hold) |
| `select/*.json` (`mksel.py`) | versioned selections: `fast-v1`, `full-v1`, `lsa-v1`, `kv-long-v1`, `dev08-v1`, `dev08-long-v1`, `lsa-repo-128k`, `all-v1` |
| `configs/*.json` | configs: `rad-kvbase-v2-400-{bf16,fp8}` (the KV task's baseline build, what the validation used), `rad-v2-400-*` (the frozen install-116: needs > 7 GB of free host RAM to load, see RESULTS.md), `hf-q35-08b-bf16[-c2k]` (the 0.8B) |
| `manifests/ruler-qwen38-v1.json` | the set's manifest (per-cell sha256 of the uncompressed rows, environment, sources) |
| `tests/fidelity.py` | runs NVIDIA's own generator scripts beside the port, compares rows |
| `tests/lsa_hash.py` | checks rows against the LSA repo's pinned Qwen3.8 prompt-set hashes |
| `tests/killtest.sh` | preemption test: rrun SIGKILLed 5× at random moments and resumed = the clean run (passed: 75/75 records identical) |
| `tests/kvt2_test.py` | the proposed kv_judge T2 path with stubbed holds and a fake server (passed: correct candidate kept, broken one rejected) |
| `tests/logs/` | fidelity, build and LSA-hash logs |
| `runs/` | raw results of the 27B validation runs (per-sample records, reports) |
| `patches/autoresearch-v3-kv-t2.patch` | the proposed kv_judge T2 change (not applied) |
| `RESULTS.md` | validation runs (0.8B, 27B V2 4.00) |

## The set: `ruler-qwen38-v1`

- Cells (`<task>_<length>.jsonl.gz`, gzip mtime 0): 12 tasks × 4k / 8k / 16k / 32k / 64k / 128k × 100 rows, and 7 probe
  tasks (niah_single_2, niah_multikey_1/2/3, niah_multivalue, vt, cwe) × 256k / 512k / 768k (786432) × 20 rows.
  A row: `index, input, outputs, answer_prefix, length` (RULER's: prompt tokens + `tokens_to_generate`), `max_length`,
  `tokens_to_generate`. `input + answer_prefix` is RULER's base-model prompt. Rows are one seeded stream per cell after
  the haystack-size search, so the first k rows of a cell are what a k-row generation gives.
- Seed 42 (RULER's default). Sources (sha256 in the manifest): PaulGrahamEssays.json `58e35253…` (the LSA repo's
  release asset; the essays state no licence: generated files stay on our machines, git-ignored), english_words.json
  `affcd6d4…` (RULER's LFS object), SQuAD v2 dev `80a5225e…` (CC BY-SA 4.0, 4.4 MB; HotpotQA (46 MB, unpinned
  upstream) is skipped: `qa_hotpot` builds if `hotpotqa.json` is given), wonderwords 3.0.1 word lists (MIT, vendored in
  `assets/`), nltk 3.10.3 punkt_tab (`e57f6418…` zip), Qwen3.8 `tokenizer.json` (the 0.8B's encodes text identically:
  same BPE model, only extra audio added tokens).
- Where: built on colo `/data/work/w66/sets/ruler-qwen38-v1` (513 MB gzipped, 93 cells); copies in `eval/ruler/sets/`
  (git-ignored; the 4070 sees it through `/workspace`) and gpubox `/data/work/w66/sets/`. `python3 -I rgen.py check SETDIR` re-hashes every cell.
- **Fidelity:** `tests/fidelity.py` ran NVIDIA's scripts (unchanged, `python3 -I`, a shim for the word lists) and the
  port on 27 cells: all 12 tasks at 4k (3 rows) and 32k (2 rows), cwe at 128k, qa_squad at 256k (the document-repeat
  branch), niah_single_2 at 768k (the essay-repeat branch): **byte-identical rows**. Against the LSA repo's served
  Qwen3.8 rows (`tests/lsa_hash.py`, their pinned prompt-set hashes for 4k / 32k / 128k, first 100 rows): niah ×8, vt
  and qa_squad (after their prompt's trailing-whitespace strip) **match at all three lengths**, cwe at 4k; cwe at
  32k / 128k (the english_words branch) and fwe differ (their environment: our rows equal the official script's in
  ours; not chased). `lsa-repo-128k` is therefore exactly their probe rows for the four niah tasks; its cwe rows differ.

## Running

```bash
# a radiance config on gpubox card 0, one hold (rerun until exit 0; exit 3 = more to do)
eval/ruler/rungpubox.sh /data/work/w66/runs/kvbase-bf16-a eval/ruler/configs/rad-kvbase-v2-400-bf16.json --select eval/ruler/select/fast-v1.json
# the 0.8B on the 4070 (transformers; radiance has no CUDA path), one unit under gpu.lock
eval/ruler/run4070.sh /data/yb/w66/runs/q08-bf16 eval/ruler/configs/hf-q35-08b-bf16.json --select eval/ruler/select/dev08-v1.json
# any server already running (radiance's /v1/completions)
python3 -I rrun.py --set SETDIR --select select/fast-v1.json --config CFG.json --out RUNDIR --tokenizer tokenizer.json
python3 -I rreport.py report RUNDIR [--json report.json]
python3 -I rreport.py compare RUN_A RUN_B [--select select/fast-v1.json] [--json compare.json]
python3 -I rreport.py same RUN_A RUN_B      # determinism
```

- **Prompt** (`--template chat`, default): one Qwen user turn holding `input`, thinking off
  (`<|im_start|>assistant\n<think>\n\n</think>\n\n`), the assistant reply opened by the answer prefix (stripped), as the
  LSA repo serves Qwen's RULER. `--template base` sends RULER's raw `input + answer_prefix`. Prompts go as token ids
  (`tokenizers`, the tokenizer.json); radiance runs record whether the server's own `/tokenize` of the text gives the
  same ids (`tokenize-check.json`). Greedy (`temperature 0`), `max_tokens` = RULER's `tokens_to_generate` per task,
  stop at EOS. One sequence at a time, no prefix cache.
- **Config** = one JSON file (`rrun.py` docstring): backend, binary, plugin home, model, `kv_cache_dtype`, free `flags`
  (D-LSA switches, KV recipes, `--rope-scaling yarn:F:262144` past 256k, chunk sizes) and `env`. Everything but the name,
  url / port and the memory gates is the run's identity (`run.json`): resuming a RUNDIR with another config, set,
  template or tokenizer is refused.
- **Resumable:** each sample is `pred/<cell>/<row>.json` written `.part` → fsync → rename (w41's convention); a rerun
  skips finished samples. `--budget-s` stops before a sample that may not finish (per-token time of the worst sample
  so far × 1.3). Host-memory guard for radiance: no start below `mem_start_gb` MemAvailable, the engine group killed
  under `mem_kill_gb` (exit 4).
- **Scores** (RULER `scripts/eval`): reply stripped, control characters → newline; `string_match_all` (fraction of
  references found, case-insensitive) for niah / vt / cwe / fwe, `string_match_part` (any reference) for qa. A cell's
  score = mean ×100; a length's = mean over its tasks; **overall = mean over lengths** (RULER's average). Report adds a
  bootstrap 95% interval (rows resampled within cells).
- **Paired comparison** (`rlib.paired`, `rreport.py compare`): B − A on the rows both runs hold, same aggregation; 95%
  interval from 2000 paired bootstrap resamples (rows within each cell, the same rows for A and B — as `mix/paired.py`
  does with documents), per length / task / cell, plus flip counts.

## Selections (cost on one R9700, V2 4.00 bf16 KV, chunked prefill 2048: see RESULTS.md)

| selection | rows | use |
|---|---|---|
| `fast-v1` | 12 tasks × (8k + 32k) × 2 = 48 | gate probe: ~11 min of the 27B (5.5 s a row at 8k, 21 s at 32k); the V2 4.00 scores 100 on it, so it catches breakage, not small losses |
| `full-v1` | 12 tasks × 20 rows × 4k–64k + 10 rows × 128k = 1320 | final checks (~9 h of the 27B, extrapolated from the 8k / 32k rates; 64k / 128k not measured) |
| `kv-long-v1` | 7 probe tasks × 128k × 4 = 28 | KV finalists at long context |
| `lsa-v1` | 7 probe tasks × 128k / 256k / 512k / 768k × 8 = 224 | the D-LSA gate (YaRN past 256k) |
| `lsa-repo-128k` | the LSA repo's 128k probe rows (40) | comparison with their numbers |
| `dev08-v1`, `dev08-long-v1` | 240, 38 | the 0.8B validation |

## Proposed interfaces (the lead decides; nothing applied)

**kv_judge T2** (`patches/autoresearch-v3-kv-t2.patch`, against `kv-task` `1dbf8c6`): `tasks/kv/rlib.py` = a copy of
this `rlib.py` (stdlib, mypy --strict clean), uploaded with the other runner files; a model profile's `ruler` block
gains `set` (path of `ruler-qwen38-v1` on the bench host), `set_sha256` (`rlib.cells_sha256` of the selection's cells, checked),
`select` (a selection inline: `{"name": "fast-v1", "cells": {...}}`), `template`, `tol_points` (0), `tol_length_points`
(10) and optional `flags`. With `set` present T2 runs the candidate and the anchors (fp8; bf16 where it fits) on the same
rows over as many holds as needed (anchors cached per reference key), scores with `rlib.score`, and writes
`v['t2'] = {set, select, template, skipped, reports: {setup: ruler-report/1}, paired: {anchor: ruler-compare/1}, ok,
failed}`. **Rejected** when, against any anchor, the paired difference's 95% upper bound is below −`tol_points`
overall or below −`tol_length_points` at any length (the candidate is worse beyond chance). Without `set` the legacy
`ruler.py` path runs unchanged (the mock gate S13 uses it).

**LSA gate** (lsa.md "Gate" item 1): two runs on `select/lsa-v1.json`, the same quant and KV, full attention (A) vs D-LSA
(B), both with `--rope-scaling yarn:F:262144`: `rreport.py compare A B --json lsa-gate.json`. Proposed rule: pass when
`diff_ci95[0] > -5` (non-inferior by 5 points overall) and no length's `diff` below −10; the quants' own long-context
check is `rreport.py report A` against the bf16-KV / higher-bpw run of the same selection.

JSON schemas: `ruler-report/1` = `{overall, overall_ci95, overall_se, n, by_length{L}, by_task{t}, cells{t_L: {task,
length, n, score, all_correct, wilson95_all_correct}}, meta}`; `ruler-compare/1` = `{n_paired, a_overall, b_overall,
diff, diff_ci95, diff_se, flips_*, identical_scores, by_length{L: {a, b, diff, ci95}}, by_task{…}, cells{…}, meta}`.
Scores are on RULER's 0–100 scale.
