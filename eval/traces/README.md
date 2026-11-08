# eval/traces — the long agentic calib traces as 32k turn-aligned chunks (w67, 2026-11-07)

Routine scoring of the 8 long (~250k-token) agentic calib traces uses chunks of ≤ 32k tokens. Long-context *capability*
is RULER's job (w66). Each chunk starts with the trace's system message (system prompt + tool definitions) as an unscored
header. Both verify traces are also kept whole for rare full-length checks.

**Runtime cost:** neither (evaluation data and tools only; no model, kernel, prefill or decode change).

## Pools and slices

Pools come from `eval/pools/pools-v3.json`'s per-document `pool` (sha256 `f0706518…`, identical on gpubox). A chunk
never leaves its trace's pool, and nothing moves between pools.

> **Pool file over task text.** The w67 task listed lua and requests under qad. The pool file (and AGENTS.md, amendment
> 2026-11-01) has them in **verify**, so they stay verify. Their whole traces are already the V3 line's `golden-ver3-long`
> (lua, V3 monitor) and `golden-ver4-long` (requests, V3 final). They are chunked in the manifest only: no corpus (unless
> `--reserved`), no golden.

| trace (calib doc) | pool | tokens | chunks | slice | whole copy |
|---|---|---|---|---|---|
| ripgrep (1241) | search | 245,774 | 9 | `agent-search32` | — |
| cobra (4029) | search | 245,199 | 9 | `agent-search32` | — |
| rich (905) | verify | 250,158 | 9 | `agent-verify32` | `agent-verify-whole` |
| express (2675) | verify | 253,163 | 9 | `agent-verify32` | `agent-verify-whole` |
| lua (2300) | verify | 248,066 | 8 | — (reserved: ver3-long) | — |
| requests (4723) | verify | 245,667 | 9 | — (reserved: ver4-long) | — |
| attrs (1772) | qad | 246,898 | 9 | `qad32` (QAD / healing data, no golden) | — |
| click (2651) | qad | 245,120 | 8 | `qad32` | — |

| slice | docs | tokens | scored rows | gen rows (the official score's) | max doc |
|---|---|---|---|---|---|
| `agent-search32` | 18 | 505,237 | 489,190 | 261,437 | 29,580 |
| `agent-verify32` | 18 | 517,561 | 501,541 | 275,375 | 30,657 |
| `qad32` | 17 | 505,368 | 490,238 | 237,168 | 32,761 |
| `agent-verify-whole` | 2 | 503,321 | 503,319 | 275,375 | 253,163 |

- **Whole verify traces.** The task first asked to keep one verify trace whole. The lead's note (11-07 ~12:10 AM CST,
  the user's concern about overfitting to one trace) changed that: both rich and express are chunked into
  `agent-verify32` **and** both are kept whole as read-only copies (`agent-verify-whole`).
  - The whole copies are only a "does anything break at 200k+" check on finished mixes, never a selection or tuning
    target.
  - Report both traces together, never one alone.
  - The golden is cut from the published calib golden (radiance TP2 reference, no bf16 pass).
  - At 250k tokens it needs TP2 for high-bpw mixes.
- **The chunks lose no gen rows.** A header token is never a gen row, so `agent-verify32` has exactly the whole traces'
  275,375 gen rows. Only their context is shorter (≤ 32k).

## Chunk rule (`chunk.py`)

- **Messages.** A trace is ChatML: one system message (the header: 890–893 tokens), then strictly alternating user /
  assistant messages. A user message is either a task statement (3–9 per trace) or a tool result (Qwen renders tool
  results as user messages).
- **Shape.** A chunk = header + consecutive messages starting at a user message. Cuts happen only before a user
  message, so no message, tool call or tool result is ever split. This is the shape `xcorp/kldcorpus.py --trace` uses
  for our own sessions; here the header is unscored.
- **Fill.** Chunks are filled greedily with whole turns up to a body cap.
  - If a task statement starts in the last quarter of the cap, the chunk is cut there, so the next chunk opens with its
    task.
  - The cap is bisected down to the smallest value that still gives the greedy chunk count, so a trace splits into
    nearly equal chunks (23.1k–32.8k tokens) instead of full chunks plus a stub.
  - Most tasks are longer than 32k, so most chunks start at a tool result.
- **Header mask.** Corpus lines carry `score_from = header − 1`, so the scored rows predict exactly the body tokens
  (rows = body tokens). Header targets are never scored.
- **gen** (rows the official score counts): tokens of an assistant message after `<|im_start|>assistant\n`, through its
  `<|im_end|>`. This is the published golden's rule, checked exactly against its `gen.u8` on all 8 traces.
  `mkgold27.py` writes gen all 1; `agent32.py finalize` replaces it with this mask.
- **Tokenisation.** `tokenizers` with the 27B's `tokenizer.json` (`encode(prompt, add_special_tokens=False)`, as
  radiance) reproduces the calib golden's `tokens.i32` for all 8 traces. Each chunk's prompt is cut from the original
  text at the `<|im_start|>` offsets, and its re-tokenisation equals `trace[:header] + trace[from:to]` (asserted).
- **Deterministic.** Two runs on the VM and one on gpubox (different tokenizers build) give byte-identical outputs.
  The frozen manifest is `manifest-32k.json` (sha256 `3aec8df0…`), and `agent32.sh chunk` refuses to go on if a rerun
  differs.

`manifest-32k.json` has, per trace: doc, source, pool, tokens + sha256, header length + sha256, messages, tasks, gen
rows, body cap. Per chunk: index, `from` / `to` (body token range in the trace), first message kind, tokens, header
length, `score_from`, rows, gen rows, prompt / token sha256. Per slice: file sha256, members, sizes.

Provenance record: `eval/pools/traces32-2026-11-07.json`.

## Goldens and the run (gpubox, `/data/agent32`)

`agent32.sh` runs on gpubox and is started by the q2 insert `/data/q1106/insert-61-agent32.sh` (copy here;
`insert-60` was the first try, see "Incident"). Each stage is idempotent, so a rerun resumes.

| stage | where | does |
|---|---|---|
| chunk | CPU | `agent32.py traces` (trace tokens / gen from the calib golden) + `chunk.py --gold`; manifest must equal `manifest-32k.json` |
| whole | CPU | `golden-agent-verify-whole` cut from the published calib golden (streamed, low RSS) |
| probe | card 0 | kernel ladder k1–k5 (below). Each set must pass the **gate**, then fit the longest chunk (qad32's 32,761 tokens; `VRAM_CAP_GB=30`). Nothing fits → chunks re-cut at 24k (`chunks24/`, slices `*24`) and the ladder again. `probe.env` records the set (`KSET`) and the projected golden time; `probe/` has every set's gate / probe logs, peak GiB per stage, run time |
| golden | card 0 | `mkgold27.py` (the single-card staged bf16 reference, `eval/gold27`) on `agent-search32` and `agent-verify32`, with the gate-passed kernel set, bf16 scratch, `VRAM_CAP_GB=30`, memguard 2.5 GB. Refused without a gate-passed set or if the probe projects > 3 h (`A32_MAXH`) |
| finalize | CPU | `agent32.py finalize`: tokens must equal chunk.py's, gen.u8 ← chunk.py's mask, `orig / chunk / from / to / header_tokens` into kld.json; sanity checks on every row |
| refgap | CPU | chunk 0 of each trace has the published golden's context exactly (the trace's own prefix), so it measures the HF-vs-radiance reference gap free (top-1 agreement, \|Δ nll\|, KL(published ‖ ours) on gen rows) |
| score | card 0 | 8 mixes on `agent-search32`: K2.5 `base`, `s1`, `baseswj`, `s1swj`, and the head-matched K4 `base4v4`, `bn4v4`, `base4swj`, `bn4swj`. Flags as swap27.sh's score (radiance 1.16.0 + plugins home, bf16 KV, TP1), only `--max-model-len 32768` |
| report | CPU | `report32.py` → `report.txt` / `report.json` |

- **`agent-verify32` is golden only, not scored.** Verify is for finished mixes.
- **Why mkgold27, not golden_pick.** golden-search27 / golden-verify27 were cut from the published golden with
  `golden_pick.py` (no bf16 pass). A chunk with a repeated header has a context that no published row has, so it needs
  its own bf16 pass. The 27B in bf16 does not fit one 32 GB card under radiance, and card 1 is not ours, so the pass
  uses the staged HF reference.
- **Paired comparisons are unaffected by the reference.** refgap reports how far the HF reference sits from radiance's.
- Locks per GPU step: `/root/card0.lock`, then `/root/ram.lock` shared.
- **Dry run (no card):** `A32_DIR=<scratch> A32_FAKE=1 RXM_WORK=<own lock dir> agent32.sh`. It uses `fakemkgold.py`
  (`A32_FAKE_OOM="3:0"` / `A32_FAKE_MAXTOK=N` simulate out-of-memory) and `fakeradiance.sh`.
- **Kept whole on purpose:** `golden-agent-verify-whole` (2 GB).

### The gate and the kernel ladder

**The gate.** mkgold27 runs on two documents whose context the published golden already has: ripgrep's first 2,288
tokens, and its chunk 0 (26,670 tokens, the trace's own prefix). Its rows are compared with the published ones on gen
rows (`agent32.py gate`). It passes if KL(published ‖ ours) ≤ 0.005 and top-1 agreement ≥ 0.95 on both documents.

- **Why those numbers.** Two bf16 references should differ by ~1e-4–1e-3 KL. K4 mixes score ~0.01, and a broken
  reference is off by nats.
- **Not calibrated on a known-good stack yet.** The 4070 was down; the measured gap is in the gate lines.
- **Positive control.** A chunk-0-shaped cut of the published golden gives KL 0 and top-1 1.

The ladder, in order:

| set | stages | `--attn-chunk` | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `P2B_BLOCK` (torch fallback) |
|---|---|---|---|---|
| k1 | 3 | 0 (sdpa whole document) | 1 (aotriton flash) | — (fla GDN kernels) |
| k2 | 3 | 2048 (explicit causal mask per block) | 1 | — |
| k3 | 4 | 2048 | 0 | — |
| k4 | 3 | 0 | 1 | `fla` (transformers' torch GDN) |
| k5 | 4 | 2048 | 0 | `fla` |

- **Reading the k1 gate.** It is the default stack, as used by `mkhin27.py`'s 27B H_in collection (`/data/hin27`, ≤ 2k
  windows). Its 2.3k result says whether that stack is right at short context; its 26.7k result says it at long context.

### Incident (2026-11-07, 07:46–07:56 UTC = 1:46–1:56 AM CST): a false alarm

- **What happened.** insert-60's probe ran mkgold27 with k1 on the 32,761-token qad chunk. It fit (peak 22.6 / 23.4 /
  22.7 GiB per stage, 73 s compute). Its DONE line said **mean nll 10.33 (ppl 30,626) at mean entropy 0.87**, and the
  first golden's chunk 0 printed ppl 57,305. w67 read that as a broken kernel and killed the golden (mkgold27's pid
  only; q2 went on to bn4s1). It was not broken.
- **Cause: the ppl is over all rows, and that number is normal for these traces.** mkgold27's ppl averages every scored
  row, user and tool turns included (its gen.u8 is all 1 until finalize). Inside user / tool turns the 27B puts
  p ≈ 0.98 on `<|im_end|>` at almost every position. So those rows have nll ≈ 11 at low entropy, in the published
  radiance golden too.

  | rows (published calib golden) | all-row ppl | gen-row ppl | non-gen nll | non-gen entropy |
  |---|---|---|---|---|
  | ripgrep chunk 0's 25,777 rows (892 .. 26,668) | **57,556** (mkgold27 k1 on ROCm: 57,305) | 1.375 (1,786 rows) | 11.75 | — |
  | rich, whole | 251 | 1.554 | 11.28 | 0.89 |
  | express, whole | 163 | 1.517 | 11.13 | 0.90 |

  On the same rows the means differ by 0.004 nats (57,556 vs 57,305): the k1 stack agrees with radiance's reference.
  The KL / top-1 gate is still the real check.
- **What it changes.**
  - No kernel bug is shown, and mkhin27's H_in (same stack) is not implicated.
  - `golden-agent-verify-whole` is the published golden's rows, so it is fine (gen ppl above).
  - The insert-61 gate stays as a cheap check of the ROCm stack (~5 min if k1 passes).
  - Log lines now say "all-row ppl"; for these slices a value around 1e4 is normal.
  - Lesson: compare a new reference with the old one on the *same rows* before calling it broken.
- **Kept for the record:** `/data/agent32/probe/bad-0756/` (the probe log, timing and the stale `probe.env`).

## Cluster bootstrap (`paired.py`, patch `paired-cluster.patch`)

The chunks of one trace are not independent: they share a repository, a task and the trace's state. `paired.py` gets an
optional cluster key:

```
paired.py GOLDEN A.rows B.rows [--cluster KEY [--block K]]
```

- **KEY.** A kld.json doc field (`source`, `orig`, …) or a JSON file with one label per document.
- **`--block K`.** Splits each cluster into blocks of K consecutive documents (trace × chunk blocks).
- **Output.** Without `--cluster` it is byte-identical to the original (the code path is unchanged). With it, the line
  also has `cluster`, `block` and `units`.
- **Applying it.** The patch applies to `mix/paired.py`. gpubox's `/data/swap27/tools/paired.py` is byte-identical to
  it, so copy the patched file there.
- **Tests:** `test_paired.py ORIG NEW SCRATCH`, 29 checks on synthetic goldens:
  - byte-identical default output;
  - singleton clusters and `--block 1` = the per-document bootstrap exactly;
  - cluster / block SE = an independent implementation;
  - a per-trace shift makes the cluster SE larger than the per-document one.

  All pass. On real data, the default output on `golden-search27` base → s2 equals q2's logged line byte for byte.

With only 2 traces per slice, the per-trace bootstrap has 2 units: too few to trust. `report32.py` shows three
intervals per pair:
- per chunk (18 units, paired.py's default);
- trace × chunk blocks of 3 (6 units), the honest one for these slices;
- per trace (2 units, shown only).

## Tests

| test | result |
|---|---|
| `test_chunk.py CHUNKDIR TRACEDIR POOLS` (independent of chunk.py: tokens = header + trace range, cuts at `<|im_start|>user` / after `<|im_end|>\n`, balanced tool tags, tiling, pools, reserved, whole lines, gen = calib gen, sha256s) | ALL PASS on the VM output, on gpubox's `/data/agent32/chunks`, and on a 24k `--reserved` cut |
| determinism (2 runs VM + 1 gpubox) | byte-identical |
| `test_paired.py` | ALL PASS (29) |
| dry run of `agent32.sh` (fake mkgold27 / radiance, own locks) | all stages; fake OOM → ladder; all-32k-OOM → 24k fallback; `A32_MAXH` guard; report32 self-check against paired.py (all 18 pair × bootstrap SEs equal) |
| refgap positive control (chunk-0-shaped sub-golden cut from the published golden) | top-1 agreement 1, \|Δ nll\| 0, KL 0 |
| gate ladder dry run (fake: k1 "bad", k2 OOM) | k1 fails the gate, k2 out of memory, k3 passes and makes the goldens; a stale `probe.env` without a gate pass is refused |

## Ledger

- **Roles.** `golden-agent-search32` (select, v3), `golden-agent-verify32` (check, v3) and `golden-agent-verify-whole`
  (check, v3, 0 decisions) are added to `ledger-roles.json`.
- **`golden-agent-verify-whole`.** Added with `ledger_add.py` (2 lines, verify).
- **The chunk goldens.** `ledger_add.py` refuses them: their corpus hash is the chunk corpus's, and it would keep only
  the first chunk's range per doc.
  - `ledger_add-chunks.patch` makes it accept goldens made here (`manifest_sha256` + docs with `orig / from / to`) and
    merges each document's chunk ranges into one line.
  - It is unchanged for plain calib goldens (tested: byte-identical ledger).
  - The 4 lines it gives are in `ledger-pending.jsonl`: append them once the goldens exist.

## Files

| file | what |
|---|---|
| `chunk.py` | the chunker (corpus + pools → slices, gen masks, tokens, manifest) |
| `manifest-32k.json` | the frozen manifest |
| `agent32.py` | traces / whole / finalize / refgap / gatecorpus / gate helpers (streamed, low RSS) |
| `agent32.sh` | the gpubox driver |
| `insert-61-agent32.sh` | the q2 insert (copy of `/data/q1106/insert-61-agent32.sh`); `insert-60-agent32.sh` the first try (no gate) |
| `report32.py` | per arm mean / p99 / top-1, per trace; pairs with three bootstraps |
| `paired.py`, `paired-cluster.patch`, `test_paired.py` | the cluster bootstrap |
| `test_chunk.py` | the independent chunk checks |
| `fakemkgold.py`, `fakeradiance.sh` | dry-run stand-ins |
| `ledger_add-chunks.patch`, `ledger-pending.jsonl` | the ledger for the chunk goldens |

## Results

(filled in from `/data/agent32/result.txt` after the insert ran; see below)
