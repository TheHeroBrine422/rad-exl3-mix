# gold27 — single-card bf16 27B reference goldens (staged layer passes)

`mkgold27.py` makes a bf16 reference golden of Qwen3.8-27B (or any Qwen3.5-family checkpoint) on **one** GPU, without
TP2: the model is cut into N stages of consecutive decoder layers, and each stage runs over the whole corpus before
the next loads. The output is the layout radiance's `--kld-ref` reads (`--kld-record --kld-top 512`):
`kld.json` (format `radiance-kld-top-1`), `tokens.i32`, `gen.u8`, `top.bin` `[rows, 512] {i32 id, f32 log p}`,
`rows.f32` `[rows, 3]` = nll of the true token, entropy, tail mass summed outside the 512. The rows are those
`proxy2b/mkgold.py` writes for the same documents (the arithmetic of `proxy2b/common.py`: log-softmax in f32 a chunk
of 256 rows at a time, full-vocabulary sums in f64).

**Runtime cost:** neither (an offline reference; no change to any model, kernel, prefill or decode).

## Usage

```
python3 eval/gold27/mkgold27.py MODEL CORPUS.jsonl OUT --scratch DIR [--stages N] [--scratch-dtype f32|bf16]
        [--split 0,22,44,64] [--docs LO:HI] [--only-stages A:B] [--attn-chunk Q] [--model-name NAME] [--keep-scratch]
VRAM_CAP_GB=30 ...                    # caps this process's allocator; MKG27_DEV=cpu: a slow CPU dry run
eval/gold27/run.sh CORPUS.jsonl OUT   # gpubox card 0 wrapper: N=3, bf16 scratch, card-0 lock per block (env in the header)
```

- **Corpus:** radiance's `--kld-corpus` JSONL (`xcorp/kldcorpus.py`: `{prompt, source, score_from}`), one document a
  line, empty lines skipped. Tokenised as radiance tokenises it: the raw prompt, control tokens parsed, nothing
  prepended (HF `tok(prompt, add_special_tokens=False)`; it reproduces the published calib golden's `tokens.i32` from
  `calib.jsonl` exactly, 479 of 479 documents checked). Positions `score_from .. n-2` are scored; `gen.u8` is all 1 (a
  line carrying a `gen` field is refused rather than ignored). `kld.json` has radiance's header (`model`, `corpus`,
  `corpus_fnv1a64` hashed as radiance does, `created`, `n_vocab`, `top`, `rows`, `docs` with `source`, `tokens`,
  `score_from`, `row0`) plus `kernels` (as mkgold.py) and `maker` (split, scratch dtype, attn chunk).
- **Stages:** `--stages N` splits the layers by weight bytes, counting the final norm + LM head (2.5 GB on the 27B) in
  the last stage; `--split` sets the boundaries explicitly. Stage 0 takes the embedding rows on the CPU (no 2.5 GB
  embedding on the card). Each stage is transformers' own `Qwen3_5DecoderLayer`s (bf16, `sdpa`), run on one whole
  document at a time with a fresh `DynamicCache` and positions 0 .. n-1 (rotary and masks per document, never
  packed), exactly as `Qwen3_5TextModel.forward` runs them.
- **Scratch:** the hidden state after each stage, `DIR/b<s>/<doc>.{f32,bf16}`, one boundary at a time (a stage
  deletes each input file once its output is renamed into place). **Use `--scratch-dtype bf16`:** the residual stream
  is bf16 in transformers, so bf16 storage is bit-exact (see the tests) and half of f32's disk; f32 is the default only
  because the task asked for it.
- **Restartable:** a document's progress is the highest boundary file it has (written `.part`, renamed, checked by
  size; stale or torn files are removed), its rows are fsync'ed into the preallocated `top.bin` / `rows.f32` before
  `DIR/done/<doc>` marks it, and `kld.json` is written last, once every document is marked (rerunning a finished OUT
  of the same corpus is a no-op). Kill it anywhere and rerun the same command. The stage plan (`DIR/plan.json`) is
  fixed at the first run; a different plan needs a fresh `--scratch`.
- **Splitting the work:** `--docs LO:HI` runs only those documents (all stages), so the scratch holds only them:
  `run.sh`'s `BLOCK=` loops over blocks, each block reloading the weights. Disjoint `--docs` ranges can also run on
  two cards at once with the same OUT and DIR (rows go to disjoint offsets; whichever finishes last writes
  `kld.json`). `--only-stages A:B` runs a subset of the stages (e.g. to fit a lock-hold limit).
- **Long documents:** `--attn-chunk Q` computes full attention a block of Q queries at a time with an explicit causal
  mask, for a backend where `sdpa` has no flash / memory-efficient kernel and the math path would hold
  `[24 heads, n, n]` scores (32k: ~100 GB). Off by default (exact transformers path). `run.sh` sets
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` so ROCm torch can use its flash kernel on gfx1201.

## Tests

`test08.sh` (Qwen3.5-0.8B, 4070): a 5-document public test corpus (`mktest.py`: transformers' modeling source 8k
tokens, the model card 3k, a rendered chat with control tokens and `score_from` 24, 1.2k and 101 tokens; 12,643
scored rows) through mkgold27 at N = 1, 2, 3 (f32 and bf16 scratch), each compared with `proxy2b/mkgold.py`'s rows on
the same documents by `cmpgold.py` (top-512 overlap, max |Δ log p| over shared ids, rows.f32 differences).
`test27.sh` (27B, 4070, N = 12 under a 7.2 GB cap): 2 documents (modeling source 8k + model card 3k), with the 0.8B on
the same 2 documents as a yardstick.

Results (4070, 2026-11-04 ~11:45 PM CST; raw in `results/`):

| run | vs `mkgold.py` (12,642 rows) |
|---|---|
| N = 1, 2, 3, f32 and bf16 scratch (5 runs) | **bit-identical**: top.bin and rows.f32 byte-equal, top-512 overlap 1.0, max \|Δ log p\| 0 |
| resume from partial runs (`--only-stages 0:1`, then `--docs 0:2`, then all; N = 3) | bit-identical |
| `--attn-chunk 1024`, N = 2 | not bit-exact by design: exact in fp64 (max diff 0 vs whole-document sdpa), in bf16 another kernel path: mean \|Δ nll\| 1.3e-2, mean Δ nll +2.3e-4, top-1 99.2 %, overlap ≥ 0.95. Don't mix goldens made with and without it |

27B smoke run (N = 12, `[0, 6, 11, 17, 22, 28, 34, 39, 45, 50, 56, 62, 64]`, bf16 scratch, peak 6.0 GiB under the
7.2 GB cap; 2 documents, 10,998 rows, whole run 2 min 20 s):

| document | 27B ppl | 0.8B ppl | 27B top-1 = next token | 27B entropy | 27B tail |
|---|---|---|---|---|---|
| transformers' modeling source, 8k tokens | 1.820 | 3.265 | 85.9 % | 0.468 | 1.8e-3 |
| the 27B model card, 3k tokens | 1.777 | 2.663 | 86.1 % | 0.416 | 9.1e-4 |

Head + tail mass sums to 1 within 2e-6 on every row; the head's log p are sorted.

Tokenisation: HF `tok(prompt, add_special_tokens=False)` reproduces the published calib golden's `tokens.i32`
(recorded by radiance) from `calib.jsonl` on 479 of 479 documents checked (every 10th).

## Disk and time

Per 1M scored positions: `top.bin` 4.10 GB + `rows.f32` 12 MB + `tokens.i32` 4 MB (the golden itself, as radiance's).
Scratch, peak: one boundary of the documents in the pass, **10.2 GB per 1M tokens in bf16** (20.5 GB in f32), plus
one document. For the 7.5M-position batch (xcorp 8k 5.0M + own traces 32k 2.5M, two goldens): 30.7 GB of goldens,
and 51 GB of bf16 scratch at the peak (the 5.0M corpus in one pass; `BLOCK=` to cut it).

Measured on the 4070 (27B, bf16, transformers 5.22.1, torch 2.11 + fla 0.6.0 / causal-conv1d 1.7.0):
- decoder layers 23.9 µs per token and layer (stages 1–10, 11k tokens; 1.53 ms per token for all 64), of which the
  GEMMs are about half (0.77 GFLOP per token and layer; torch bf16 GEMM 61–63 TFLOPS on these shapes, `gemmbench.py`
  beside this README);
- the head stage's rows (LM head, f32 log-softmax, top-512, f64 sums, writes) ~0.125 ms per row;
- the stage's weight load ~2 GB/s from a SATA SSD (4.3 GB in 8–9 s);
- the first stage of a process pays ~20 s of one-off warm-up (kernel compilation / autotuning).
So 1M positions take ~28 min on the 4070 (N = 12) plus ~2 min of loads per pass.

**R9700 estimate (to be confirmed by the first block's tok/s):** GEMMs at the ~140 TFLOPS that hipBLAS reaches on
these shapes (5.5 µs a token-layer) plus the non-GEMM part scaled by bandwidth (640 vs 504 GB/s: ~9 µs; up to
~18 µs if fla's triton kernels are slower on gfx1201) gives 15–24 µs a token-layer: **~16–26 min per 1M positions**
plus ~1.5 min for the head rows, and ~1 min of loads per stage and pass. **The 7.5M-position batch (two goldens):
~2.2–3.4 h on one card**, N = 3, bf16 scratch (51 GB peak for the 5.0M corpus), 30.7 GB of output. If fla is missing
or fails on ROCm (transformers then falls back to its torch GDN path; `kernels` in kld.json records which ran), expect
the GDN layers (48 of 64) to be several times slower: time the first block before committing the card.

## Scaling notes

- More stages cost almost nothing: every layer still runs once per token; a stage adds one weight load (~17 GB from
  disk at N = 3) and one write + read of the boundary (10 GB per 1M tokens in bf16). Choose N by VRAM: a stage's
  weights + the activations of the longest document must fit. The 27B is 64 x 0.76 GB of layers + 2.5 GB head:
  N = 2 puts ~25.6 GB on each stage, too tight on a 32 GB card next to a 32k document's activations (measured
  1.7 GB at 8k tokens on the 27B, so ~7 GB at 32k if linear: MLP intermediates, the GDN path's f32 copies, the head's
  256 x 248k f32 logits); N = 3 (~17 GB a stage) leaves room for 32k documents. On the 4070 (7.2 GB cap) N = 12.
- Per-document work is independent, so throughput is linear in tokens; the head stage adds a full-vocabulary
  log-softmax, top-512 and two f64 sums per position (memory-bound, ~1 MB a row in f32).
- Two cards: disjoint `--docs` halves, one process per card, same OUT and scratch.
- Backward collections (Σdy², Fisher) could reuse the same boundary files (forward boundaries kept with
  `--keep-scratch` and per-stage recompute), not built here.

## Files

| file | what |
|---|---|
| `mkgold27.py` | the staged golden maker |
| `run.sh` | gpubox card-0 wrapper for the lead (`ROCR_VISIBLE_DEVICES=0`, `/root/card0.lock` per block, bf16 scratch on `/data`) |
| `cmpgold.py` | compares two goldens of the same corpus (layout, overlap, Δ log p, rows.f32) |
| `mktest.py` | the public test corpus (no corpus data) |
| `test08.sh`, `test08b.sh`, `test27.sh` | the 0.8B correctness test, its `--attn-chunk` / resume follow-up, the 27B smoke run (4070, each under `gpu.lock` itself; never wrap them in another flock) |
| `gemmbench.py` | bf16 GEMM TFLOPS on the 27B's shapes, to scale timings between cards |
| `results/` | the tests' raw output (cmpgold lines, the 27B stage timings and per-document stats, logs) |

## Open issues

- **Not run on ROCm yet.** `run.sh` is untested on gpubox: check that fla / causal-conv1d import there (else the
  torch GDN fallback: correct, slower), that `sdpa` takes a flash kernel at 32k (else `ATTN_CHUNK=2048`), and time
  the first block. Suggested first step: `BLOCK=50 run.sh` and watch the first stage's tok/s.
- **HF reference vs radiance's.** These goldens are transformers' bf16 forward, not radiance's: candidates scored by
  radiance against them carry the kernel difference between the two bf16 paths (on the 0.8B, two HF kernel paths
  differ by ~1.5e-4 KL). Measure it once: take ~50 documents that a TP2 radiance golden of the 27B already covers
  (e.g. tonight's xcorp golden, or `golden-search27`), make the same documents with mkgold27 (a corpus of those lines),
  and score the bf16 golden against the other (or a quant against both). Paired comparisons against one golden are
  unaffected.
- **gen.u8 is all 1**; no corpus tool writes a generation mask yet (a corpus line with `gen` is refused).
- **One document per forward** at any length: a 32k document needs ~7 GB of activations on top of the stage's
  weights (N = 3 on a 32 GB card). Longer documents (whole 250k traces) need `--attn-chunk` and more stages; the MLP
  is not chunked over tokens (chunking would change the GEMM shapes, so the rows would no longer be mkgold.py's).
