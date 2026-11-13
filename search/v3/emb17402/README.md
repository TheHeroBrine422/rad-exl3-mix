# emb17402 — trellis token embedding served from the stored form (w144, #174-02)

w135 (#174) found that a trellis-coded token table saves ~0.6 bit against V2's Lloyd tables at equal row error, but radiance
had no kernel that gathers a row from a stored trellis, and the VRAM rule bars expanding it at load. This folder holds the
quantization, donors, checks and menu for the kernel added on radiance-hero `qc-exl3-v3-emb144` (a clean merge into `qc-exl3-v3`, left to the lead):

- **exl3dense `embed_exl3i`** (`plugin/exl3dense/embed_exl3i.{h,cpp,hip}`): `x[m] = row tokens[m]` of an exl3i table,
  bit for bit the oracle's row (`exl3i_decode_rows`, RNE bf16) from the stored record, suh and svh. The output-side Hadamard
  spans 128 vocabulary rows, so a token's row needs its 128-row block: per 128 columns the workgroup decodes the block's 64
  tiles, rotates each row over k in the oracle's f32 order, and folds the rows along the cone of the token's element of the
  column butterfly. Codebook values come from a per-workgroup LUT indexed by byte sum (mul1 / mul1i8 / mul1i4, any rate
  m = 2..128, natural or R16 rows). Prefill (≥ 64 tokens): one workgroup per token, and repeated ids are decoded once (a
  second launch copies them). Decode: one workgroup per (token, 128 columns). There are no extra buffers: LDS only, no
  scratch, no expanded table.
- **exl3quant** imports `token_embd.weight exl3i src=<level>/src codebook=…` (source tensors
  `model.language_model.embed_tokens.*`; the old `exl3` quantizer declines it). **qwen35_exl3** declares the record / suh /
  svh views (whole table on every rank, like embed_tq) and issues `embed_exl3i`. The DFlash2 drafter refuses it by name, as
  it refuses embed_tq.
- The GPU test's `--host` runs the same launch on mapped pinned host memory (the host-RAM embedding case).

| file | what |
|---|---|
| `RULES.md` | pre-registration (identity / VRAM / time / pricing anchors) + amendment 1 (build control) |
| `build144.sh` | box2 / ds8: the three plugins against rad10-perf's install (core 5c9956c), host tests, `W/home` |
| `prep144.py` | the token table as a one-tensor "checkpoint" (`lm_head.weight`) + an identity input Hessian, for ourq's head path |
| `quant144.sh` | the levels: ourq.py `--only-head` with the head donors' settings minus H_out (`--qtf2 --frac-ext --scaled-hessian --sigma-reg 0.025 --codebook i8|mul1 --save-wq`), one level a job, first free card |
| `embsrc.py` | ourq output → the importer's source (`src/`, keys renamed), row error rel = Σ(w−ŵ)²/Σw² from `wq/lm_head.rowerr.npz` |
| `donor144.sh`, `donors144all.sh` | ds8: donors `/data/q27/emb/trellis/emb-<level>-K<k>.rad` (K0.125 pack `--reuse` + the token_embd line, as w135's embll.sh), pushed to box2 over the 10 GbE link |
| `embref.py` | the dequantized reference: a container copy whose bf16 token_embd = the oracle's rows (ctypes into exl3i_decode.c) |
| `mix144.sh` | box2 identity / perf containers on w138's K4.125 + R16 K4 + A8 trunk (l3 / tK / tK-ref) + the confounder check |
| `score144.sh` | KLD on a golden (box2 score.sh), first free card |
| `rowcheck144.{cpp,py}` | the host row on the REAL table vs the reference container's rows |
| `gpu144.sh` | `exl3dense_embed_exl3i_gpu_test`: device == host bit for bit + the kernel bench (`--tokens` real ids) |
| `perf144.sh` | VRAM as served, 8k prefill, decode step (w137's #175 protocol), `--profile-ops` per-op µs |
| `a8anchor144.sh` | ds8: A8 pricing anchors on s171b, w135's setup |
| `menu17402.py` | menu174 + the trellis options (bytes from the donors, err = w135's floor model on the measured rel) |
| `int8rel144.py` | row error (rel) of the simple int8 host tables: per-row RTN, G128, G128 + FWHT128 (amendment 2) |
| `embrtn144.py` | the int8 per-row RTN reference container (bf16 rows = the RTN dequant, served by embed_lookup) |
| `host144.sh` | ds8: K8 mul1 donor, the int8-row reference, the three host-candidate A8 arms (bf16 / t8m / i8row) |
| `patches/core-embed-exl3i-host.{patch,md}` | core / ABI change for a host-pinned exl3i table + the dual-embedding loader design (not applied) |

Data: `data/emb17402/` (rel.jsonl, donors-inspect.jsonl, scores, perf summaries); handoff and menu:
`rad-local/workers/w144/`. Results: LOG #174-02.

Rebuild: `git archive` qc-exl3-v3 → `/data/q27/w144/src` (+ `COMMIT`), `bash build144.sh` (on box2 or ds8).
