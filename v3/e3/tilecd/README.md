# Tile coordinate descent (#77 cd3) in the GPU quantizer: `ourq.py --tile-cd`

w50, 2026-11-06. The user adopted tile CD for **attn_qg only** (11-06) on the CPU proxy (w43,
`v3/e3/settle/README.md` item 3). This folder holds the GPU implementation, its checks against settle.py, its time
cost, and the 27B "does no harm" check (rule: `v3/e3/settle/CD27.md`).

**Runtime cost:** none. Codes only: same format, kernels, bpw, prefill and decode. Quantizer time: ×3.7 per tensor
that gets CD (below).

## What it does

After exllamav3's `ldlq()` has produced the codes for a linear, N sweeps revisit every 16-row block a of the
regularized weight (k, n), in order. Each block is re-encoded against the target that is optimal given every other
block's current error:

    T_a = Q_a + H_aa^-1 (H E)_a,   E = W - Q,   H = the metric LDLQ factored (damped)

- **Metric.** With `--scaled-hessian` (the ladder) H is scaled_ldl's P diag(su) H diag(su) P; otherwise it is the
  rotated H from finalize_capture_H.
- **Acceptance.** A tile (16 × 16) takes its new code only if it lowers tr(Eᵀ H E). The change is computed exactly,
  per column, and summed over the tile's 16 columns; columns are independent, so the tiles of one block are decided
  separately.
- **Search.** The trellis search is exllamav3's own (`quantize_tiles_multigpu`, i.e. qtf2 / exl3cb as installed), in
  ldlq()'s tensor-core layout. The stored indices of accepted tiles are replaced.
- **After CD.** The proxy error and the scale refit run on the result, as after plain LDLQ.
- **Reference.** The math is `trellis2.ldlq_cd` (settle.py's reference) in exllamav3's (k, n) layout.
- **Not implemented:** an output-side Hessian (`--hout`, ldlq_2hess). That combination exits with an error.

| file | what |
|---|---|
| `tilecd.py` | the module: `install()` wraps exllamav3's `finalize_capture_H` / `scaled_ldl` (they record the metric), `ldlq` (CD after it when the linear's quant_args say so) and `ldlq_2hess` (refuses); `args()` per linear; stats to `OUTDIR/tilecd.jsonl` |
| `ourq-tilecd.patch` | against `quantize/ourq.py` as committed in ca99b25 (md5 3cfe422b…): `--tile-cd N`, `--tile-cd-families F,F` (default attn_qg). Every added line is tagged `# +cd` |
| `cdselftest.py` | CPU check of the bookkeeping with a stub quantizer: the error falls at every sweep, the stored indices decode to the returned weight, the reported rel equals a direct recomputation |
| `gpucheck.sh`, `capq.py`, `gpureport.py` | the 4070 checks below (gpu.lock, VRAM capped at 7.2 GB by capq.py) |
| `cdprec.py` | CPU diagnostics on settle.py's proxy: CD in fp32 vs fp64, fp16 Viterbi costs, LDLQ order reversed (`CDPREC_REV=1`) |
| `cd27.sh`, `cd27verdict.py`, `insert-40-cd27.sh` | the 27B swap pairs (CD27.md) |

**Code stamp.** Lines tagged `# +cd` are hashed into the stamp only when `--tile-cd` is given, through `_io_canon`
(an `+io` line, itself outside the stamp). `--tile-cd` and `--tile-cd-families` default to `argparse.SUPPRESS`, so
without them the stamp's argument set is unchanged. With them, the stamp covers the `+cd` lines, `tilecd.py` and both
arguments. Without `--tile-cd`, a patched ourq.py gives the same stamps and bytes as today's (checked below), so the
patch can go into the live `/root/ourq` without breaking the resume of a running queue.

**Use:** `ourq.py ... --tile-cd 3 --tile-cd-families attn_qg`. ourq.py looks for `tilecd/` next to itself, else
`../v3/e3/tilecd` (as for exl3frac2).

Precision is fp32. `TILECD_DTYPE=float64` exists for the test below only and is not in the stamp: drop it when the
patch is merged.

## Checks (4070, 27B attn_qg = `self_attn.q_proj` of layers 3 / 19 / 43, ladder settings: qtf2, i8, scaled H, σ 0.025 / input 0.005)

### 1. Off = today, byte for byte: PASS

`gpucheck.sh check`, run at the same path, compares today's ourq.py ("orig") with the patched one without
`--tile-cd` ("off"), at K2 / 3 / 4:
- **9 of 9 shards are byte-identical.** Codes, scales and the stamp in the metadata all match.
- **stamp.json stamps are equal:** `8767a965…` (K2), `f3e5d189…` (K3), `1c94105e…` (K4).

### 2. GPU ratio vs the CPU proxy

The GPU ratio is the proxy error after the scale refit, cd3 ÷ off: tr(dW H dW^T) / tr(W H W^T) under the undamped
calib H, in the original basis. It is recomputed per output row from `--save-wq`'s weights, and matches ourq's
proxy_err to 4–5 digits.

The "same blocks" columns restrict that sum to the 128-row blocks settle.py sampled (its seeds 0 / 1). The CPU
columns are settle.py's cd3 ÷ LDLQ on those blocks.

| layer | K | GPU whole tensor | GPU same blocks s0 / s1 | CPU proxy s0 / s1 (geo) | GPU − CPU (geo) |
|---|---|---|---|---|---|
| 3 | 2 | 0.9868 | 0.9838 / 0.9894 | 0.9511 / 0.9322 (0.9416) | **+4.8%** |
| 3 | 3 | 0.9886 | 0.9850 / 0.9900 | 0.9503 / 0.9381 (0.9442) | **+4.7%** |
| 3 | 4 | 0.9886 | 0.9853 / 0.9899 | 0.9562 / 0.9253 (0.9406) | **+5.1%** |
| 19 | 2 | 0.9943 | 0.9937 / 0.9942 | 0.9927 / 0.9930 (0.9929) | +0.14% |
| 19 | 3 | 0.9956 | 0.9958 / 0.9960 | 0.9942 / 0.9952 (0.9947) | +0.09% |
| 19 | 4 | 0.9955 | 0.9958 / 0.9952 | 0.9948 / 0.9949 (0.9948) | +0.07% |
| 43 | 2 | 0.9951 | 0.9954 / 0.9948 | 0.9913 / 0.9924 (0.9919) | +0.33% |
| 43 | 3 | 0.9961 | 0.9951 / 0.9957 | 0.9951 / 0.9930 (0.9940) | +0.21% |
| 43 | 4 | 0.9957 | 0.9961 / 0.9958 | 0.9927 / 0.9933 (0.9930) | +0.28% |

- **Layers 19 and 43 are within ±0.5% at every K: PASS.**
- **Layer 3 misses by about 5%**, on the same rows too, so row sampling is not the cause.
- In the quantization domain (tilecd.jsonl, before the refit) the GPU ratios are nearly the same: 3 → 0.987–0.989,
  19 → 0.993–0.995, 43 → 0.995–0.996. The refit does not explain the miss either.

**Why layer 3 misses: LDLQ order, not CD.** Tested in order:
- **Precision:** ruled out.
  - The GPU in fp64 gives 0.9867 (fp32 0.9867).
  - settle.py's sweeps in fp32 on the CPU give 0.9324 (fp64 0.9322), on 3:attn_qg K2 seed 1.
- **LDLQ order:** this is the cause.
  - settle.py's LDLQ (`trellis.ldlq_trellis`) quantizes the input blocks first to last. exllamav3's `ldlq()` goes
    last to first.
  - With the proxy's input order reversed (`cdprec.py`, CDPREC_REV=1, same rows, signs and scale), 3:attn_qg K2
    seed 1 changes as follows:
    - LDLQ error **7.07e-4 → 5.62e-4 (−20.5%)**;
    - cd3 ÷ LDLQ **0.932 → 0.987**, against the GPU's 0.989 on the same block.
  - Layer 19 seed 1: LDLQ −1.9%, cd3 ÷ LDLQ 0.9930 → 0.9932 (unchanged).
  - Layer 3's input Hessian has one massive channel (diagonal 4,765× the mean). The likely mechanism: LDLQ in the forward
    order leaves an error there that CD then repairs, and exllamav3's order does not (not checked further).
- **fp16 Viterbi costs** (E3_F16=1, trellis.py's emulation of the GPU search): 0.9320, ruled out.
- **The order-corrected proxy against the GPU.** Layer 3, the same blocks, cd3 ÷ LDLQ, CDPREC_REV=1 (fp64):

  | K | CPU rev s0 / s1 (geo) | GPU same blocks s0 / s1 | GPU whole | GPU − CPU (geo) |
  |---|---|---|---|---|
  | 2 | 0.9836 / 0.9874 (0.9855) | 0.9838 / 0.9894 | 0.9868 | +0.13% |
  | 3 | 0.9838 / 0.9871 (0.9855) | 0.9850 / 0.9900 | 0.9886 | +0.32% |
  | 4 | 0.9854 / 0.9891 (0.9873) | 0.9853 / 0.9899 | 0.9886 | +0.13% |

  With exllamav3's order, layer 3 also passes ±0.5%. **All 9 (tensor, K) are then within ±0.33%.**

**Reading.**
- The GPU implementation agrees with the CPU reference wherever both quantize the same way.
- The CPU proxy's large attn_qg gain (w43: −2.2…−3.7% cd3, the basis of "attn_qg only") came mostly from layer 3
  under the proxy's forward LDLQ order.
- On the real quantizer, attn_qg gains −0.4…−1.3% (geo of the three tensors ≈ 0.992–0.993 at K2–4). That is still
  the largest gain of the families (check 4).

### 3. Time per tensor (4070, 27B q_proj 12,288 × 5,120, end to end from ourq's per-linear time)

| K | no CD | cd3 | ratio | CD alone (3 sweeps) |
|---|---|---|---|---|
| 2 | 6.7 s | 24.5–24.9 s | ×3.7 | 17.9 s |
| 2.5 | 8.0–8.1 s | 29.9–30.3 s | ×3.7 | 21.9 s |
| 3 | 6.3 s | 23.4 s | ×3.7 | 17.1 s |
| 4 | 6.2 s | 22.9–23.0 s | ×3.7 | 16.8 s |

- One sweep costs about one LDLQ pass: 320 sequential 768-tile searches.
- Peak VRAM 4.9 GB (capq.py).
- Per ladder level: +16 × ~20 s ≈ +5 min of card time on the 4070 for attn_qg only.

### 4. Every family on the card, K2.5 (`gpucheck.sh fam 3` and `fam 2,19`, whole layers)

The ratio is the proxy error after the refit, cd3 ÷ off (ourq's proxy_err). ffn_down is not measured: its CD needs
more than the 4070's 7.2 GB cap at 27B (OOM).

| tensor | family | cd3 ÷ off | time off → cd3 |
|---|---|---|---|
| 3.q_proj | attn_qg | **0.9885** | 8.3 → 29.8 s |
| 19.q_proj | attn_qg | **0.9951** | 8.2 → 29.8 s |
| 3.k_proj | attn_k | 0.9910 | 1.1 → 3.3 s |
| 19.k_proj | attn_k | 0.9936 | 1.2 → 3.3 s |
| 3.v_proj / 19.v_proj | attn_v | 0.9968 / 0.9979 | 1.1 → 3.2 s |
| 3.o_proj / 19.o_proj | attn_output | 0.9980 / 0.9961 | 4.3 → 15.3 s |
| 2.in_proj_qkv | ssm_qkv | 0.9971 | 7.0 → 25.0 s |
| 2.in_proj_z | ssm_z | 0.9965 | 4.1 → 14.8 s |
| 2.out_proj | ssm_out | 0.9960 | 4.3 → 15.4 s |
| 3.gate_proj / 3.up_proj | ffn_gate_up | 0.9972 / 0.9970 | 11.5 → 42.6 s each |

- **attn_qg is still the family with the largest gain on the card**, now about 0.5–1.2% (the proxy said 2.2–3.7%).
  attn_k is close (0.6–0.9%) and costs about 2 s per tensor. Every other family gains 0.2–0.4% at ×3.6 the time.
- **"attn_qg only" still ranks right.** Adding **attn_k** would cost ~32 s per level for 16 tensors and is the one
  cheap extension. This rests on two tensors per family (n is small), and it is the user's call.

## Open / for the lead

1. **Merge:** `ourq-tilecd.patch` into `quantize/ourq.py` + `tilecd.py`. Drop the `TILECD_DTYPE` test hook, or keep
   it out of use. Then copy both to the GPU boxes' ourq copies (no stamp change without `--tile-cd`). It is not an
   exllamav3 patch, so the `quantize/exl3-patches/` series is unchanged.
2. **Schedule** the 27B check (`v3/e3/settle/CD27.md`; insert `insert-40-cd27.sh`, ~1.3 h).
3. **Scope** (user): the measured gain is ~0.5–1.2% on attn_qg (proxy error, at ×3.6 its quant time), not 2–4%.
   attn_k is a cheap addition.
4. **settle.py's LDLQ order** differs from exllamav3's (first-to-last vs last-to-first). On 3:attn_qg it makes the
   proxy's LDLQ error 26% higher (7.07e-4 vs 5.62e-4, K2 seed 1); on 19:attn_qg 2%. Anything order-sensitive that
   settle.py measured on tensors with a massive input channel is biased (absolute errors; CD-like repairs). A
   per-tensor order choice in the real quantizer (first-to-last vs last-to-first, or by H diagonal) may be worth a
   proxy item: on this tensor the order alone is worth more than cd3.
5. **ffn_down CD** at 27B does not fit the 4070's 7.2 GB cap. On the R9700 it should. The memory is not optimized,
   and it only matters if CD is extended beyond attn_qg.

## Raw data

`data/e3/tilecd/gpu/`:
- `*.out`: the units' lock and time lines;
- `out/<arm>/K*/proxy.jsonl`, `tilecd.jsonl`, `stamp.json`, and `out/*.log`;
- `out/gpureport.json` (check 2).

Arms: orig / off / cd3 = check 1–2; cd3f64 = fp64; toff / tcd3 = time; foffL / fcdL = families.

`data/e3/tilecd/cpu/`: cdprec.py outputs (colo):
- `-rev` = exllamav3's order;
- `-f160` = fp64 vs fp32 sweeps;
- `f161` = fp16 Viterbi costs, fp64 sweeps: 0.9320. Its fp32 step was stopped, so the result is noted in the
  `.note` file.

GPU units (each one hold of `/data/yb/gpu.lock`, CST):

| unit | held | time |
|---|---|---|
| check | 6.8 min | 12:44–12:51 PM |
| time + fp64 | 9.0 min | 1:10–1:19 PM |
| fam 3 | 5.7 min | 1:35–1:41 PM |
| fam 2,19 | 9.2 min | 1:46–1:56 PM |
