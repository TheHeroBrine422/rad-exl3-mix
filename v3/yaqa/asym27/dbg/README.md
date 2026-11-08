# w78 · why asym27's moments step died on gpubox (2026-11-07)

**Cause:** `asym27.sh` exported `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for every card step. With that
allocator mode, the 27B sibling-moments run (`sibmom27.py`) faults on the R9700 (gfx1201; torch 2.15.0+rocm7.14).
None of the code's kernels is at fault. The faults land in random **stock** kernels, 30 s – 2 min into the run:

- `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` in the RMSNorm mean reduce (`reduce_kernel<MeanOps>`, grid [83424,16,1]), the
  CK grouped conv of the reference `causal_conv1d`, and hipBLASLt bf16 / f32 GEMMs (`Cijk_*`);
- `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in an `at::native` kernel;
- then `illegal memory access`, reported asynchronously at the next sync: `y.cpu()` (sibmom27.py l.270).

Seven runs faulted: s1 ×3 and base ×3 on 11-07, plus the repro A here. One base run got 686k tokens into layer 0
before it died. The gate step (layers 2–3, 131k tokens, ~1 min) always passed, so the gate never saw it.

**Second problem, found while testing:** host RAM at the end of layer 0. Without expandable segments, layer 0 runs,
but the memory guard (3,000 MB MemAvailable) killed the run at the moment save (C, D).

- An open checkpoint shard (`_open`) kept the embedding's host copy (2.5 GB) alive through all of layer 0. RssAnon
  was 5.4 GB in layer 0 against 2.9 GB in layer 1.
- `save_lean` copied each 17408² f32 to the host whole (1.2 GB), and left ~3.6 GB of dirty pages per down group.
- iceman had ~9 GB available with nothing of ours running. That left ~3.1 GB during layer 0, ~100 MB above the guard.

## Runs (layer 0 or 0–1 of s1's level, `dbgmom.sh`; logs in `logs/`)

| run | allocator | other | tokens / layers | result |
|---|---|---|---|---|
| A | expandable | — | 400k / 0 | **ILLEGAL_INSTRUCTION at 30 s** (hipBLASLt bf16 GEMM) |
| C | default | — | 400k / 0 | no GPU fault; 377k tokens done, then killed by memguard at the save (avail 2.3 GB) |
| D | expandable | `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` | 400k / 0 | no fault (serialization hides it); memguard at the save |
| G | expandable | fixed sibmom27 | — | refused at start (the new guard) |
| E | default | chunked save | 400k / 0–1 | **MOMENTS DONE** 4 groups; peak RSS 6.3 GB |
| F | default | chunked save + shards dropped | 400k / 0–1 | **MOMENTS DONE**; peak RSS 3.5–3.8 GB (MemAvailable ≥ 6.0 GB in layer 0); moments = E bit for bit (`cmpmom diff`: 0) |
| s1 (real) | default | all fixes | 1M / 0–63 | layer 0: 165 s, 6,059 tok/s, peak RSS 3.5 GB, ETA 174 min (running 21:27 UTC) |

**Not shown:** why expandable segments fault here. It may be the VMM map / unmap racing with queued kernels, which
would fit "hidden by serialization", but that was not checked. Other ROCm jobs on this box (`swap27` quants, scores)
never set the variable. One `swap27` quant (11-06 08:04 UTC, `quant-xbn-c0.log`) had a single aperture violation
without it, so a rare fault of another kind also exists.

## Third problem: mkdenoise's f64 solve on ROCm (found when s1's moments finished, 11-08 00:15 UTC)

- **Symptom:** the first 17408 down group of the mk step ran > 20 min in `torch.linalg.solve(Gs, A.T)`, with rocBLAS
  "Device memory allocation size is too small for TRSM; TRSM performance is degraded". At that rate the 64 down groups
  would take > 20 h.
- **Bench** (`solvebench.py`, logs `solvebench1-3.log`; the f64 GEMM runs at 0.6–0.8 TFLOP/s):
  - `torch.linalg.solve` is fine up to 8192 (2.35 s);
  - at 17408 it takes > 6 min even alone on the card;
  - a Cholesky factor + blocked triangular solves (TRSM only on 1024-row diagonal blocks, the rest GEMMs) takes
    **16 s** at 17408, residual 1.2e-13.
- **Fix:**
  - `mkdenoise.py` copy + `mkdenoise-w78.patch`: only the lean path (k > 8192) uses the blocked solve, and falls back
    to LU if G_s' does not factor;
  - `stats.jsonl` gains `solve` and `solve_res`;
  - groups of 8192 or narrower take the default path, unchanged (so the 0.8B G1 bit-exactness holds).
- **Check on real moments** (`mkcheck.sh`, `mkcheck.log`), s1 layer 0:
  - **GDN out_proj forced through the lean path, old vs new:** hin bit-identical; wpre differs in 1,069 of 37.7M f32
    values (last-bit rounding), max rel 2.2e-10;
  - **down_proj, new:** 112 s for the whole group, including an eigh clip (3.1e-4), solve residual 4.4e-15, F_dev
    0.067, ‖WF‖/‖W‖ 1.018.
  - Also seen: hsub 0.585 on layer 0's down group (moment H vs calib H), reported, not judged here.

## Fix (in `v3/yaqa/asym27/`, live in `/data/asym27/code/`)

- `asym27.sh`:
  - no `expandable_segments`, and it unsets the three allocator-config variables;
  - DRY / STOP / missing pair never logs "DONE";
  - a STOP exits 3.
- `sibmom27.py` (`sibmom27-w78.patch` against `v3/sib27/sibmom27.py`):
  - refuses expandable segments on ROCm;
  - `save_lean` in 64 MB row blocks, synced and dropped from the page cache: byte-identical files;
  - `load()` drops open shards.
- `after-asym27.sh` (live: `/data/q1106/after-asym27.sh`; old copy `.pre-w78`) logs each arm's exit status. It writes
  "asym27: done" only after both arms have `pair.json` and the verdict printed; any failure ends in "asym27: FAIL".
  Status file: `/data/asym27/queue.log` ("ASYM27 QUEUE DONE").
- `go.sh` → `exec after-asym27.sh` with `MINMEM_GB=4`.
- `mkdenoise.py` (`mkdenoise-w78.patch` against `proxy2b/exp/w07-denoise/mkdenoise.py`): the blocked Cholesky solve
  on the lean path (above).
- Regression check: `test_w78.py` (CPU, seconds; `test_w78.out`, F–I).
