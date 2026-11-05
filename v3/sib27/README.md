# sib27: #87 (sibling-sequential asymmetric target) for the 27B

**Question.** Does #87 transfer to Qwen3.8-27B? On the 0.8B, requantizing the writer tensors against inputs from
their own layer's already-quantized siblings, with the GPTAQ-style target W·F (λ 1, ρ 0.5), cut KL by 14% at K3 and
18% at K2 (`proxy2b/exp/w07-seqsib`). It also stacked on the diagonal H_out (`proxy2b/exp/d1103`, #87-02).

- **Writers:** o_proj / out_proj / down_proj.
- **Siblings:** q/k/v (GDN qkv/z) and gate/up.

At 27B there are **128 writer tensors** (64 down_proj, 48 GDN out_proj, 16 o_proj). Only those change; every other
tensor is the level's own.

## Design

1. **Moments (`sibmom27.py`).** The statistics, file names and metadata are those of `moments.py --mode sib`:
   - per writer group, H = Σxxᵀ, C = Σxηᵀ, N = Σηηᵀ;
   - η = the writer's input with the quantized siblings minus the clean input;
   - f32 sums per 32 windows, flushed into f64.

   It is streamed one decoder layer at a time, ModelOpt's layerwise loop (`docs/research/modelopt.md`):
   - **Boundaries:** layer *l*'s clean bf16 inputs for the whole token set are on disk (`b<l>.bf16`). Layer 0's input
     is recomputed from the embedding.
   - **Per layer:** load its weights and decoded siblings, push every window through it (writing `b<l+1>`), and rerun
     the attention/GDN block and the MLP on the same inputs with the quantized siblings, as `moments.py` does.
   - **Exactness:** the residual stream between layers is bf16 in the full model too, so the boundaries hold exactly
     what the full forward passes on. Each layer is called with the kwargs `Qwen3_5TextModel.forward` builds (rotary,
     masks), captured with recorder modules standing in for the layers.
   - **Resumable:** per layer (`.done/L<l>`). The consumed boundary is deleted. `OUT/STOP` stops before the next layer.
   - **Siblings:** taken from the trunk level itself.

2. **Siblings from a level (`deq.py`).**
   - **Storage:** ladder levels (`/data/v3ladder2/K<k>`, `ladder2.sh`) store only EXL3 shards; `--save-wq` is not used
     there.
   - **Why a decoder:** exllamav3's stock `reconstruct` decodes plain mul1. The ladder searches with the int8-rounded
     codebook (`exl3cb` swaps only the search kernels), so stock reconstruct would not give the level's weights.
   - **What `deq.py` is:** a pure-torch decoder transcribed from the plugin's C oracle (`exl3_decode.c`): trellis →
     16-bit states → mul1 value v.
   - **i8:** i8 = fp16(clamp(rint(v / 4K_INV), ±127) · 4K_INV) is a function of v (`v3/nf/check/i8round.py`), so the
     decode is exact.
   - **Assembly:** W = diag(suh)·H₁₂₈·Ŵ·H₁₂₈·diag(svh) in f32, then cast f16 → bf16 as the `--save-wq` files are.
   - **Scope:** i8 and mul1 at integer K only; i8c, mcg and fractional rates are refused.
   - **No sibling requant is needed.** If a level ever had to be reproduced instead, `ourq --save-wq --only <siblings>`
     with the level's args (`levelargs.py`) would do it.

3. **Metric and target (`mkdenoise.py`, patched).**
   - **Why the patch:** at 17408² the current file needs about 25–30 GB of f64 plus a 2.4 GB `triu_indices`. That fits
     neither on a 32 GB card beside anything nor in gpubox's 30 GB of host RAM.
   - **`mkdenoise-lean.patch`:** adds a lean path for groups wider than `MKD_LEAN_K` = 8192 (`MKD_LEAN=1` forces it). It
     is the same arithmetic in the same order, in place and blockwise, with C/N/H_sub reloaded for F. That gives at
     most ~5 k×k f64 matrices at once (~12 GB at 17408), the `eigh` clip case included.
   - **Bit-identical:** its files equal the default path's (`test_lean.py`).
   - **Small groups:** groups up to 8192 wide run the unchanged code.

4. **Driver (`sib27.sh`, gpubox card 0).** It follows `night27.sh` / `ladder2.sh`: card0 lock then a shared ram lock,
   a log under `/data`, per-step markers, and a STOP file. Steps:
   - **pre:** checks and the C-oracle decode check.
   - **gate:** speed on layers 2–3, giving an estimate for N; stops over `MAXH`.
   - **mom.**
   - **mk:** λ 1, ρ 0.5.
   - **dose:** F at N/2 vs N on layers 0 / 3 / 31 / 63.
   - **noop:** the writers requantized with the base's own args (from its `stamp.json`, `levelargs.py`) and the calib
     H must equal the base bit for bit.
   - **arm:** `ourq --only writers --hin dn/hin --wpre dn/wpre`, plus the base's args, so a `--hout` base (K3-dy)
     stacks automatically.
   - **asm:** base shards with the writers replaced (`assemble.py`) → `level-asym`, an ourq-style export for
     `exl3i src=…`.
   - **ctrl (optional):** a writers-only reseed at seed offset 7 → `level-rs7`.

**Windows.**
- **Default:** `WINS=moments`, the 0.8B arm's own list: qad docs cut into 2048-token windows, shorter docs ≥ 64 tokens
  as one window, shuffle seed 0. With the same calib token stream and pools that is the same 843 windows /
  1,002,035 tokens the 0.8B used (gpubox's golden meta should match the 4070's `calib-meta`; check `windows.json`).
- **`WINS=long`:** collect.py's list (whole 2048-token windows only, same shuffle; checked equal in `test_cpu.py` D).
- **Recommendation:** keep `moments`. It is the validated recipe, and short documents keep the subset H closer to
  calib's H, which mkdenoise's control-variate G = H_calib + (G_sub − H_sub) assumes.

**Choice of N: 1M tokens.**
- **Fewer tokens:** n/k for the 17408-wide down groups is already 57 at 1M, against 280 on the 0.8B. Going lower, e.g.
  256k (n/k 14), risks the noisy Wiener filter seen at small n (`w07-denoise/selftest/fsub-vs-cv.txt`).
- **More tokens:** 2M doubles every cost below.
- **Dose check:** `dose` reports ‖F_{N/2} − F‖ / ‖F − I‖. If it is large (≳ 0.5), F is still sampling noise and a 2M
  collection is the follow-up. A longer list keeps 1M as its prefix, so this could be a second collection over the
  suffix windows plus a summing script; that script is not written yet.

## Budget on gpubox (K3, N = 1M, card 0)

| resource | need | note |
|---|---|---|
| GPU memory, moments | ~15 GB | f32 + f64 accumulators 12.3 GB (down 3 × 17408²), one layer 0.8 GB, siblings 0.6 GB, activations < 1.5 GB |
| GPU memory, mkdenoise | ~12 GB (≤ ~15 GB with clip) | lean path |
| host RAM | ~4–8 GB peak | moment save of a down group (3 × 1.2 GB f32 + the safetensors buffer); boundary page cache bounded (fdatasync + fadvise every 16 windows); `anon rss` logged per layer |
| disk `/data` | ~450 GB at peak | boundaries 2 × 10.3 GB (transient); moments 262 GB; half snapshots 16 GB; dn 131 GB (hin 44 + wpre 87); 3 writers-only quants 3 × 2.9 GB; 2 assembled levels 2 × 9 GB. The moments can go after mkdenoise if no other ρ is wanted (−262 GB). `pre` checks free space |
| moments, time | **~2–3.5 h** | per token and layer 1.53 GFLOP bf16 (forward + reruns) + 2.04 GFLOP f32 (H, C, N). At 55 / 25 TFLOPS that is 7.0 ms/token (1.95 h); at 35 / 15 it is 11.5 ms/token (3.2 h). Plus ~15 min of I/O. `gate` measures it (layers 2–3 at 131k tokens) and prints the estimate |
| mkdenoise, time | ~1–2 h | 64 f64 solves at 17408 (~1.4e13 flop each; RDNA4 f64 is slow: ~30–60 s each); `eigh` for clip groups (minutes each; 10% of the 0.8B groups clipped at 1M) |
| requants | ~2 × ⅓ of a K3 level | noop + arm, 7.7 B of the trunk's 24.3 B weights; ctrl another ⅓ |
| assembly | minutes | CPU, one shard at a time |

Total: about 5–7 card-hours before the evals.

**Runtime cost (user rule, 2026-11-03): none.** This is quantizer-side only: same format, same kernels, same bpw, so no
change to prefill or decode. The cost is at quant time: ~4–5 card-hours per (level, base seed).
- **If adopted for every ladder level:** ~30 levels × ~5 h ≈ 150 card-hours. Whether one level's moments serve its
  neighbours is a separate test before any rollout.
- **0.8B caveat:** proxy KL runs use the f16 path. At the A8-shipped sizes, a relative gain shrinks by A8's constant
  KLD.

## Run (gpubox, once it is back)

From the VM:

```
cd /workspace/rad-exl3-mix; I=local/bin/ssh-gpubox.sh
tar -C v3/sib27 -cf - sibmom27.py deq.py assemble.py levelargs.py cmpmom.py sib27.sh lean/mkdenoise.py \
  | $I 'mkdir -p /root/sib27 && tar -C /root/sib27 -xf - && cp /root/sib27/lean/mkdenoise.py /root/sib27/mkdenoise.py'
$I 'grep -c -- "\"--wpre\"" /root/ourq/ourq.py'   # 0: copy quantize/ourq.py over (diff first; the noop step proves it unchanged otherwise)
```

On gpubox:

```
nohup bash /root/sib27/sib27.sh < /dev/null > /data/sib27/K3.out 2>&1 &   # BASE=/data/v3ladder2/K3 TAG=K3 by default
# stack on H_out: BASE=/data/v3night/K3-dy/K3 TAG=K3-dy (its own siblings: a second moment collection)
# writers-only floor: STEPS=ctrl bash /root/sib27/sib27.sh
# stop: touch /data/sib27/K3/STOP (between steps / layers); rerun resumes
```

- **Logs:** `/data/sib27/<TAG>/sib27.log`.
- **Last line:** the `exl3i src=… codebook=mul1i8 rate=3` recipe for the d0.sh-style pack of `level-asym`.
- **Checks to run first:** `DRY=1` prints every command without running anything. `STEPS="pre gate"` alone gives the
  time estimate.

## Tests

- **`test_cpu.py`:** done on the 4070's CPU, real 0.8B, the K3 base, 2,923 tokens of qad windows, layers 0–3; output
  in `test_cpu.out`. **ALL PASS.**
  - **A.** `sibmom27 --wq` = `moments.py --mode sib`, bit-exact: H, C, N, n and metadata (tolerance stated 1e-6;
    measured 0).
  - **B1.** `--level` = `--wq` on `deq.py wq`'s files, bit-exact.
  - **B2.** Against the real `--save-wq` siblings: H is bit-exact. C/N/G move by 1–5% / ~2e-3, which equals the floor
    of a same-size (3.6e-4) perturbation of the siblings. Any sibling change re-draws the bf16 rounding of the rerun.
  - **C.** Resume after layer 1 = one run, and consumed boundaries are deleted.
  - **D.** `--wins long` = collect.py's order.
  - **E.** Half snapshots leave the full sums unchanged.
- **`test_deq.out`:** Ŵ is bit-exact against the C oracle (6 tensors). All 150 tensors of `q-k3base` decode within
  4.1e-4 of `--save-wq`, which is the fp16 rounding of suh / svh; the wrong codebook gives ~9e-3.
- **`test_lean.py`:** done on the 4070's CPU, real 1M-token moments of layers 0 / 3 / 10 / 11 / 14, including 5 groups
  that clip; output in `test_lean.out`. **ALL PASS.**
  - The patched default path = the repo's (files and stats).
  - The forced lean path = the repo's: hin / wpre bit-identical, stats within 3e-14.
  - λ = ρ = 0 gives the calib H bit-exactly.
  - Peak RSS 1.2 vs 2.3 GB at k = 3584.
- **`test_smoke27.out`:** the real 27B checkpoint on the 4070's CPU, one 256-token window. Layers 0–2 run
  forward-only (GDN torch path) and layer 3 (attention + MLP) collects moments with its own weights as siblings. The
  config, key mapping and kwargs capture work, the shapes are 6144² / 17408², and C, N ≈ 0 (trN/trH ≤ 4e-9).
- **`test_gpu.sh`:** for the lead, 4070 GPU, ~30–40 min; not run (no GPU for workers).
  - **Moment rule:** the floor is `moments.py` run twice. If the floor is 0, ours (`--wq`) must be within 1e-6 (max
    rel); otherwise every stat must be within 2× the floor.
  - **mkdenoise:** the lean path on the GPU against w07's dn files (reported).
  - **End to end, the 27B path:** level-decoded siblings → sibmom27 → mkdenoise → the 48 writers requantized → search
    KL. **PASS** if |log(KL / KL(k3asym))| ≤ 0.01 (w07's writers-only control SD was 0.0083) and KL / KL(k3base) ≤ 0.87
    (w07: 0.8585).
- **Also checked:** `assemble.py same` passes on w07's q-noop (192 / 192 tensors) and `build` is idempotent.
  `levelargs.py` reproduces q-k3base's ourq args. `sib27.sh` passes a `DRY=1` run.

## Draft pass rule: 27B swap test (for the lead to commit before data)

**Arms.**
- **K3:** the ladder's level (`/data/v3ladder2/K3`, seed 0).
- **K3-asym:** K3 with the 128 writers requantized on sibling moments (λ 1, ρ 0.5, same seeds, 1M qad tokens).
- **K3-s1:** night27's full-trunk rotation-seed-1 quant, the quant-noise floor.
- **Optional, K3-rs7:** writers-only reseed (`ctrl`), a tighter floor for a writers-only change.

**Scoring.**
- **Path:** the same as H27.md's swap test, i.e. whole uniform-K3 trunk, d0.sh path, bf16 KV.
- **Slice:** a search-pool slice of ~100k scored positions (8k cap), the same slice as the H_out test.
- **Statistics:** paired per document, bootstrap over documents (`mix/paired.py`).

**Rule.**
- **Noise floor:** n = |log(KLD(K3-s1) / KLD(K3))|. With K3-rs7 run, n = the larger of the two.
- **PASS (real):** r = KLD(K3-asym) / KLD(K3) has its bootstrap 95% upper bound < 1 **and** |log r| ≥ 2n.
- **Worth it:** runtime cost is none, so any real gain is adopted as a quantizer setting.
  - The 0.8B gave −14%. The expectation at 27B is ≥ −5%, since the sibling share of the input error and n/k are both
    smaller there.
  - If r ≤ 0.95, run K2.5 next. Before a ladder-wide rollout, test whether one level's moments serve its neighbours.
  - If H27's K3-dy is adopted, the stack test is next: `BASE=K3-dy`, its own siblings, the same rule against K3-dy.
- **FAIL:** r ≥ 1.00. Then log it in BACKLOG "proxy negatives to reconsider", with the mechanism (n/k 57 vs 280;
  siblings' share of the upstream error falls with depth, w07-seqsib README).
- **INCONCLUSIVE otherwise:** not adopted. If the dose check showed F still noisy (‖F_{N/2} − F‖ / ‖F − I‖ ≳ 0.5), the
  one retry is 2M tokens.
- **Gates before scoring:** the noop gate must pass. If it fails, K3-rs7 is mandatory and n includes it. The plausibility
  of the mkdenoise stats is reported: clip share, hsub, F_dev, ‖WF‖/‖W‖ within ~1 ± 0.01 as on the 0.8B.
- **Verify:** read once, after the decision, on a verify slice built from clean verify docs (`provenance.py`). It is
  reported, never iterated on.

## For the lead

- **Merge:** `mkdenoise-lean.patch` into `proxy2b/exp/w07-denoise/mkdenoise.py` (`git apply`; `test_lean.py` is the
  no-op test).
- **Ledger:** the moment collection reads qad-pool calib docs. Add it to `eval/pools/ledger.jsonl` (`ledger_add.py`,
  with the role used for #87's 0.8B moments).
- **Docs:** INDEX row and LOG entry for "#87 at 27B", Runtime: none.
- **Decide:** N (1M recommended); `WINS` (moments recommended over collect.py's long-only list); whether to run
  `ctrl`.

## Files

| file | purpose |
|---|---|
| `sibmom27.py` | layer-streamed sibling-moment collector |
| `deq.py` | EXL3 level decoder (mul1 / i8) + `wq` / `check` / `oracle` CLIs |
| `lean/mkdenoise.py`, `mkdenoise-lean.patch` | the lean mkdenoise path |
| `levelargs.py` | a level's ourq args from its `stamp.json` |
| `assemble.py` | `build` / `same` (no-op gate) |
| `cmpmom.py` | `diff` / `verdict` / `fdose` |
| `sib27.sh` | gpubox driver |
| `test_cpu.py`, `test_lean.py`, `test_gpu.sh` (+ `.out`) | tests |
