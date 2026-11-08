# #87 at 27B: writers' sibling-sequential asymmetric target (ρ 0), swap-test rule

Written 2026-11-05 ~9:55 PM CST (03:55 UTC 11-06), before any 27B asym data. Code: `v3/yaqa/asym27/`.

## Question and arms

Does #87 transfer to the 27B? On the 0.8B, the change (writers asym ρ 0) gave:

| comparison | K3 | K2 | source |
|---|---|---|---|
| vs base, ρ 0.5 | −14% | −18% | #87, every seed |
| stacked on H_out | −12% | −15% | #87-02, 6 seeds |
| ρ 0 vs ρ 0.5 | a further −5.5% | a further −8.4% | #87-05 |
| at K4 / K5 | holds | holds | #87-04 |

What the change does, per level:

- **Which tensors:** the 128 writers, i.e. 64 down_proj, 48 GDN out_proj and 16 o_proj.
- **The target:** they are requantized against W·F. F is the Wiener filter from moments of the writers' inputs, taken
  through the level's **own** quantized siblings (q/k/v, GDN qkv/z, gate/up): H = Σxxᵀ, C = Σxηᵀ, N = Σηηᵀ over
  1M qad tokens. The metric is G = H_calib + (G_sub − H_sub), with λ 1 and ρ 0.
- **What stays:** every other tensor is the level's own.

**Arms.** Two seed pairs (H27 amendment 4), K2.5, no H_out, ladder settings (qtf2, scaled H, σ 0.025 / 0.005, i8):

| arm | level (siblings + every non-writer tensor) | control (paired) |
|---|---|---|
| `basea` | `/data/v3ladder2/K2.5` (seed offset 0) | `base` (`/data/swap27/base.json.rows`) |
| `s1a` | `/data/swap27/q-s1/K2.5` (seed offset 1) | `s1` (`/data/swap27/s1.json.rows`) |

**Scoring.** Scoring is exactly the swap test's (`swap27.sh` / `swap27k.sh` functions, sourced unchanged):

- **Build:** each arm's assembled level is packed as exl3i (mul1i8, rate 2.5) and mixed into **V2 3.00** in place of its
  336 trunk weights, then checked by census. The plugin home is `/data/swap27/home` on `/data/install-116`.
- **Golden:** `golden-search27`, a search-pool slice of 103 documents and 123,460 scored positions, with bf16 KV.
- **Statistics:** `tools/paired.py` gives a per-document paired difference with a bootstrap SE.

## Pass rule

r_i = KLD(arm_i) / KLD(control_i) for i = base, s1, and g = √(r_base · r_s1).

- **PASS:** g ≤ 0.97 **and** both r_i < 1.
  - **Adopted:** writers asym ρ 0 becomes a V3 quantizer setting.
  - **Order:** the writers are requantized **last**, on each level's final shipped siblings. The correction does not
    transfer across sibling roundings (#87-07). If swj / H_out are adopted, the order is H_out → swj → writers
    (#87-08).
- **FAIL:** g ≥ 1.00. Logged in BACKLOG "proxy negatives to reconsider", with the mechanism to check:
  - n/k: the 27B down group sees 57 tokens per input dimension at 1M tokens, the 0.8B's 280;
  - the siblings' share of the writers' input error.
- **INCONCLUSIVE** (anything else): not adopted.
  - **One third pair, pre-registered:** if both r_i < 1 and g ≤ 0.99, run `s2a` on `/data/swap27/q-s2/K2.5` (seed
    offset 2, control `s2`, from q2.sh). Then apply the same rule over three pairs: geometric mean ≤ 0.97 and every
    r_i < 1.
  - No other extension.
- **Reported, not gating:**
  - each pair's z;
  - the asym arms' own seed floor (`basea → s1a`), next to the controls' 0.975;
  - mkdenoise's plausibility stats: clip share, hsub, F_dev, ‖WF‖/‖W‖ (0.8B: within ~1 ± 0.02);
  - the dose: ‖F_{N/2} − F_N‖ / ‖F_N − I‖ on layers 0 / 3 / 31 / 63. If it is ≳ 0.5, F is still mostly sampling noise
    at 1M, and a 2M collection is the follow-up.
- **No-op gate (per arm, before scoring):**
  - **Test:** the writers requantized with the level's own args and the calib H must be bit-identical to the level.
  - **If it fails:** the control becomes that no-op level (`<arm>n`, packed and scored by the driver). Otherwise a
    code or build change since the level was made would be read as the asym effect.
  - **Expected:** `base` (11-02, before the gpubox rebuild: other torch / ROCm) may well fail; `s1` (11-05, current
    code) should pass.
- **Optional K4 pair**, same rule, read separately:
  - `base4a` on `/data/swap27/q-base4/K4` (control `base4`);
  - `base4s1a` on `/data/swap27/q-base4s1/K4` (control `base4s1`, q2.sh);
  - mixed into V2 4.00.
- **Verify:** read once, after the decision, on a clean verify slice (`provenance.py`). Never iterated on.
- **Runtime cost: none.** This is quantizer-side only: same format, kernels and bpw, so no change to prefill or
  decode. The cost is at quant time: ~4–5 card-hours per (level, seed), dominated by the moments. Two things follow
  from #87-07 (no transfer across siblings):
  - **Rollout:** every shipped level needs its own run, about 100+ card-hours for a ~25-level pass-1 ladder. Before a
    rollout, test whether one level's moments serve its neighbours.
  - **Order:** anything that later changes a sibling, healing included, must come before the writers' run, or be
    restricted to the writers' scales.

## Commands (gpubox; the lead queues them, card 0)

**Stage** the code (from the VM; recopy if any of these files change):

```
cd /workspace/rad-exl3-mix; I=local/bin/ssh-gpubox.sh R=/workspace/rad-exl3-mix
$I 'mkdir -p /data/asym27/code' < /dev/null
tar -cf - -C $R/v3/yaqa/asym27 deq.py sibmom27.py asym27.sh verdict.py -C $R/v3/sib27 levelargs.py assemble.py cmpmom.py \
  -C $R/proxy2b/exp/w07-denoise mkdenoise.py | $I 'tar -C /data/asym27/code -xf -'
```

**Dry run** (prints every command; runs levelargs, writes `writers.txt`, builds the oracle .so):

```
DRY=1 ARM=s1 BASE=/data/swap27/q-s1/K2.5 CTL=s1 bash /data/asym27/code/asym27.sh
```

**The two K2.5 arms** (sequential; each is idempotent, so rerunning the same line resumes; stop with
`touch /data/asym27/<ARM>/STOP`). As a q2.sh insert hook, put these lines in `/data/q1106/insert-<NN>-asym27.sh`:

```
cd /data/asym27
ARM=s1   BASE=/data/swap27/q-s1/K2.5 CTL=s1   bash /data/asym27/code/asym27.sh >> /data/asym27/run.out 2>&1
ARM=base BASE=/data/v3ladder2/K2.5   CTL=base bash /data/asym27/code/asym27.sh >> /data/asym27/run.out 2>&1
/data/venv-exl3/bin/python -I /data/asym27/code/verdict.py /data/asym27 base s1 \
  --paired /data/swap27/tools/paired.py --golden /data/swap27/golden-search27 | tee -a /data/asym27/result.txt
```

- **Order:** s1 goes first. Its no-op gate should pass, so it is the clean pair. base's may need the extra `basen`
  control.
- **Optional pairs:**
  - **K4:** `ARM=base4 BASE=/data/swap27/q-base4/K4 CTL=base4 V2=/data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad`,
    likewise `base4s1` once q2.sh has scored it;
  - **third K2.5 pair:** `ARM=s2 BASE=/data/swap27/q-s2/K2.5 CTL=s2`.
  - Verdict: `verdict.py /data/asym27 base4 base4s1`.
- **Outputs:**
  - log: `/data/asym27/<ARM>/asym27.log`;
  - all arms: `/data/asym27/result.txt`;
  - scores: `/data/asym27/<ARM>a.json(.rows)`;
  - pairs: `/data/asym27/<ARM>/pair.json`.
- **Cleanup once logged:** `rm -rf /data/asym27/<ARM>/{mom,momc,dn-*}` frees ~410 GB an arm. Keep `mom` only if
  another ρ is wanted. Remove `q-noop` / `level-noop` if the no-op gate passed.
- **Ledger:** the moments read 843 qad windows (1,002,035 tokens, `windows.json`): add them with the role #87's 0.8B
  moments have.

## Budget per arm (K2.5, card 0)

| step | card time | VRAM | host RAM (anon) | disk | basis |
|---|---|---|---|---|---|
| pre (oracle on 4 tensors, CPU) | — | — | ~1.5 GB | — | |
| gate (layers 2–3, 131k tokens) | ~5–8 min | as mom | as mom | 2.6 GB transient | prints the mom estimate; stops over MAXH 5 h |
| mom (1M tokens, 64 layers) | **~2.5–3 h** | ~15 GB | ~3–4 GB (low-RAM save; without it ~9 GB) | boundaries 2 × 9.8 GiB transient, moments 262 GB, half 16 GB | mkhin27 on this card: 62 µs per token·layer for a forward + 0.79 GFLOP of f32 Grams; sibmom27 runs a forward + a sibling rerun + 2.05 GFLOP (H, C, N) per token·layer |
| mk (8 chunks) + dose | ~0.5–2 h | ≤ 15 GB (lean, f64) | ~4 GB | dn 131 GB | f64 on RDNA4, not measured on the 27B |
| noop (128 writers) | ~15–20 min | as a quant | ~2.2 GB | 2.4 GB | s1's quant log: 862 s of writer tensor time, anon RSS 2.2 GB |
| arm (128 writers, --hin / --wpre) | ~20 min | as a quant | ~2.2 GB (+1.2 GB Ft per down) | 2.4 GB | |
| asm | — (CPU, minutes) | — | ~0.5 GB | 7.1 GB | one shard at a time |
| pack + mix + census | — (CPU, conv.lock) | — | as every swap arm | mix ~10 GB | 955–1,700 s in swap27 |
| score | ~3 min | as every swap arm | headroom 8 GiB | — | 138–161 s in swap27 |

- **Per arm:** ~3.5–5.5 card-hours, ~4–6 h wall time. The two K2.5 arms take ~8–11 h, plus ~0.5 h per extra `<arm>n`
  control (pack + score) if a no-op gate fails.
- **Disk:** peak ~450 GB per arm, ~900 GB for both before cleanup. gpubox had 3.1 TB free (11-06).
- **Host RAM:**
  - gpubox has ~10 GB available when nothing of ours runs (the user's card-1 server holds the rest).
  - Every card step of the driver starts only with MemAvailable ≥ 6 GB (`MINMEM_GB`).
  - Each step runs under its own `memguard.sh` at 3,000 MB, above the queue's 2,500, so ours dies first.
  - Restart costs on a kill: sibmom27 loses at most one layer (~3 min), mkdenoise one chunk, ourq one layer.

## Changes to shared code (as copies + patches; the lead merges)

- **`asym27/deq.py` = `v3/sib27/deq.py` + `deq-frac.patch`:** decodes every exl3i rate in 1/8 steps (fractional K,
  sub-1 salted steps), not only integer K.
  - **Why:** the K2.5 swap arms are qtf2 fractional; the old `deq.py` refused them.
  - **Integer K:** bit-identical to the old decode.
  - **`oracle`:** now checks against the plugin's exl3i oracle (`exl3i_decode.c`, every rate, both codebooks), with
    the record laid out exactly as `exl3quant`'s `quantize_i` imports it.
- **`asym27/sibmom27.py` = `v3/sib27/sibmom27.py` + `sibmom27-lowram.patch`:** moment files are written one tensor
  at a time.
  - **Why:** `save_file` held 3 × 1.2 GB of f32 plus a serialized copy (~7 GB anon) at each 27B down group. With
    ~10 GB available on gpubox that ran into the memory guards.
  - **Output:** same tensors and metadata (test E, and G1m on the GPU).
- **mkdenoise:** the merged `proxy2b/exp/w07-denoise/mkdenoise.py` (lean path) is used as is.
- **Driver:** `asym27.sh` follows `v3/sib27/sib27.sh`, with these changes:
  - ρ 0;
  - a fractional-rate oracle check;
  - mkdenoise in resumable chunks;
  - a memory floor and guard;
  - the no-op result chooses the control;
  - pack / mix / score / pair through the swap test's own functions.

## Files (`v3/yaqa/asym27/`)

| file | purpose |
|---|---|
| `asym27.sh` | gpubox driver, one arm per run (env ARM / BASE / CTL / V2) |
| `verdict.py` | the pass rule over the arms' `pair.json` |
| `deq.py`, `deq-frac.patch` | level decoder for every rate in 1/8 steps (copy + patch of `v3/sib27/deq.py`) |
| `sibmom27.py`, `sibmom27-lowram.patch` | layer-streamed sibling moments with a low-RAM save (copy + patch of `v3/sib27/sibmom27.py`) |
| `scalecheck.py` | `--save-wq` vs shard: per-row / per-column scale factors (explains test D's fallback tensors) |
| `test_cpu.py`, `test_cpu.out` | CPU tests A–E |
| `test08.sh`, `test08.out` | the 0.8B GPU gate G1–G3 |

Also used, unchanged: `v3/sib27/{levelargs,assemble,cmpmom}.py`, `proxy2b/exp/w07-denoise/mkdenoise.py`, and gpubox's
`/data/swap27/tools/{swap27,swap27k}.sh` (functions sourced read-only), `paired.py` and `memguard.sh`.

## Tests

**CPU, 4070 box (`asym27/test_cpu.py`, output `test_cpu.out`).**

- **A.** `deq.py oracle` bit-exact vs `exl3i_decode.c` (W_hat in mul1 and i8; W within 3e-7, the f32 Hadamard order).
  Covered: real 27B K2.5 tensors (hconv's layers 10 / 39 / 50: in_proj_z, out_proj, down_proj, k_proj, o_proj,
  up_proj), a 0.8B K1.5 level and the 0.8B K3 base.
- **B.** 16 synthetic rates (K0.125 … K8, odd eighths, sub-1 salt), both codebooks: bit-exact.
- **C.** Integer K = the old `deq.py`, bit for bit.
- **D.** The fractional level vs its own `--save-wq` files: 16 tensors ≤ 4.1e-4, the fp16 scale floor; the wrong
  codebook gives 8e-3.
  - The 3 down_proj that took ourq's **CPU refit fallback** (`OURQ_REFIT_CPU_FALLBACK`, 4070 only) differ by 1.7–3.1e-3.
    The gap is entirely per-output-row factors, up to 6%, that `--save-wq` has and the shard's svh does not. With
    them fitted out, the gap falls to the 2.1e-4 fp16 floor of the wq file itself.
  - The decode matches the oracle (what radiance serves) on those same tensors. The fallback path's `--save-wq` is
    what is off. See open questions.
- **E.** `save_lean` = `save_file`: tensors, dtypes and metadata.

**0.8B GPU gate (`asym27/test08.sh`, one hold of the 4070's gpu.lock, ~10 min).** Rules written before running:

- **G1 (bit-exact).** `mkdenoise` ρ 0 on `sib27test/ref2` is moments.py on q-k3base's `--save-wq` siblings,
  bit-identical to the w07 moments d1103's arm used. Then the 48 writers are requantized via `ourq` with the
  levelargs-derived args, as `asym27.sh` calls it. PASS if every tensor equals d1103's `q-k3asym_r0` (#87-05,
  K3 seed 0) bit for bit. If not, explain: the stamps tell whether code changed since 11-03.
- **G1m.** The patched sibmom27 (`--wq`, layers 0–3) = ref2 bit for bit (`cmpmom diff` 0).
- **G2 (the 27B path).** Level-decoded siblings (`sib27test/lv`) → mkdenoise ρ 0 → 48 writers → assembled level →
  search KL.
  - **PASS:** |log(KL / KL(k3asym_r0))| ≤ 0.025 **and** KL / KL(k3base) ≤ 0.85.
  - **Why 0.025:** the level path re-draws F like a writers-only reseed (SD 0.0083 at K3, w07), and two independent
    draws differ with SD ≈ 0.012, so 0.025 is ≈ 2 SD.
  - **Why 0.85:** d1103's ρ 0 arm sits at ≈ 0.81 of k3base.
- **G3 (fractional on the GPU).** sibmom27 `--level` on a K1.5 level (layers 0–3): H = G1m's bit for bit (the clean
  inputs do not depend on the siblings) and trN/trH above K3's in every group.

**Results (2026-11-05 10:00–10:06 PM CST, 335 s of GPU in one hold; `test08.out`): all PASS.**

- **G1:** 192 / 192 writer tensors and 48 / 48 wq files are bit-identical to d1103's ρ 0 arm, although the source
  stamp changed since 11-03 (`b5a6d925` → `59620c63`). The 27B driver's argument path, the merged lean mkdenoise at
  ρ 0 and ourq's `--hin` / `--wpre` reproduce #87-05 exactly.
- **G1m:** H, C, N max rel 0 vs ref2, peak GPU 1.3 GiB.
- **G2:** the 27B path (level-decoded siblings) gives KL 0.07961.
  - It is 0.9887 of d1103's arm (z −2.0, |log| 0.011) and 0.805 of k3base (d1103's own arm: 0.814).
  - **Not bit-exact, explained:** the level-decoded siblings differ from `--save-wq`'s by the fp16 rounding of
    suh / svh, which moves C / N by a few % (#87-06 B2) and re-draws F. #87-06 saw the same path at ρ 0.5 come out
    0.992.
  - The assembled level decodes to the scored weights (150 tensors ≤ 4.06e-4).
- **G3:** 8 / 8 groups have H bit-identical and trN/trH above K3's. The fractional decode runs inside sibmom27 on the
  GPU.
- **Also, on gpubox's CPU (11-05 ~9:55 PM CST):** `deq.py oracle` on the two real arms (`q-s1/K2.5` and
  `v3ladder2/K2.5`; layer 0 in_proj_z / out_proj, layer 3 k_proj / o_proj) is bit-exact in gpubox's venv.

## What only the card run can tell

- moments speed and peak VRAM / RSS on the R9700: the gate prints them;
- mkdenoise f64 time at 17408;
- the no-op gates;
- whether F at 1M tokens is still noise-dominated at n/k 57 (the dose);
- the effect size itself.

## Open questions for the lead

- **`OURQ_REFIT_CPU_FALLBACK`:** the CPU-fallback refit leaves `--save-wq` inconsistent with the shard for those
  tensors (test D). This affects the hconv 27B proxy numbers that read wq or rowerr for down_proj (#33-27 / #33-29
  family). It does not affect the swap arms (gpubox never takes the fallback) or the 0.8B (all 150 tensors
  ≤ 4.15e-4 on q-k3edy).
- **Third pair:** with s2 arms coming in q2.sh, a third pair is cheap to add. The rule above allows it only in the
  pre-registered INCONCLUSIVE case.

## Status (2026-11-07; the rule above is unchanged)
- #105 is still queued on card 0 (no 27B data yet).
- **w78 (11-07 3:30 PM CST):**
  - **Moments fault fixed:** it was the `expandable_segments` allocator setting, which `asym27.sh` exported (see
    `asym27/dbg/README.md`). Also fixed: the host-RAM peak at the moment save, and the "DONE" logged after a DRY run.
  - **Run:** both arms run from `/data/asym27/go.sh` → `/data/q1106/after-asym27.sh`. It logs each arm's exit
    status; status file `/data/asym27/queue.log`.
  - **mkdenoise:** its lean-path f64 `torch.linalg.solve` (k 17408) took > 20 min a group on ROCm. It now uses
    `asym27/mkdenoise.py` (+ `mkdenoise-w78.patch`): a blocked Cholesky solve, the same F to f32 rounding.
  - **Stage:** take `mkdenoise.py` from `asym27/`, not `proxy2b/exp/w07-denoise/`. Also stage
    `asym27/after-asym27.sh` (to `/data/q1106/`) and `go.sh`.
  - **Rules:** unchanged.
- **#111 (0.8B, `v3/e3/settle/w64.md` item 1, 11-07):** per-level asym writers placed next to siblings of *another* K,
  as in a knapsack mix, are 8–24% worse than plain writers (geo 1.077 / 1.229 / 1.239 over three cells); a per-mix
  writer pass keeps the gain (0.81–0.90). So the "Rollout" above (one run per shipped level) does not carry to mixed-K
  mixes. If #105 passes, the user chooses: a per-mix writer pass (~4–5 card-h a mix at 27B), the writer's K tied to its
  siblings' K in the knapsack, or plain writers where the siblings differ.
