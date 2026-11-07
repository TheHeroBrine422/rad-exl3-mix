# w46 · Re-validating ladder / format decisions made on the old trellis2 proxy, on w43's settle.py proxy

Worker w46, 2026-11-06 (CPU: colo + Thor; one short 4070 check). w43's `settle.py` (README here) follows exllamav3's
real steps: output Hadamard, per-input-channel scales with `--scaled-hessian`, its own g_scale search and drift, the
per-output-channel scale refit. Its #39 rerun found the old `trellis2.py` scale handling produced an artifact. w45
confirmed it against the GPU smoke (`v3/e3/sub1/r94b.md`). Here every decision that sets the ladder or the format and
rests on the old proxy is re-run on settle.py **with its original rule text unchanged**. Each one is reported as
**holds** or **flips**; a flip is a finding, and the lead decides what changes.
**Runtime cost:** none of the runs changes anything by itself (proxy measurements). What the decisions price: #90-02 →
which ladder levels get an i4 twin (A4 path, prefill); #100 → decode / prefill codebook ALU; #106 → none (a folded
permutation); #90 → knapsack pricing only.

## Method (common to all items)

- `reval/reval.py` = settle.py unchanged, plus codebooks `iqrh` / `iqrh:A:B` (trellis4.py's, the ones #100 used),
  `int4x` / `mul1x_i8` (the fork's exact EXL3_MUL1_I4 / EXL3_MUL1_I8 values, as a check), and a mode `order` for #106.
  `int4grid`, `mul1_i8`, `mul1`, `cubic0.02_i8` are trellis.py / trellis2.py's, as in the original runs.
- **Primary statistic: rel after the refit at m = 1**, i.e. each codebook at the scale exllamav3's own g_scale
  search picks for it, as the quantizer does. The old runs instead took the best of a factor grid on per-row RMS
  scales; that scale path is what settle.py replaced. Robustness, reported beside it: the best of
  m ∈ {0.9, 1, 1.1} per job.
- Samples: seed s = sign seeds s + row block s (128 rows = one output-Hadamard block), w43's convention; ≥ 2
  samples everywhere (0, 1; a third, 2, for #90-02 / #100 at the lowest priority). Per tensor, the samples' ratios
  are pooled by their geometric mean. Weights = whole-tensor parameters (ffn_gate_up = gate + up), as the original
  reports. settle.py quantizes gate_proj on even seeds and up_proj on odd ones.
- Damping: adopted ladder values (input group 0.005, rest 0.025). The old runs used 0.01 everywhere.
- Sub-1 K runs settle.py's trellis with 0-bit steps (salted read) as in trellis.py; the drift is 1.0 below K1
  (exllamav3's lookup has no entry there).
- Checks done before launch: the `order` mode's natural arm reproduces w43's settle.py rel for 19:attn_output K2
  seed 0 exactly (0.02772837462809245, same s_exl); codebook value sets (`reval.py cbcheck`).

## Rules (quoted unchanged from the originals)

1. **#90-02** (`sub1/README.md`, 2026-11-04): "below the highest K where i4 / i8 ≤ 1.01 on the parameter-weighted
   mean (and no tensor > 1.03), the i4 ladder replaces the i8 one (one quant serves f16, A8 and A4); between that K and
   K3 both are built …; above K3 i8 only." K0.125 … 3.5 as `i4run.sh`. #90-03: "if a verdict flips [when the data are
   pooled], the pooled one is used and the flip is logged" → the verdict uses the 20 tensors pooled, 2 samples. The
   old result: passing run up to K1.5 (the user then set i4-only at K ≤ 1, both at K1.125–3).
   *Addendum, not part of the rule:* the same statistic at the ladder's 1/8 steps K1.125–1.875 to locate the
   boundary.
2. **#100** (`sub1/README.md` + amendment): ratio = rel(hash) / rel(mul1_i8), parameter-weighted per K. "≤ 1.005
   everywhere in K1–6 [and no tensor > 1.02] → adopt; otherwise no automatic reject below ×1.05: the decision waits
   for the measured speed …; above ×1.05 at any K in 2–5: rejected." Same 20 tensors and K (0.25, 0.5, 1 … 3.5, 4,
   5, 6, 7). Old result: no automatic call (+1.2–1.4% at K2–3.5, +2.7% K4, +3.5% K5, +5.4% K6).
   Side tables (no rule): iqrh / mul1 (the f16-path codebook), mul1_i8 / mul1, the tuned global A = 0xd0474b.
3. **#90 sub-1 curves** (pricing only): rel(K) / rel(K1) per family at K0.125–0.875 on #90's 49 tensors, with #90's
   rule 1 ("rel(K) falls monotonically at every 1/8 step for every tensor, flat within 1%"). The ladder's sub-1 codebook
   is the int4 grid (`int4grid`, 2 samples). The old run's codebook was i8c (`cubic0.02_i8`, sample 0), so the
   proxy change is read on the same codebook. Output in `sub1ratios.json`'s format (`data/e3/settle/reval/`).
4. **#106** (new, `actblk/README.md` "#106", rule written before data): head-permutation act order on o_proj /
   GDN out_proj; adopt per family if geo-mean head / natural ≤ 0.98 at ≥ 2 of K2/3/4 and no rule tensor > 1.00.
5. **i8 vs mul1 on real tensors (4070)**: `ourq.py --codebook i8` vs mul1 at K6.5 / 7 / 8 on two 0.8B layers
   (6 = GDN, 7 = attention), rel from `--save-wq`'s rowerr (tr(dW H dW^T) / tr(W H W^T), undamped calib H_in).
   No rule (a check of w45's ×1.018 K6.5 / ×1.27 K7 / ×4.5 K8).

## Results

Queues ran 10:07 AM – 6:05 PM CST on colo + Thor (4,703 jobs, 0 failures, ≈ 15.6 box-hours); item 5 on the 4070
(GPU lock held 2.6 min in total). Full tables: `data/e3/settle/reval/report.md` (`revalreport.py`).

| item | old verdict (old proxy) | settle.py | holds / flips |
|---|---|---|---|
| 1 · #90-02 i4 vs i8 | passing run to K1.5 (K2: 1.0107) | passing run to **K2** (K2: 1.0088) | **flips** K1.5 → K2 (no plan change forced) |
| 2 · #100 iqrh vs mul1_i8 | no automatic call (+1.2–1.4% K2–3.5, +2.7–5.4% K4–6) | no automatic call (same within ±0.3%) | **holds** |
| 3 · #90 sub-1 curves | monotone; knap4 rule 2 below 3% (−0.07% at 1.5 bpw) | monotone; K ≥ 0.375 within ±4%, K0.125–0.25 lower; knap4 −0.06% at 1.5 bpw | **holds** (old overstated only K0.125–0.25, which knap4 never picks) |
| 4 · #106 (new) head-order act order | – | o_proj **0.927** (pass); out_proj 0.986 (fail) | o_proj **adopt** by its rule |
| 5 · i8 vs mul1, real 0.8B quants | (w45 proxy: ×1.018 / ×1.27 / ×4.5) | ×1.014 / **×1.272** / ×4.48 at K6.5 / 7 / 8 | w45 **holds** |

### 1 · #90-02 i4 vs i8: **the rule's boundary moves from K1.5 to K2** (complete: 20 tensors × 2 samples)

Ratio int4grid / mul1_i8 at m = 1, parameter-weighted mean over the 20 tensors (samples pooled per tensor):

| K | 0.125 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | **2** | 2.25 | 2.5 | 2.75 | 3 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| settle.py, pooled (verdict) | 0.999 | 1.000 | 0.999 | 1.000 | 1.001 | 1.002 | **1.0088** | 1.0130 | 1.018 | 1.029 | 1.062 | 1.439 |
| max tensor | 1.010 | 1.021 | 1.023 | 1.010 | 1.012 | 1.007 | 1.016 | 1.015 | 1.035 | 1.036 | 1.069 | 1.488 |
| old trellis2 (#90-02, 10 tensors) | 0.985 | 0.999 | 0.996 | 1.001 | 1.002 | 1.003 | 1.0107 | 1.015 | 1.018 | 1.026 | 1.067 | 1.445 |

- **Rule output: passing run K0.125 → K2** (old: → K1.5). By the rule's text: "the i4 ladder replaces the i8
  one" up to K2, both from K2 to K3, i8 only above K3. The flip is K2 alone: 1.0088 against the 1.01 bar, where
  the old proxy gave 1.0107. Every other K is within ±0.5% of the old value.
- It is not one sample's luck. Every view gives K2: #90-02's own 10 tensors on sample 0 (the original design,
  1.0087), samples 0 + 1 (1.0085), #90-03's 10 new tensors (1.0090), three samples (1.0083), best
  of m ∈ {0.9, 1, 1.1} per job (1.0088), and the exact fork codebooks int4x / mul1x_i8 (1.0090; 8 of #90-02's 10 tensors, ffn_down not run).
  The margin is small, though: 0.12% under the bar, and the worst tensor (L46 ssm_z) is at 1.016.
- Addendum (the ladder's 1/8 steps; 16 tensors, ffn_down not run): 1.0009 K1.125, 1.0016 K1.25, 1.0010 K1.375, 1.0037
  K1.625, 1.0046 K1.75, 1.0070 K1.875. i4 costs ≤ 0.5% up to K1.75 and < 1% up to K2.
- **For the lead:** the ladder plan (user 11-05: i4 alone only at K ≤ 1, both i4 and i8 at K1.125–3) is more
  conservative than both the old and the new rule output, so nothing is forced. If the user wants to save the
  K1.125–2 i8 twins (8 levels at 1/8 steps), the rule now allows it: the i4 cost there is 0.1–0.9% proxy error
  (equal-quality ≈ 0.0006–0.006 bpw at the mul1_i8 slope of ~2 log2/bit).

### 2 · #100 iqrh vs mul1_i8: **holds, no automatic call** (complete: 20 tensors × 2 samples)

Parameter-weighted iqrh / mul1_i8 at m = 1 (samples pooled), next to the old run (trellis2/trellis4, sample 0, best
of a 0.5–1.5 factor grid):

| K | 0.25 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| settle.py (verdict) | 1.013 | 1.007 | 0.996 | 0.998 | 1.013 | 1.012 | 1.013 | 1.012 | **1.029** | **1.035** | **1.052** | 1.067 |
| max tensor | 1.038 | 1.023 | 1.027 | 1.000 | 1.034 | 1.025 | 1.019 | 1.018 | 1.048 | 1.039 | 1.057 | 1.074 |
| equal-quality bpw | +0.006 | +0.004 | −0.003 | −0.001 | +0.009 | +0.009 | +0.009 | +0.009 | +0.021 | +0.026 | +0.042 | +0.061 |
| old (r100.md) | 1.011 | 1.007 | 0.997 | 0.999 | 1.014 | 1.013 | 1.012 | 1.012 | 1.027 | 1.035 | 1.054 | 1.065 |

- Rule: not ≤ 1.005 in K1–6 (so not adopt); worst K in 2–5 is K5 at 1.035 < 1.05 (so not reject) → **no automatic
  call; the user decides from this table + the card speed bench**, exactly as before. Every K agrees with the old
  run within ±0.3%; sample 0 alone, three samples and best-of-m give the same table (±0.1%). The proxy change does
  not touch this decision. SE over tensors ±0.05–0.2%.
- Side tables (no rule; 16 tensors, ffn_down not run):
  - tuned A = 0xd0474b vs mul1_i8: 1.010 K2, 1.008 K3, 1.010 K4, 1.020 K5, 1.015 K6. That is better than IQR's A
    at K4–6 (old r100.md: 1.007 / 1.021 / 1.020 at K4 / 5 / 6, so it agrees).
  - iqrh vs mul1 (V2's f16 codebook): the same as vs mul1_i8 up to K6 (1.013–1.055), **1.37 at K7**.
  - mul1_i8 vs mul1: ≤ 1.004 up to K6, **1.28 at K7** (w45's #94 rerun and item 5 again, now on 16 more tensors).

### 3 · #90 sub-1 curves: **the shape holds from K0.375; the old proxy overstated K0.125–0.25** (complete)

rel(K) / rel(K1) per family, geometric mean over #90's 49 tensors (int4grid = the ladder's sub-1 codebook, samples
0 + 1; ffn_down sample 0 only), and the same codebook as the old run (cubic0.02_i8 = i8c, sample 0, ffn_down not
run) for a pure proxy comparison:

| family | K0.125 new (i4) | old (i8c) | i8c on settle.py | K0.25 new / old | K0.5 new / old | K0.75 new / old |
|---|---|---|---|---|---|---|
| attn_qg | 7.80 | 13.56 | 6.09 | 5.23 / 7.40 | 2.65 / 3.02 | 1.56 / 1.60 |
| attn_k | 7.27 | 9.15 | 6.95 | 5.09 / 5.65 | 2.72 / 2.73 | 1.61 / 1.56 |
| attn_v | 5.64 | 7.94 | 5.79 | 4.24 / 5.18 | 2.52 / 2.69 | 1.57 / 1.55 |
| attn_output | 5.33 | 5.27 | 5.30 | 3.96 / 3.83 | 2.38 / 2.30 | 1.52 / 1.49 |
| ssm_qkv | 5.99 | 7.12 | 5.95 | 4.30 / 4.62 | 2.48 / 2.47 | 1.54 / 1.52 |
| ssm_z | 7.91 | 9.79 | 7.88 | 5.40 / 5.58 | 2.85 / 2.75 | 1.63 / 1.56 |
| ssm_out | 4.59 | 4.81 | 4.66 | 3.54 / 3.61 | 2.23 / 2.23 | 1.48 / 1.47 |
| ffn_gate_up | 4.74 | 5.24 | 5.05 | 3.56 / 3.73 | 2.22 / 2.24 | 1.47 / 1.47 |
| ffn_down | 4.43 | 4.75 | – | 3.43 / 3.56 | 2.19 / 2.22 | 1.46 / 1.47 |

- #90's rule 1 (rel falls at every 1/8 step, flat within 1%) **holds** for every tensor and sample, on both
  codebooks.
- From K0.375 up the curves agree with the old ones within a few percent (K0.75: ±4%). At K0.125–0.25 the old
  proxy put the error higher, by ×1.19–1.74 at K0.125 on attn_qg / attn_k / attn_v / ssm_z / ssm_qkv. The same codebook
  on the new proxy (column 3) shows this is the proxy (the old factor grid stopped at 0.5, which #90's README
  flagged), not the i4 codebook. So sub-1 options were priced too expensive at the lowest K.
- Pricing input for knap4 in `sub1ratios.json`'s format: `data/e3/settle/reval/sub1ratios-settle.json`.
- **Re-priced in knap4** (`reval/r90knapx.py`: r90knap.py's rule 2, menu v3e, 256 KiB buckets, 24 solves on colo,
  ~10 min; the base and the old-ratio arm reproduce `data/e3/sub1/r90knap.md` exactly). Predicted KLD change from
  sub-1 options, new vs old curves: **−1.56% vs −1.77% at 1.25 bpw**, −0.54 vs −0.69% at 1.3, −0.06 vs −0.10% at
  1.4, −0.06 vs −0.07% at 1.5, ≈ 0 from 1.75 bpw on. The knapsack only picks K0.75 / 0.875 (never below), where the
  new curves are 0–4% *higher*, so the lower K0.125–0.25 prices do not matter. #90's rule 2 (≥ 3% at 1.5–2.5 bpw)
  stays **below 3% at every size: holds**. Output: `data/e3/settle/reval/r90knap.md` / `.jsonl` (column "sub1" =
  settle.py curves, "sub1max" = the old curves).

### 4 · #106 o_proj / out_proj head-order act order: **o_proj passes (×0.927), out_proj fails** (complete)

Rule, setup and the full table: `v3/e3/actblk/README.md` "#106". Rule tensors × 2 samples, rel(head) / rel(natural):
o_proj 0.928 / 0.927 / 0.927 at K2 / 3 / 4, no tensor > 1.00 → **adopt** (by the rule; 27B confirmation next).
out_proj 0.986 / 0.986 / 0.988 → not adopted. Extension (sample 0, all 16 o_proj layers): 0.942–0.944 at K1.5 / 2 / 3
/ 4 / 5 / 6 (flat in K), every layer < 1 (0.902 L43 … 0.979 L3). 13 out_proj layers: 0.977–0.978, one > 1.00 (L30
K4). The head-constrained order gets nearly the free order's gain; the reversed order costs +5–7% (sign check).

### 5 · i8 vs mul1 on real quants (4070, 0.8B layers 6 + 7): **w45's proxy numbers hold**

`ourq.py --codebook i8` vs mul1, ladder settings (qtf2, scaled Hessian, σ 0.025 / input group 0.005), 13 linears
(GDN layer 6 + attention layer 7), rel recomputed from the saved wq against the bf16 weights under the undamped
calib H_in (`reval/i8rel.py`):

| K | median i8 / mul1 | geo mean | min / max | w45 settle.py (27B, median) | GPU smoke (Gaussian tiles) |
|---|---|---|---|---|---|
| 6.5 | **1.014** | 1.015 | 1.008 / 1.022 | 1.018 | – |
| 7 | **1.272** | 1.291 | 1.158 / 1.435 | 1.271 | 1.024 |
| 8 | **4.48** | 4.40 | 3.50 / 5.10 | 4.478 | 1.347 |

The real quantizer agrees with the CPU proxy to within 0.5% and not with the smoke: **i8 K7 costs ×1.27 (≈ 0.18
bit)**; q_proj and GDN in_proj_z / in_proj_qkv are the worst (×1.38–1.44 at K7), as attn_qg was on the proxy. Each
arm took 10–15 s; the GPU lock was held 96 s in total (16:14–16:16 UTC). Raw data: `data/e3/settle/reval/i8real/`
(`i8rel.out` the table, `rel.json`, ourq logs and proxy.jsonl per arm).

**Side finding: ourq's `--save-wq` row errors are wrong.** `wq/<key>.rowerr.npz` came out err ≈ ref (rel ≈ 1.000
for every linear) because `quantize_exl3` regularizes `w` in place (the fork's own comment: "regularize works in
place"), and ourq computes the row errors against that transformed `w`. The wq files themselves are fine.
`reval/ourq-rowerr.patch` keeps a CPU copy of the original weight when `--save-wq` is on. Tested on the 4070
(`reval/rowerrtest.sh`: layer 7, K7 mul1). The patched rowerr matches the independent recomputation within 0.2%
(f32 vs saved f16 wq), and wq is bit-identical. **Consumer to re-check: `v3/yaqa/headslab.py`** (#34 slab head:
reads `lm_head.rowerr.npz` for per-row costs).

## Files and how to run

- `reval/reval.py` (the proxy wrapper), `reval/revalplan.py OUTDIR` (job lists `jobs-cpu1.txt` colo / `jobs-cpu2.txt`
  Thor, priority blocks, tensors pinned to a box for the prep cache), `reval/revalq.py` (queue runner: w45's r94bq.py
  with mode / codebook / multipliers as job fields; re-reads the job file when it changes, so appending lines queues
  more work), `reval/revalreport.py DATADIR [--sub1json OUT]` (all tables and verdicts above),
  `reval/r90knapx.py run WD [PAR] | report WD` (item 3's knap4 repricing; a wrapper around `sub1/r90knap.py`; it needs
  a repo copy on the box: v3/nf, v3/e1b, v3/e3/data, v3/bench/data, data/v3, as `/data/work/w46/knap/repo` on colo).
- Item 5: `reval/i8real.sh OUTDIR [LAYERS]` (4070, under `/data/yb/gpu.lock`; 6 ourq runs of 10–15 s),
  `reval/i8rel.py QDIR CKPT HESSDIR OUT.json` (rel from the saved wq, CPU), `reval/rowerrtest.sh` (the
  `ourq-rowerr.patch` check), `reval/ourq-rowerr.patch` (for the lead: fixes `--save-wq`'s rowerr.npz, see item 5).
- Boxes: `/data/work/w46/` on colo (cpu1) and Thor (cpu2): code flat there, `libviterbi.so` built with
  `-march=native`, outputs `out/<mode>/<cb>/s<seed>/<layer>-<tensor>-K<k>.jsonl`, `logs/queue.log` ("REVAL QUEUE
  DONE" at the end), `touch logs/STOP` pauses. Launch from that dir:
  `python3 /data/work/w46/revalq.py jobs-cpu1.txt --max 30 --maxdown 7 --anoncap 120` (colo) /
  `... jobs-cpu2.txt --max 40 --maxdown 7 --anoncap 118` (Thor), with setsid nohup. Box-side `cache/` (the
  transformed rows per tensor / sample, 1.2–1.3 GB per box) can be deleted; `out/` is copied to the repo.
- Run log: started 10:07 AM CST (16:07 UTC) with 5,242 jobs (`jobs-*.v1.txt` on the boxes). Revised 12:19 PM CST:
  ffn_down jobs (~12 GB, 12–18 min each) can run only 5–8 at a time per box, so their ~945 pending jobs would have run
  ~20 h past everything else. The fix: ffn_down dropped from the optional blocks (third sample, #100 side arms,
  1/8-step addendum, old-codebook sub-1 curves, exact-codebook check); #90's sub-1 ffn_down on sample 0 only;
  39:ffn_down's #100 jobs moved to colo; maxdown 7 / 8 (colo cap 110 → 120 GiB, within AGENTS' 128 GB usable;
  Thor 112 → 118 GiB, w43's). No rule's tensors or K changed.
  1:25 PM CST, runner fix: small jobs had taken every GiB that came free, so ffn_down was starved. Now an ffn_down
  blocked by memory alone holds back small starts until it fits, and small jobs count at 1.2× their measured RSS.
  Thor maxdown 7.
  5:10 PM CST: colo's ffn_down work was done, so Thor's 22 pending #90 sub-1 ffn_down jobs (layers 3, 43) moved
  there. Colo also got block `i4k`: o_proj's head order at K1.5 / 5 / 6, reported only (57 jobs).
  `revalplan.py OUT jobs-v1-dir` reproduces the boxes' final job sets (4,703 jobs; checked).
- Watchdog: `local/watch.conf` rules `reval1` / `reval2` (proc + fresh + grepinc).
- Data: `data/e3/settle/reval/{cpu1,cpu2}/` (out/, queue.log, job lists), `report.md` (revalreport output),
  `sub1ratios-settle.json` / `sub1ratios-old.json`, `r90knap.md` / `.jsonl`, `i8real/`.
- Report: copy `out/` from both boxes into `data/e3/settle/reval/<box>/`, then
  `python3 v3/e3/settle/reval/revalreport.py data/e3/settle/reval --sub1json data/e3/settle/reval/sub1ratios-settle.json`.
