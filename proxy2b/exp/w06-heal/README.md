# w06-heal: frozen-code healing gates on the 0.8B (ranked.md H1, H4)

**Question.** How much KL can be bought back by training only the continuous parameters of a fixed quant, with the
trellis codes frozen? Does the gain sit in the last layers? Is a bf16 backward good enough? And can healing be done
once per ladder level (a library), or must it be done per mix? These answers fix the order of the healing track:

- H2 = the S27 stack, about a week of engineering. It is the largest-EV item in ranked.md.
- Cheaper alternatives come first if healing is small: H3 (forward-only scale refit) and E3 (head healing).
- H5 (depth curriculum, fp32 early layers) depends on the depth and precision answers.
- H4 decides between a per-level library and a per-mix final pass.

## Mechanism

`quantize.py` stores W_q = diag(su) H Q H diag(sv) (in × out). A per-input multiplier r and a per-output multiplier c
on the dequantized weight are therefore exactly multipliers on the stored fp16 `suh` / `svh`: no format change, and
the codes are untouched.

**The student.** It is the bf16 0.8B with the existing `--save-wq` weights substituted (as `evalq.py` does). Each
trunk linear is wrapped as y = e^c ⊙ (W_q (e^r ⊙ x)), with the multipliers computed in fp32.

**Norm gains trained** (log multipliers on the effective gain 1 + w; found numerically, `norm_form`):
- `q_norm` and `k_norm` of the six full-attention layers;
- the final norm. The head is tied, so this is the only head-side knob.

**Norms not trained.** The input, post-attention and GDN gated norm gains lie in the span of the r of the linears
that read them, so training them would only add gauge directions.

**Gauge fixes** (both checked in the self-test):
- `down_proj` r ≡ `up_proj` c, because down reads silu(gate)·up;
- `v_proj` c ⊂ `o_proj` r tied over the GQA group, because o reads (softmax V)·gate.

**Parameter count.** 283 vectors, 554k values, all on the trunk's 150 quantized linears.

**Teacher and loss.**
- The teacher is the same checkpoint in bf16, run online with no_grad. Its logits are made the same way as the
  golden's (`head(h).float()`).
- The loss is the full-vocabulary forward KL(p_bf16 ‖ q), summed over the scored (`gen`) rows of each window. That is
  the challenge's per-token weighting.
- Context flows through every position.
- The 248k-wide logits are computed in 128-row chunks and recomputed in the backward pass (as in `collect.py`), with
  gradient checkpointing over layers.

**Optimizer.** Adam (0.9, 0.99) with lr 1e-3, 16 warm-up steps and cosine decay. Each step accumulates 4 windows.
After every step the parameters are clamped to |log| ≤ 0.3 (a log-scale trust region).

**Early stopping and output.**
- Every 32 steps the held-out KL is measured. The run stops after 4 evals without improvement and keeps the best
  held-out parameters.
- `fold.py` then writes W' = diag(e^c) W_q diag(e^r) as f16 `wq` files, plus a checkpoint copy with the new norm
  tensors. `evalq.py` scores that on `/data/g08-search1`.
- A run killed mid-way resumes from `state.pt` bit-exactly (self-test 6).

## Data (counts from `datacheck.txt`, run CPU-only on the 4070's files)

| role | source | size |
|---|---|---|
| train | qad-pool docs (pools-v3), windows of ≤ 2048 tokens from each doc's start (short docs whole, ≤ 8 a doc), fixed shuffle (seed 0), first 1,024 with a scored row | 912 docs, 1.11M tokens, 1.01M scored rows (91%) |
| early stop | search-pool docs **not in g08-search1**, first 2048 tokens, shuffle seed 1, first 48 | 39.8k tokens, 31.1k scored rows, overlap with the golden 0 |
| score | `/data/g08-search1` via `evalq.py` / `pcmp.py` | 171 docs with scored rows, 166,155 gen rows |
| oracle (H1a only) | the scoring golden's own documents, whole, trained *and* stopped on them | 192.6k tokens; the item rows sum exactly to the golden's gen rows |

The verify pool is never read.

Ledger entry needed: qad-pool use for 0.8B healing training, and the 48 search docs used as an early-stop slice. If
the lead builds new 0.8B search2 / verify slices (G1), they should exclude these 48 docs, or this slice should be
rebuilt from what is left.

## Arms

All arms keep the codes fixed (ranked.md §1), so one paired comparison per arm (`pcmp.py`) suffices; no seed
replicates are needed. Training-data order is fixed by the seed. Controls are the existing `/data/yb` rows: `k3base`,
`k2base`, `k3bdiag`.

| tag | idea | codes | trained | data / stop | steps |
|---|---|---|---|---|---|
| `k3jit` | control | k3base | none: random r, c ~ N(0, 0.003) folded | — | — |
| `gc-k3`, `gc-k2` | H1c | k3base / k2base | gradients only (64 windows), bf16 vs fp32 | — | — |
| `k3all`, `k2all` | H1a realistic, H1b | k3base / k2base | all layers + norms | qad / held-out search | ≤ 512 |
| `k3last`, `k2last` | H1b | k3base / k2base | layers 18–23 + final norm (backward stops at 18) | qad / held-out search | ≤ 512 |
| `k3orc`, `k2orc` | H1a ceiling | k3base / k2base | all, lr 3e-3, 6 epochs, no early stop | **g08-search1 itself** | ~256 |
| `k3dall` | stack (reported) | k3bdiag | all + norms | qad / held-out search | ≤ 512 |
| `mixF`, `mixD` | H4 control | family mix (MLP from K2, attention / GDN from K3, 2.47 bpw) / depth mix (even layers K2, odd K3) | none | — | — |
| `mix?x` | H4 transfer | the mix | `k2all` / `k3all` parameters per tensor; q / k norms from the layer's q_proj level; final norm = mean log | — | — |
| `mix?h` | H4 mix-specific | the mix | all + norms, same budget as `k?all` | qad / held-out search | ≤ 512 |
| `mix?w` | H4 polish | the mix | starts from `mix?x`, ¼ budget | qad / held-out search | 128 |

**Why the `k3jit` control.** `evalq.py` casts the f16 weights to bf16, so folding any multiplier re-draws the bf16
rounding of every weight. This arm measures that floor.

**What `gc` computes.** Summed gradients of all 554k parameters at the unhealed point, over 64 qad windows. The
student runs in bf16, then in fp32 (the same bf16 weight values cast per op; norms, GDN and attention in fp32). Per
layer it reports cos(bf16, fp32) and the even/odd split-half cosine of each.

## Draft pass rules (for the lead to commit before data)

ratio = KL(arm) / KL(control) on g08-search1; z from `pcmp.py`. `summ.py` applies these rules mechanically.

**Control (jitter).**
- `k3jit` vs `k3base` should give |ratio − 1| ≤ 0.005. Its z is reported but not used: with fixed codes the paired
  SE is tiny, so even a harmless 0.1% shift is "significant".
- Otherwise, arm differences up to |ratio − 1| of the jitter arm are re-rounding noise and are reported as unreadable.

**H1(a) headroom.** Ceiling = 1 − ratio(`k3orc`).
- Ceiling ≤ 5% (ratio ≥ 0.95): **defer S27**; do H3 / E3 first.
- Ceiling ≥ 15% (ratio ≤ 0.85): **build S27 now**.
- In between, the realistic arm decides: build if `k3all` ≤ 0.97 with z ≤ −3, else defer.
- Reported beside the rule:
  - the H2 pilot bar (ranked.md H2: "≥ 5%"), met if `k?all` ≤ 0.95 with z ≤ −3 at K3 or K2;
  - the share of the ceiling that generalises, gain(all) / gain(oracle).

**H1(b) depth.** Compare `k?last` against `k?all` (paired).
- **Reproduces** ("last quarter ≥ all layers"): diff ≤ 0 at both K3 and K2. Strong if, in addition, z ≤ −2 at either
  K. Then H5 fires: 27B healing starts on the last 16 layers.
- **All layers:** last is worse with z ≥ 3 at both K.
- Otherwise inconclusive, and the default is all layers.
- Separately, **backward truncation to the last quarter is allowed** if gain(last) / gain(all) ≥ 0.70 at both K. That
  cuts the 27B healing backward to about ¼.

**H1(c) gradient precision.** **PASS** (a bf16 backward is fine for healing) if every layer has cos(bf16, fp32) ≥ 0.95,
or ≥ that layer's fp32 split-half cosine (bf16 error below the 32-window sampling noise), at both K. Layers that fail
need an fp32 backward on the 27B (H5).

**Stack (reported, no rule).** gain(`k3dall` vs `k3bdiag`) / gain(`k3all` vs `k3base`): does healing add on top of
the diagonal H_out?

**H4 transfer**, per mix. Precondition: `mix?h` beats `mix?` with z ≤ −3; otherwise the mix is uninformative. Share =
gain(x) / gain(h).
- **LIBRARY:** share ≥ 0.70 and `mix?x` beats `mix?` with z ≤ −3, in **both** mixes. Healing becomes a per-level
  option: each ladder level is healed once and radmix carries its scales.
- **LIBRARY + POLISH:** otherwise, if gain(w) / gain(h) ≥ 0.90 in both mixes. Each level is healed once, plus a
  ¼-budget polish per shipped mix.
- **PER-MIX** otherwise: healing stays a final pass on each shipped mix.

## Decisions it changes

| result | decision |
|---|---|
| H1a | whether H2 (S27: torch exl3i op, golden teacher, metric-exact loss; ranked #17, largest EV) is built now, or H3 (#24) / E3 (#11) go first |
| H1b | H5 depth curriculum, and whether the 27B healing backward can be truncated at layer 48 (4× cheaper) |
| H1c | whether 27B healing can run with a bf16 backward (#33-03: the bf16 27B backward was wrong in early layers for H_out rows; this tests the healing gradient itself) |
| H4 | ranked.md H2's fit (FP vs PL): a library option enters knap4 at no per-mix cost, while per-mix healing costs ~3 h per size on the GPU box |
| stack | whether healing and the diagonal H_out (#33-16) compound |

## GPU time (4070, one GPU, `VRAM_CAP_GB=7.2`)

These are estimates. The `gate` stage prints the measured s/window and peak memory first.

**Assumptions.**
- About 0.4–0.8 s per training window (average 1,083 tokens; two bf16 0.8B models plus 128-row logit chunks).
- About 2 min per `evalq` run.

**Per-arm times.**
- A heal-all run (≤ 2,048 windows plus 16 held-out evals): about 15–30 min.
- `last`: about 60% of that.
- Oracle: about 15 min.
- gradcos: about 5 min per K.

**Totals.**
- H1: about 2–3.5 h.
- H4: about 1–1.5 h.
- **Total: about 3–5 h.**

**Peak memory.** Expected about 5 GB.

**Disk.** About 1 GB of folded f16 weights per healed arm, plus a 1.7 GB checkpoint copy when norms are trained. About
35 GB in all under `/data/yb/w06-heal`.

## Run

```
proxy2b/exp/w06-heal/run.sh                 # 4070; STAGES="gate h1 h4" (default); results /data/yb/w06-heal/result.txt
```

- The stages run in order: `gate` (identity check plus 8 timed windows; aborts on failure), `h1`, then `h4`. `h4`
  needs `h-k2all` / `h-k3all` from `h1`.
- `summ.py` prints the decisions at the end and writes `summary.json`.
- No shared code changes and no patches. `evalq.py` / `pcmp.py` are used as they are: the healed arm is materialised as
  a `--wq` dir plus a model dir (`fold.py`).

## Files

- `heal.py`: training, gate, gradcos.
- `fold.py`: healed arm → evalq inputs; `--jitter` control.
- `mkmix.py`: H4 mixes and transferred parameters.
- `summ.py`: the rules.
- `run.sh`: the driver.
- `datacheck.py` / `datacheck.txt`: data counts.
- `selftest.py` / `selftest.txt`: the CPU self-test.

## Self-test (`selftest.txt`, CPU on colo, real 0.8B checkpoint, synthetic quantized weights)

1. **Identity.** The wrapped student at zero equals the plain substituted bf16 model bit for bit.
2. **Fold.** The student with random multipliers (SD 0.05) equals the `fold.py` output loaded as evalq loads it (fp32).
   The difference is 0.6% of the multipliers' effect, which is the f16 storage of the product.
3. **Gauge.** Both equivalences hold to about 5e-5.
4. **Gradient.** fp64 autograd agrees with central differences.
5. **Data rows.** The window rows match `evalq`'s scored rows, and the pools and exclusions are right.
6. **Resume.** 6 steps run straight are identical to 2 steps, a kill, and 4 more after resume. The trust clamp holds.
7. **mkmix / jitter.** Levels, bits per weight, transferred tensors and the jitter SD are right.

Not testable on the CPU: fla / causal_conv1d kernels (colo uses the torch reference path), speed, and memory. The
`gate` stage covers those on the 4070.

## Caveats

- **Proxy scale.** The 0.8B healing gain is an upper-ish guide for the 27B. The base is already refit
  (`refit_scales`), so expect the low end of ranked.md's −3…−10%.
- **Format realism.** Real containers keep `suh` / `svh` in fp16 and the eval multiplies in the f16 kernel. Here the
  folded product is rounded to bf16 by evalq. The `k3jit` arm measures that floor.
- **Oracle bias.** The oracle trains and stops on the scoring slice by design. It is a ceiling only and informs no
  pick.
- **Single training seed.** H1b compares two trained arms with one training-data seed each. With fixed codes the noise
  is the optimisation's, which is smaller than rounding noise but not zero. A diff near 0 is reported as such.

## Runtime cost (user rule, 2026-11-03)
none if folded: per-row c and per-column r fold into svh / suh, norm gains into the checkpoint norms (the fake-quant `wq` files here are for evaluation; verify the fold in a real pack).
