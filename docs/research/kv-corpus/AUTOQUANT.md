# AutoQuant (rad-autoquant, #416) and the KV autoresearch loop

Sources:
- Code 000's #416 worktree `/workspace/forks/radiance-kv416` @ `30d97f9`: `core/kv_autoquant.{h,cpp}`,
  `tools/rad_autoquant.cpp`, `docs/TOOLS.md` §rad-autoquant, `tests/kv_autoquant_test.cpp`, and commit `d13facf`
  (k5_v4), the template for adding a layout.
- The paper: IKV §5 ([infinite-kv-cache.md](infinite-kv-cache.md)).
- w40's judge: `/workspace/autoresearch-v3` branch `kv-task` @ `1dbf8c6` (README "The KV task", HANDOFF 3.6.0).

Written 2026-11-06; KV is parked, so nothing here was run.

## 1. What AutoQuant does

For each attention layer, AutoQuant picks one variant (a layout plus the words `rotated` / `vmean` / `vpred`).
The target is the fewest bits within limits on mean KL (`--max-kl`, or `--max-dkl` = floor + D), ΔNLL and top-1.

Every candidate is a child `radiance` run in KL mode against a bf16-cache reference (`--kld-ref`). The run order:
1. Uniform runs, one per variant. The base is the variant with the least KL (bf16).
2. Screening drops any variant that a variant with no more bits beats.
3. A noise run: the base again under `--noise-flags`.
4. Per-layer costs: L·(K−1) runs, one layer at a time in each kept variant.
5. The knapsack (an exact DP over bits) proposes a recipe, and **every proposal is measured**. If it fails, the
   room shrinks by the measured excess, at most `--verify` (6) times. Costs do not add: one early layer costs as
   much as all of them.
6. A second pass, from the best measured recipe: each layer moves one step down the lower hull of the uniform runs.
7. Global knobs.

The recipe file records every measurement as comments. `--bits` gives an unmeasured prediction; do not use it in a
loop.

## 2. Adding a new format

**A new layout (storage format)** needs no change to the tool. `kv_autoquant_variants()` asks the recipe parser
which word combinations exist, so a row in the layout table becomes a candidate on its own. What the k5_v4 commit
(`d13facf`, 26 files) had to touch:
1. `abi/rad_abi.h`: `RAD_KVQ_<NAME>` plus its `kv_quant_ok` probe bit. **This is an ABI bump**: every plugin
   rebuilds (~25 min at `-j2`, per `kvproblems.py`).
2. `core/config.cpp`:
   - a `kKvLayouts` row `{name, kvq, group_bits}`. `group_bits` = bits of one group of 32 K + 32 V values,
     scales included. **This is the price the knapsack solves with**; table order is the tie-break, widest first.
   - the parser's refusals: which of `rotated` / `vmean` / `vpred` the layout takes.
3. libref store + attention: the contract the byte-exact tests compare against.
4. libavx.
5. libr4d: a store row, plus prefill and decode attention rows **at the target geometry**, wired through
   `r4d_rows` / `r4d_shapes` / `r4d_plugin.h`, with selftest cases.
6. The arch plugin: the probe bit, one KV group per distinct layout, the fused-op exclusions, and rotation.
7. Tests: hand-worked store bytes, emulator, selftest, recipe parse and price, arch declaration.

**A new transform word** (in the way of `vmean` / `vpred`) needs:
- a parser word and `RAD_KVQ_*` flag bit;
- a calibration hook (`arch/common/rad_calib_*.h`, `--calib-dir`);
- **tool changes**:
  - `kv_autoquant_variants` hard-codes the V words `{"", vmean, vpred}` × {plain, rotated};
  - `rad-autoquant` knows only the calibration files `--vmean` / `--vpred`, and only their hashes are part of the
    reuse key.

**A new global knob** is a row in `kKnobs` (`tools/rad_autoquant.cpp`).

**Pitfalls:**
- *Stale reuse.* The `--reuse` key is the recipe text plus the argv, plus the vmean/vpred content hashes. **The
  engine binary is not hashed** (IKV §5.7), so a rebuilt engine silently reuses old runs unless the work dir or the
  `--radiance` path changes.
- *Screening.* Screening uses uniform runs only. A format that helps only the late layers can be dropped; use
  `--keep-dominated`, or name it in `--variants`.
- *`--min-context`.* The default is 32 768. kvlong-v1's 32k documents are cut at 32 768 and may re-tokenize a few
  tokens short, so set the bound explicitly.
- *Knobs on the 27B.* The `bkey_*` knobs are QSA-only. Use `--no-knobs` until our plugin serves `--state-cache-dtype
  f16`.

**On the 27B AutoQuant cannot run today.** Every run is a `--kv-recipe` run, and our `qwen35_exl3` has no
`kv_recipe_ok`. That plugin also lacks per-layout groups, rotation, the vmean/vpred hooks and a bf16 recipe layout.
At 16-token pages and gqa 6, libr4d has rows only for bf16, fp8, fp8_v4 and k6_v4 (model profile `baseline_gaps`).
Without bf16 as a recipe layout, the base becomes the best quantised variant, and no limit under its KL can be
reached.

## 3. Plugging the loop into the kv-task judge

The loop: an agent proposes a format → AutoQuant → frontier check → repeat.

**Today's kv-task.** A candidate is a fixed setup (cache dtype + recipe + optional `src.patch`). The tiers:
- T0: byte-exact tests.
- T1: KL with tails, plus speed.
- Gates.
- Pareto against the anchors.
- T2 RULER, opt-in.

Under this design a format's value is "how much it moves the frontier once every layer can choose it", so the agent
should not hand-write recipes. Proposal: a second candidate kind, **format**, in which the recipe is AutoQuant's
output.

1. **setup.json** gets `"search": {"variants": ["NEW_LAYOUT rotated", ...]}` instead of `recipe`. The agent names
   the variants it means; the judge adds the base frontier's hull variants (`--variants`).
2. **T0 unchanged**, plus two checks:
   - a **price check**: the table's `group_bits` against the engine's pool report (KiB/token = 4 × bpw on the
     27B);
   - **ABI-free slots**: reserve experimental `RAD_KVQ_*` values in a baseline patch, so a candidate is not an
     ABI bump.
3. **TA, a new AutoQuant tier**, on the bench host in `kvjudge.py`:
   - Use the **subject build's** `rad-autoquant`, because the trusted one links the baseline layout table and does
     not know the new row. To keep that safe:
     - take `core/kv_autoquant.*` out of `EDITABLE` for this problem (`tools/` already is);
     - treat its answers as **proposals only**. T1 re-measures every answer with the judge's own `kvstats.py` on
       the engine's `.rows`; `core/kld.cpp` is not editable.
   - Set `--radiance` to a judge wrapper (`execv`'d per run). The wrapper takes the card then ram locks, applies
     `engine_env` and the 0.2 s memory watch, and passes the engine log through, since AutoQuant reads bytes/token
     from it. Each child run is one hold of ≤ 540 s. A failed run stops the tool; rerunning with `--reuse` resumes.
   - Inputs:
     - `--kld-ref` = the judge's `refs/<key>/ref`;
     - after `--`: the profile's serve flags + `kl_extra`;
     - `--noise-flags` from `noise_chunk`;
     - `--strict`;
     - limits fixed per problem (below).
   - **Base frontier.** Once per base id, run AutoQuant on the baseline tree (cached like the anchors). Its uniform,
     per-layer and noise runs seed each candidate's work dir. A **reuse guard** must pass first: the subject build
     reruns two base uniform recipes (fp8 and the narrowest), whose `.rows` must match byte for byte. If they
     differ, run the full search. This guard is needed because of the unhashed binary. Byte determinism holds on
     Flash-Next; on the 27B it is unverified, which the first A/A must check.
4. **T1 on each answer recipe**: tails, bins, speed, gates, as today. AutoQuant prices bits only, and the dense
   27B reads its whole KV every decode step, so the speed runs are what catch a slow kernel.
5. **Frontier check** (verdict) against the base frontier:
   - At each limit: the bytes/token of the candidate's answer against the base answer, with T1's Pareto over
     (bytes, decode, prefill, mean KL, p99.9).
   - Ledger score: bits saved, summed over the limits (0 if none), plus being on the front.
   - **Rolling base (lead's call):** a promoted winner's patch joins the baseline, so later formats must beat the
     improved frontier.

**Limits on the 27B.**
- The floor is likely tiny: the model is dense with no routing, and IKV measured a dense bf16 2B at 0.00018, so
  `--max-dkl` ≈ `--max-kl`.
- Anchor the limits on fp8's measured KL instead: `--max-kl` = KL(fp8) ("fp8-equivalent") and 2·KL(fp8)
  ("compact"), each with `--max-dnll`.

**Against KL/NLL overfitting.**
- **Two corpora.** AutoQuant selects on a new, disjoint `kvsearch-v1` (`mkcorpus.py`, sha-pinned, documents ≥
  32k). T1 and its gates stay on `kvlong-v1`. This is the project's rule: never judge on the data that tuned.
- **What agents see.** Full AutoQuant logs (the recipe comments) from the search corpus. From kvlong, only the
  verdict class (front / dominated / rejected + stage), so the loop cannot iterate against it.
- **Refresh at each promotion:** kvlong-v2.
- **Per-document signs.** `--strict` limits, plus agreement of signs per document. With 3–4 documents a t-interval
  is 3–6× the block bootstrap (IKV §6).
- **RULER (T2) on finals only.** Run it on the top few on the front, at the end of a round or before a promote, on
  the lead's `--t2`. Use it as a hard gate (Wilson bounds against fp8 / bf16), never as a score fed back to agents.
  Teacher-forced KL on a few documents misses retrieval: KIVI gives RULER-32K 39.8 vs 56.4; plain INT4 gives 0.00
  ([kivi.md](kivi.md), [saw-int4.md](saw-int4.md)).
- **No drafter in the V2 container.** For a target that has one, add drafter acceptance as a limit (IKV §7.9 future
  work; a tool change).

**Cost per format candidate.**
- Runs = 2 (guard) + n_v (uniform) + 16·k_v (per layer) + Σ over limits (verify ≤ 6 + pass-2 steps ≤ 16 + pass-2
  verify ≤ 6), plus T1 per answer.
- Typical: 1–2 variants, 2 limits ≈ 60 runs × t_run. Each child run reloads the model.
- t_run on the 27B is **unmeasured**: w40's T1 was blocked by host RAM. IKV's figure is ~70 s per run on
  Flash-Next.

**Order before any agent run.**
1. Port #416's recipe path into `qwen35_exl3` as a harness baseline patch, not agent work: `kv_recipe_ok`,
   per-layout groups, rotation, bf16 as a recipe layout, the vmean/vpred hooks.
2. Attention rows at gqa 6 for i8h / k5_v4 / k4h_*: 16-token pages, or reach them through the 4-token-page rows.
3. One measured T1 A/A: ref, floor, fp8, t_run, determinism.
4. The `kvsearch-v1` corpus.
5. Optional baseline patches: experimental `RAD_KVQ` slots, table-driven transform words, and a generic
   `--calib WORD=FILE`. With these, new transforms need no tool edit.
