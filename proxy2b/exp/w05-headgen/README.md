# w05-headgen — LM-head statistics from scored rows only (ranked.md E4)

**Question.** Logits at unscored positions do not enter the metric. Should the head's input Hessian come only from gen rows,
instead of every position of the calib corpus as published? Two other uses were proposed: the slab Fisher ranking and the
headsim evaluation set.

**What is already true.**
- The slab ranking already uses gen rows (`headslab.py` ranks by `fisher_qad_gen`). Step 0 only reports its overlap with
  `_all`.
- The E1b head dump (calib-A, used by headsim) samples prefill steps with no positions, so it cannot be gen-masked. `headx.py`
  produces a gen-row dump with golden row and document ids instead.

**Prior: small.** In the qad pool **80%** of positions are scored (gen.u8). Plain-text sources are 100% gen; only chat and
agent traces are partial (wildchat 42%, agent-trace 46–50%, swe-rebench 26%). A gen-only Hessian can differ only through those
prompt segments. Expected head-KL change: a few % at most, ≤ 0.2% total.

## Steps (`run.sh`, iceman)

0. CPU: overlap of the gen vs all Fisher rankings (seconds).
1. **`src/headx.py`, both cards, ~30–60 min:** a bf16 forward of qad documents (2M context tokens, 8k cap), using collect.py's
   streamed loader. It produces:
   - `hin-gen` and `hin-all`: the head input Hessians of the **same** documents, gen rows vs every position, in challenge
     layout for `ourq --hin`;
   - a 65,536-row gen-row dump of **fit-pool** documents (held out) with document ids. w05-head2s scores it as its second set.
2. **CPU gate (`src/hgate.py`):** quantize under hin-gen vs hin-all with exllamav3's own code and a scalar trellis stand-in.
   The error is measured on gen rows (H_gen metric, Fisher-weighted), in-sample for the gen arm. **< 2% at both K2 and K3 →
   stop** (no GPU quants).
3. **K3 donors:** vocabulary order, mean-row, σ 0.005, `--hin hin-gen` vs `--hin hin-all`, seeds 0/1/2. w05-head2s's `one`
   arm (published calib H) is reported beside them if present.
4. **`headarms` + `judge`** on the gen-row dump (primary) and the calib-A dump.

## Draft pass rule (for the lead to commit before data)

- **Gate:** step 2's ratio ≤ 0.98 at K2 or K3, else stop.
- **Adopt the gen-row head Hessian:** on the gen-row dump (all documents, document bootstrap), `hgen` vs `hall` must meet all
  of:
  - mean KL ratio over 3 seeds ≤ **0.97**;
  - CI upper bound < 1;
  - every hgen seed below every hall seed;
  - guard ratio ≤ **1.1**.

  The calib-A dump must not contradict it (ratio < 1).
- **Pass →** headremake.sh's donors take `--hin hin-gen` (all K). The gen-row dump replaces calib-A as headsim's pricing set
  for E1's curve: re-run `pickcurve` on it and report rank changes among plans.
- `hcalib` vs `hall` (published H on all docs vs this run's 2M qad tokens) is reported only. It separates "which rows" from
  "how much data".

**GPU time:** headx ~1 h on both cards; 6 quants ~1–1.5 h on one card; scoring ~30 min. The quants only run if the gate says go.

**Run:** `CARD=0 ./run.sh` on iceman, when both cards are free for step 1. Results go to `/data/yaqa/w05-headgen/result.txt`.

## Self-tests

- **`headx_test.txt`** (colo CPU; tiny random Qwen3.5 with the 27B's vocabulary and layer pattern, real calib golden meta):
  - the run completes;
  - all 3,000 dumped rows' next tokens match the golden's row bookkeeping, and all are gen;
  - dumped vectors equal an independent forward exactly at the mapped positions (relative error 1.4 one position off);
  - both Hessians are PSD and Ha − Hg is PSD.

  **PASS.** Run: `python3 src/headx_test.py <27B config.json> <golden meta> <pools-v3.json> <workdir>` (needs transformers +
  accelerate).
- **`hgate_nulltest.txt`** (colo CPU; the real 27B head and the calib H passed as both arms): ratio exactly 1.0000 at K2 and
  K3, and the gate says stop, as it must.

No shared code is changed: headx.py is standalone; `headarms.py`, `judge.py` and `cpuq.py` live in `../w05-head2s/src`.

## Runtime cost (user rule, 2026-11-03)
none (head statistics).
