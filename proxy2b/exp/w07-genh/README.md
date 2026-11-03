# w07-genh: gen-weighted input Hessians (ranked.md C13)

**Question.** Should prompt, tool and system positions count less in the input Hessian? The challenge scores only the
model's own (gen) positions; the published H_in is a plain sum over every calib position. Lens sources: hout E5,
measure O16, crossarea C4.
- This differs from the failed gradient-weighted H_in (#33-07, #33-10): bounded 0/1-style weights, no gradients, no
  other corpus.
- Unscored positions still enter attention and the GDN state of later scored ones, so w0 = 0 risks starving those
  directions. Hence a bounded w0 and the dilution below.

**Mechanism.** `hgen.py` (hcollect.py's groups, chunking and arithmetic) sums x xᵀ over the **unscored** positions
only (H_ng):
- "Unscored" means gen.u8 = 0, everything before score_from, and each document's last position.
- Documents: the non-verify pools (fit / search / qad, pools-v3) minus the documents of **both eval goldens**
  (g08-fit1, g08-search1).
- `mkgenh.py`: H(w0) = H_calib − (1 − w0) H_ng.

That re-weights those positions to w0 and keeps every other calib position at 1. In particular the eval documents
keep exactly base's weighting, so no arm is flattered by its own score masks. The verify pool, held out for later
slices, is left unweighted too, as a 27B run would have to leave ver3 / ver4.

**Dilution (measured on the calib meta).**
- Of 10.26M calib positions, 2.53M are unscored.
- The weighted set D holds 1.05M of them: 61k fit, 57k search, 936k qad.
- Verify keeps 1.06M and the eval docs 0.42M at weight 1.
- So w0 = 0 removes ~10% of H's positions: a mild change by construction.

## Arms (0.8B, i8, base settings, `--hin` replaces all 96 groups)
| arm | w0 |
|---|---|
| g0 | 0 (unscored positions of D dropped) |
| g3 | 0.3 |

**Design:**
- **Screen:** both arms at seed 0, K3 and K2, scored on **g08-fit1**. Per K the lower one is picked (summ.py).
- **Confirm:** the pick at seeds 0 / 1 / 2 on **g08-search1**, against `k{K}base{,_s1,_s2}` (existing).
- **Collector gate first:** hgen.py with every position of every document, layer 0 only, must reproduce calib's
  layer-0 Hessians:
  - rows equal;
  - rel diff ≤ 1e-6 on the layer-input group (the norm of the embedding: kernel-independent);
  - rel diff ≤ 1e-3 on the other three (they pass the GDN kernels, whose version may have moved since /data/h08 was
    collected).

## Draft pass rule (for the lead to commit before data)

Per K, for the fit-picked arm, on g08-search1:
- **PASS:** mean-KL ratio (3 seeds each, arm / base) ≤ 0.97 **and** every arm seed below every base seed.
  This is ranked.md C13's ≤ 0.97× with #33-16's seed convention.
- **FAIL:** mean-KL ratio ≥ 1.00.
- **INCONCLUSIVE otherwise:** not adopted.
- **Gate:** the collector gate must PASS.
- **Reported beside the rule:** both arms' fit ratios; per-group share of tr H removed and min diagonal
  (mkgenh stats.jsonl); paired z per seed.

## Run

```
proxy2b/exp/w07-genh/run.sh     # 4070, no patch needed. Results /data/yb/w07-genh/result.txt, summary.jsonl
```
- Idempotent and resumable.
- **GPU time ≈ 40 min:**
  - gate ~1–2 min (forward stops after layer 0);
  - H_ng collection ~5–10 min (~5.9M tokens);
  - 8 quants;
  - ~10 evals.

## What the result changes
- **PASS:** a 27B H_ng collection.
  - It is forward-only, over the non-verify calib documents minus the 27B eval slices, on both iceman cards, in layer
    passes (the sums are the size of the published Hessians).
  - H(w0) then joins the per-level requant (#58).
  - The same difference form keeps the published H for everything else, so the 27B run needs no H_calib
    recollection.
  - Not scripted yet.
- **FAIL / inconclusive:** C13 closed. That also bounds the cheaper "score-weighted" ideas on the input side; head
  statistics from gen rows (E4) are a separate question.

## Files
- `hgen.py`, `mkgenh.py`, `run.sh` (pick / verdict via `../w07-denoise/summ.py`).
- `selftest/test_hgen.py` → `test_hgen.out`: **ALL PASS**. Real 0.8B, colo CPU, layers 0–1, 6 qad documents with mixed
  masks.
  - H_ng + H_gen = H_all (rel 2e-7), and the rows add up.
  - The scored count equals gen.u8's row by row.
  - `--exclude` drops a golden's documents.
  - mkgenh: w0 = 1 gives calib, w0 = 0 gives calib − H_ng; a non-positive diagonal is refused.
- `../w07-denoise/selftest/dryrun.out`: this run.sh end to end with stubs.

## Runtime cost (user rule, 2026-11-03)
none (input-Hessian weighting).
