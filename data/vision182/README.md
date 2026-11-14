# data/vision182 — #182 vision KLD eval (w156, 2026-11-14)

Code and method: `eval/vision182/README.md`. Working copies on ds8 `/data/q27/w156/` (packs, goldens, rows files).

## Files

| path | what |
|---|---|
| `set/items.jsonl`, `set/text-items.jsonl` | the 450 image items and the 120 text-control items (images: ds8 `w156/set/img/`, rebuilt by `mkset.py`) |
| `gen/ans-*.jsonl` | the model-written answers (IQR-V4 S, greedy, thinking off, ≤ 512 tokens) |
| `corpus/meta.jsonl` | per document: kind, scored tail, answer rows (the gen mask) |
| `gold/vis182/kld.json` | the bf16 reference's manifest (TP2 cards 7 + 4, bf16 KV, top-512; 570 docs, 175,313 scored positions: image answers 107,433 + questions after the image 14,507; text answers 34,742 + questions 18,631) |
| `res/*.json` | radiance KL reports: `TAG-vis` (vision golden), `TAG-s173x` (golden-s173e-x), `reg-*` (text regression), `A8-vis` (vs gold/k6f) |
| `res/ana182.json`, `ana182-a8.json`, `ana182-qad.json`, `swap182.json` | `ana182.py` / `swapana.py` outputs |
| `logs/` | run182 / gen182 logs |
| `packs/` | radmix logs and mix files (tower, swaps, f16 act) |

## Results (ds8 card 7, 11-14 1:40–6:00 AM CST)

Metric: mean KL(bf16 ‖ candidate) over the **answer rows** (model-written tokens; the board scores model-written rows),
pooled over positions. "text ctrl" = the 120 dolly items in the same golden (same generator, template, thinking off,
scoring): the matched comparison. "s173e-x" = golden-s173e-x (130 xcorp docs, corpus-style text, all rows; no calib).
Ratios: 95 % CIs from 4,000 document bootstraps. Entropy-matched: the text control reweighted to the image rows'
distribution over 6 bins of the bf16 reference's entropy (image answers are more certain on average).

| cand | image ans | text ctrl ans | s173e-x | image / text ctrl | entropy-matched | top-1 image / text |
|---|---|---|---|---|---|---|
| V2 2.50 | 0.0823 | 0.0864 | 0.1968 | 0.95 [0.87, 1.05] | 1.39 [1.31, 1.47] | 92.19 / 90.48 % |
| V2 4.00 | 0.0102 | 0.00751 | 0.0531 | **1.36 [1.21, 1.54]** | 2.03 [1.86, 2.24] | 97.52 / 97.02 % |
| mad K2.5 | 0.0587 | 0.0584 | 0.1531 | 1.01 [0.91, 1.11] | 1.47 [1.38, 1.58] | 93.73 / 92.26 % |
| mad K4 | 0.00891 | 0.00677 | 0.0360 | **1.32 [1.16, 1.50]** | 1.97 [1.77, 2.19] | 97.60 / 97.26 % |
| mad K6 | 0.00098 | 0.00063 | 0.00981 | **1.57 [1.39, 1.77]** | 2.33 [2.11, 2.58] | 99.24 / 99.22 % |
| A8 alone (K6 a8g128 vs K6 f16, ref gold/k6f) | 0.00034 | 0.00021 | – | **1.68 [1.45, 1.97]** | 2.51 [2.22, 2.87] | 99.54 / 99.53 % |

- **Vision degrades no more than text at 2.5 bpw, and progressively more at higher rates:** the image / text ratio goes
  1.0 → 1.3 → 1.6 from K2.5 to K6 (V2 the same: 0.95 → 1.36). From K2.5 to K6 the text control's KL falls ×93, the
  image answers' ×60 (ln-slope 1.29 vs 1.17 a bit). As bpw-equivalent on the text control's K curve: image answers sit
  ~0.0 / 0.2 / 0.4 bpw behind at K2.5 / K4 / K6. In absolute terms the excess is small where it is largest (K6:
  0.00098 vs 0.00063).
- **Not specific to the text-only Hessians (so far):** a perturbation that owes nothing to calibration — int8 RTN of the
  same trunk's activations (A8), measured against the same pack with f16 activations — has the *same* image / text ratio
  (1.68 raw, 2.51 entropy-matched) as K6's weight quantization (1.57 / 2.33). Image-conditioned positions are more
  sensitive to any small trunk perturbation; the growing ratio with bpw is what happens when the text error shrinks to
  that regime. Caveat: A8 is not calibration-free with respect to *activation* statistics (image tokens may carry larger
  outliers, which RTN int8 also feels), so this tells "image positions are fragile", not "image-aware Hessians would not
  help" — only quantizing with image-augmented Hessians answers that.
- Per kind (image / text ctrl, raw): scenetext 1.6–2.7× at every rate and photo 1.3–1.5× are the persistent ones; screenshot
  rises 1.0 → 1.8 → 2.3 (K2.5 / K4 / K6), multi-image 1.0 → 1.2 → 1.5; charts and documents stay at or below ~1 (0.6–1.04×). Answers
  in charts / documents / screenshots are long, copy-like (OCR, tables, HTML) and confident.
- The question rows right after the image (user text, not model-written) carry 3–6× the text control's question-row KL
  (wide CIs); the first tokens after an image are the most image-dependent.
- V3 vs V2 at the same K (paired, same documents): K2.5 / V2 2.50 = image 0.713 [0.695, 0.731], text 0.676 [0.655, 0.700]
  (image gains slightly less: ×1.055 [1.011, 1.099]); K4 / V2 4.00 = image 0.872, text 0.902 (×0.967 [0.901, 1.033]).

### QAD (#181, w154's containers + the tower; the #171 2.5 incumbent in served form, a8g128)

| vs ctl (unhealed) | image ans | text ctrl ans | image / text | w154's text rows (verify27 c / x181 x) |
|---|---|---|---|---|
| ctl | 0.0647 | 0.0632 | – | 0.04337 / 0.05619 |
| a1m (arm A, 1 M decayed) | 0.942 [0.935, 0.949] | 0.939 [0.930, 0.947] | 1.003 [0.992, 1.015] | 0.04063 / 0.04970 |
| a4m (arm A, 4 M decayed) | 0.978 [0.967, 0.989] | 0.988 [0.976, 1.000] | 0.990 [0.973, 1.006] | – |
| p1m (calib-only, 1 M) | 0.936 [0.927, 0.946] | 0.961 [0.951, 0.972] | **0.974 [0.960, 0.989]** | 0.04114 / 0.05226 |

**Text-only QAD does not hurt vision:** every checkpoint lowers the image answers' KL about as much as the text
control's (a1m −5.8 % vs −6.1 %), p1m slightly more on images (−6.4 % vs −3.9 %). Per kind no kind gets worse except
a4m on photo (×1.015).

### Swap test: where is vision more sensitive than text? (K2.5 with one group of trunk weights from K6)

Share of the K2.5 → K6 answer-row KL gap each upgrade closes (paired document bootstrap on the difference):

| upgraded to K6 | trunk weights | image KL | text KL | image gap closed | text gap closed | image − text [95 % CI] |
|---|---|---|---|---|---|---|
| layers 0–15 | 84 | 0.0559 | 0.0542 | 0.05 | 0.07 | −0.02 [−0.04, −0.01] |
| layers 16–31 | 84 | 0.0465 | 0.0452 | 0.21 | 0.23 | −0.02 [−0.04, +0.01] |
| layers 32–47 | 84 | 0.0364 | 0.0394 | 0.39 | 0.33 | **+0.06 [+0.03, +0.08]** |
| layers 48–63 | 84 | 0.0425 | 0.0358 | 0.28 | 0.39 | −0.11 [−0.14, −0.08] |
| attention (16 layers: qg / k / v / o) | 64 | 0.0525 | 0.0535 | 0.11 | 0.08 | **+0.02 [+0.01, +0.04]** |
| GDN (48 layers: qkv / z / out) | 144 | 0.0485 | 0.0477 | 0.18 | 0.18 | −0.01 [−0.03, +0.01] |
| MLP (gate_up / down) | 128 | 0.0222 | 0.0159 | 0.63 | 0.74 | −0.10 [−0.12, −0.09] |

Images lean on the middle-late layers 32–47 and on the 16 attention layers (the only groups that close more of the image
gap than of the text gap; the same in every kind for 32–47); text leans more on the last quartile and the MLPs. The
effects are modest (≤ 0.11 of the gap) and measured at K2.5, where images are not worse than text overall; a K4 → K6
swap (where the excess appears) would be the sharper test. If image-aware Hessians are tried (V4), layers 32–47 and the
attention layers are where to look first.
