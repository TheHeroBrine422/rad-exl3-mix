# eval/vision182 — vision KLD eval (#182, w156)

User 11-14: is text-only quantization (and QAD) hurting vision? Our Hessians, H_out, goldens and QAD data are all
text, so this measures the KL divergence vs the bf16 model on image + text chat items, next to text at the same size.

## Pieces

| file | what |
|---|---|
| `mkset.py` | (colo) 450 public image items in 7 kinds from pinned parquet shards (seed 182): photo 75 (A-OKVQA questions + detailed descriptions), scenetext 45 (TextVQA), document 75 (DocVQA questions + full-page transcriptions), infographic 30 (InfographicVQA), chart 75 (ChartQA questions + chart-to-table), screenshot 75 (WebSight screenshot→HTML, ScreenSpot element questions), multiimage 75 (NLVR2 pairs, MMMU questions with ≥ 2 images); 558 images, original bytes (JPEG / PNG) |
| `mktext.py` | (colo) the format-matched text control: 120 databricks-dolly-15k prompts (15 per category) |
| `gen182.py`, `gen182.sh` | (ds8 card 7) model-written answers: IQR-V4 S (third-party quant, neutral to every candidate) served by radiance with the tower, greedy, thinking off, ≤ 512 tokens |
| `mkcorpus.py` | the KL corpus: the checkpoint's own chat template (images first, then the question; thinking off), image parts → radiance's media marker, + answer + `<|im_end|>`; `score_tail` = every token after the last image (text: after `<|im_start|>user\n`) |
| `run182.sh` | (ds8) stages `corpus reg gold pack v2 score` (header has the details) |
| `ana182.py` | report: answer-row KL per candidate and kind, vision / text ratios with document-bootstrap CIs, entropy-matched ratio (`GOLD=`, `CANDS=` for another reference / list) |
| `pair182.py` | paired candidate-vs-base ratios on image answers and on the text control (QAD vs its unhealed control, V3 vs V2) |
| `qad182.sh` | w154's #181 QAD containers + the tower → `packs/Q-*.rad`, `cands-qad.txt` |
| `swap182.sh`, `swapana.py` | swap test: K2.5 with one group (layer quartile / attn / gdn / mlp) from K6; share of the K2.5 → K6 gap each closes on images vs text |
| `mini.sh`, `flip.sh`, `chain182.sh` | smoke tests of the media KL path (5 docs, K4 as the reference; pictures flipped), a stage chainer |

Radiance: branch `qc-exl3-v3-kldmm` (5fa0b46, one commit on qc-exl3-v3 b83698d; core only: `core/kld.{h,cpp}`,
`core/engine_kld.cpp`, a comment in `core/engine_bringup.cpp`). The KL mode takes pictures: a corpus line may carry
`"media": [files]` (one per `<__media__>` marker in its prompt) and `"score_tail": N`; with the tower on the card
(`--mm-max-patches N`, KL mode's default stays 0) the container's processor expands each marker as the server's chat
path does and the request carries the pictures. The manifest records each document's files and where their token runs
start (`media`, `media_at`); `--kld-ref` processes the files again and refuses a document whose runs differ from the
reference's tokens, or a reference with media on a run without the tower. Text-only runs are unchanged.

## Checks

- Text regression: V2 3.00 on golden-search27c-c-cut13, rad10-perf vs this build: rows byte-identical (sha 32ec2ac7…).
- Media path: a 5-doc reference recorded with K4 and scored by K4 itself: KL 0 exactly; the same with every picture
  flipped top-bottom (same size, so the same token runs): KL 0.15 on the image docs, 0 on the text doc — the tower is
  in the path. A run without `--mm-max-patches` refuses the reference.
- The candidates' text numbers reproduce w151's stage-B rows: mad K2.5 / K4 / K6 on golden-s173e-x 0.15313 / 0.03602 /
  0.00981 = w151's M arm on the same 130 documents (5 digits), although these packs are native `codebook=mad` packs from
  this build with the tower (w151: the i8-stamped shim).
- A8 test (`run182.sh a8`): K6 with f16 activations (radmix `act ... f16` on the 336 trunk linears) records its own
  reference over the corpus on one card; K6 with a8g128 is scored against it — a calibration-free trunk perturbation.
- bf16 on one card (`--placement layer_offload`) segfaults in libavx's CPU GDN (`avx_gdn_conv_prep_v2`), the same crash
  w124 saw serving images; the golden is TP2 (cards 7 + 4, as gold3.sh), bf16 KV, top-512.

## Candidates

`W/cands.txt` (TAG PATH). Packs: mad K2.5 / K4 / K6 (ladder-mad `final.recipe`, codebook=mad, act a8g128 on the 336
trunk linears = w151's a8.mix, bf16 tower from this build's qwen35_exl3), V2 2.50 (HF Hero422/qwen3.8-27b-exl3-v2-rad) /
4.00 + the tower (radmix `add v.*`, 333 weights, from K2.5's pack; V2 runs f16 activations as released). The tower is
byte-identical bf16 in all of them, so differences are the trunk's. To add a candidate (e.g. a QAD export): a text-only
container is fine — `radmix OUT BASE packs/tower.mix`, a line in `cands.txt`, `run182.sh score`, `ana182.py`.

## Results

See `data/vision182/README.md`.
