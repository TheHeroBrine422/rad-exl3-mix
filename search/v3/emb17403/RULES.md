# RULES — #174-03 trellis embedding back in the budget (w157, 2026-11-14)

Written and committed before any run below starts (lead rule 11-08). Brief: `rad-local/workers/instr3/w157.md`.
Context: Deadcode (11-14) counts the embedding as VRAM; the user put it back in the budget. Size = container bytes
(trunk + head + embedding) = the board metric.

## Build

radiance-hero `qc-exl3-v3-emb157` = w144's three commits (d9bbc7f, 5679e6f, 1deec9a) rebased onto `qc-exl3-v3`
3bc8ad2, plus one commit: the DFlash2 drafter embeds through `embed_exl3i` over the trunk's trellis table (block hook
`op_embed_ext`, off by default). All five plugins built together (exl3i interface 5) on box2 for ROCm 10.1 (rad10-perf
install) and 7.14 (rad/install), w152's `plugb.sh` with the tree's `rad_block_dflash2.h` (OVL=1).
**Pass:** every host test; w144's `embed_exl3i` host / GPU tests (device == host == oracle bit for bit, `--host` too);
every exl3i device suite of w152's `ctall.sh`; on both toolchains.

## 1. Serving identity on a mad pack (box2 card 1)

Trunk: ds8 `packs-mad/q27v3-635bac02-k2.5-mad-s6c57a3d1.rad` (mad K2.5, its bf16 head, bf16 embedding) with
`act <w> a8g128` on every trunk linear (A8, as the search's containers). Arms, all radmix of that pack:
- **B**: the pack's own bf16 embedding (act lines only);
- **T**: B + `token_embd.weight` from w144's donor `emb-t-i8-K2.5.rad` (mul1i8 K2.5 trellis);
- **R**: B with the bf16 token table replaced by the oracle rows of the same trellis (w144 `embref.py`).
Confounder check (w144 `mix144.sh check`): B / T / R weight directories identical outside `token_embd.weight`, metadata
identical. Scored on **golden-s173e** (whole, bf16 KV, max-model-len 16384, one sequence; this build).
- **Pass (identity):** T's per-position `.rows` byte-identical to R's.
- **Reported:** T vs B, the embedding's error: paired KL ratio (per part c / x and pooled = geometric mean of the part
  ratios, 2,000 part-stratified document bootstraps, docs from `rxrun.hostd.doc_sums` with the golden's gen mask).
  Expected small and positive (w144: Lloyd l3 vs bf16 +2.2 %; trellis t2.5 below that).

## 2. Time (box2 card 1)

B vs T, w144's perf protocol (`perf144.sh`: 8k prefill-only × 4, bench.py ctx 8k gen 32 / ctx 0 gen 256, warm 1 + 3;
`--profile-ops` per-op µs of `embed_exl3i` / `embed_lookup`); VRAM as served from the load report. Reported only.

## 3. DFlash2 with a trellis embedding (box2 card 1)

w153's `P3-mx` (K4.125 mul1i8 a8g128 trunk, R16 K4 head, bf16 embedding, MXFP4 drafter, rescoring on the main head)
vs **P3-t4** = P3-mx + `token_embd.weight` from `emb-t-i8-K4.rad`. The mask token's row (248070) and the anchor rows
are the trellis's rows, decoded exactly by `embed_exl3i` (no bf16 row kept).
- **Pass (works):** P3-t4 serves with the drafter (no refusal), its drafter note names `embed_exl3i`, and greedy decode
  on w155's 16 fixed prompts × 256 tokens finishes without errors.
- **Reported:** draft acceptance (tokens / step from /metrics) P3-t4 vs P3-mx, drafter on; greedy text P3-t4 drafter on
  vs P3-t4 drafter off (spec 0), w153's `dcmp.py` / `tiemargin.py` (splits at near-ties are the engine's verify-vs-decode
  drift, as w153 / w155 found).
- **Follow-up rule:** if P3-t4's tokens / step is more than 5 % below P3-mx's, an exact bf16 mask-token row beside the
  trellis is proposed to the lead (not built here); otherwise the exact trellis decode stands.

## 4. Eighth-step donors (box2 card 1)

The 21 missing eighth steps K1.625 … K4.875 (t-i8 = mul1i8), w144's tool and settings exactly (`quant157.sh`: ourq
`--only-head` on the token table, identity input Hessian, `--qtf2 --frac-ext --scaled-hessian --sigma-reg 0.025
--codebook i8`), from the code tree w144's levels were made with (stage `code-old-c015001`, code stamp
c816b607e7ee21c1, checked per level).
- **Pass (determinism, before any eighth level is used):** a rerun of K2.5 gives `q-head.safetensors` byte-identical to
  w144's `t-i8/K2.5`. If not, the eighths are still built but flagged, and the lead decides.
- Donors: slim containers (radmix of a w144 donor with every weight but `token_embd.weight` dropped: the radmix input
  the menu's `src.path` names), `/data/q27/emb/trellis/emb-t-i8-K<k>.rad` on box2.

## 5. Eighth-step prices: floor model + spot check (box2 card 1)

Menu `err` = w135's floor model (`data/emb174/embprices174.json`: err = F + a × rel, F 2.0681e-4, a 0.024800) on each
level's measured rel (row error of the real table, identity Hessian), as w144 priced the half steps.
Spot check (reported, decides nothing): arms **t1.625, t2.125, t3.375** = B + that eighth donor, scored as §1, each vs B.
Prediction: log r_pred = log(1 + (err_arm − F) / KL_B,c), KL_B,c = B's calib-part mean KL on s173e (bf16 embedding:
rel 0, err F). An arm **agrees** when its measured pooled log ratio vs B lies within 2 bootstrap SE of log r_pred; the
calib part is reported beside. T (t2.5, a half step w144 priced) is read the same way as a fourth, known point.
If 2 or more of the 3 eighths disagree, the eighth prices are flagged in the menu notes and the lead decides.

## 6. Menu block

`token_embd.weight` options: every t-i8 level K1.5–5.0 in eighths, the half steps K5.5–7 (t-i8), and mul1 K6.5 / 7 / 8
(t-m1); bytes = the donor's token_embd entry (exact); err as §5. The old non-trellis options (Lloyd base / e2 … e7, bf16)
are kept only if on the lower convex hull of (bytes, err) of all options together (their err as in menu174).

## Data note

golden-s173e: a two-part search slice (43 calib + 130 xcorp docs, 195,082 gen positions) already read 6 times by
#177-03's stage-B spot check. These reads (B, T, R, 3 eighths: 6 scores) are an identity check and pricing rows; they
make no accept / reject decision and select nothing. No verify, no public.

## Amendment 1 (11-14 4:35 AM CST, after the K2.5 determinism rerun, before any eighth level)

The K2.5 rerun stopped on its stamp check: `211924488ac0696e`, not w144's `c816b607e7ee21c1`. ourq's code stamp hashes
each source file's **path** with its contents (`_stamp`: `h.update(f.encode())`), and the same tree now lives at
`stage/code-old-c015001` instead of `stage/code`; the 421 source files are the same list but for that one path. The
rerun's four tensors (`lm_head.trellis / suh / svh / mul1`) are **identical** to w144's K2.5 (torch.equal on each) and the
safetensors metadata differs only in `ourq_stamp`. So: the determinism check is read on the tensors and the metadata
other than `ourq_stamp` (PASS), and `quant157.sh` checks for this path's stamp `211924488ac0696e` from here on.
