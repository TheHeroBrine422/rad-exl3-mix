# v3/df2 — DFlash2 drafter + vision tower in qwen35_exl3 (#180, w153)

Radiance fork branch `qc-exl3-v3-df2` (radiance-hero, worktree `/workspace/wt/w153-radiance`), one commit on
qc-exl3-v3 e054e71: **9f0a059** (rebased from 4b124c7 on 179220f; lead note 11-13 9:45 PM). Not pushed.

What was already there: qwen35_exl3 declared the DFlash2 drafter (taps at 5/19/33/47/61 → `dflash_src`, the drafter
from the container's `--draft-model`, the stock `rad_block_dflash2.h` block) and the vision tower + projector since
8afc1fc. What this branch adds: candidates rescored against a copy of the **original bf16 lm_head**
(`dflash.rerank_head.weight`, a recipe line; int8 g128 or block fp8), the startup note naming the head grid and
whether rescoring is on, the "do not declare MTP or vision blocks" comment lifted, a host test. The draft head
(`dflash.draft_head`) maps the checkpoint's `lm_head.weight` too, never the trunk's imported `output.weight`.
Mask token 248070: the drafter embeds it through the trunk's `token_embd`; with the bf16 embedding the row is the
checkpoint's, exactly (a table-quantised embedding refuses speculation at declare, `qwen35_exl3.cpp` embed_tq).

## Files

| file | what |
|---|---|
| `build153.sh` | ds8: qwen35_exl3 at 4b124c7 into w147's home (qc-exl3-v3 179220f) → `home` |
| `build153r.sh` | ds8: exl3i + exl3quant + qwen35_exl3 at the rebased 9f0a059 over rad10-perf/home → `home-r` |
| `run153.sh` | ds8 HIP 7, restartable stages `pack kld serve reb cmp` (header has the details) |
| `recipes/` | P0 text-only, P1 MXFP4 drafter + int8 rerank head, P2 block-fp8 drafter + rerank, P3 MXFP4 no rerank; `P4-bf16-rr.recipe.refused` |
| `whash.py` | per-weight byte hashes of two containers (trunk P0 vs P1, tower / drafter vs IQR-V4 S) |
| `tiemargin.py` | first diverging step of two greedy runs + both sides' top-1 − top-2 margin there |
| `bsum.py` | bench summaries (`python3 bsum.py data/res P1on IQRon ...`) |
| `data/` | `res/` (kld json, bench, dcmp / tie, image + text generations, metrics, serve logs), pack logs + rad-info |

Pack = V3 K4.125 trunk (mul1i8, a8g128 on the 336 trunk weights, as w138's iqrcmp packs) + R16s003 K4 head + bf16
embedding. Bench prompts / tool = `v3/bench/iqrcmp/run.sh`'s (w137 prompts, warm 1 + 3 runs, ctx 0 gen 256, ctx 8k
gen 32). IQR = IQR-V4 S (`qwen3.8-27b-iq4r-s-v4.rad`, its own drafter lines), served by the same binary.

## Recipe lines (rad-convert, packs carry both)

`rad-convert ... --draft-model <z-lab/Qwen3.8-27B-DFlash2> --recipe R` with a home whose qwen35_exl3 is this branch
(the pack home decides what is declared, so a home without it writes a text-only container). The tower needs no
line (bf16, from the checkpoint). Drafter lines = IQR-V4 S's plus the rescoring head:

```
dflash.draft_head.weight rtn codes=u2 zero=u8 group=128 scale=f16 rule=search
dflash.fc.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.blk.*.attn_q.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.blk.*.attn_k.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.blk.*.attn_v.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.blk.*.attn_output.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.blk.*.ffn_gate_up.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.blk.*.ffn_down.weight rtn codes=fp4_e2m1 group=32 scale=e8m0
dflash.rerank_head.weight rtn codes=i8 group=128 scale=bf16 clamp=sym
```

(P2: the seven linears `rtn codes=fp8_e4m3 block=128x128 scale=bf16`.) **radmix:** a mix keeps BASE's metadata and
every weight it does not name, so build the BASE with the lines above and every mix inherits tower + drafter; `add`
from a donor is not enough (the `draft.*` / `vision_config` metadata would be missing).

**bf16 drafter: not servable.** The stock DFlash2 block runs its linears as `fp8a8` or `mxfp4a8`; leaving them bf16
(P4) is refused at declare (`dflash.fc.weight is declared fp8_e4m3, and ... holds codes as bf16`). Block fp8 (P2) is
the nearest to the bf16 drafter; a bf16 path would be a change in shared `arch/common/rad_block_dflash2.h`.

## Results (ds8 HIP 7; packs 11-13 6:36–6:58 PM CST, GPU 11-13 9:23–9:43 PM CST)

Bytes (not counted toward size): tower `v.` 921,894,912 B (0.859 GiB, byte-identical to IQR-V4 S's); drafter
`dflash.` P1 2,946,625,536 B (2.744 GiB) = IQR's 1,655,361,536 B (byte-identical, 410 weights) + rerank head
1,291,264,000 B; P2 3,757,916,160 B; P3 1,655,361,536 B. Trunk / head / embedding P0 vs P1: 787 weights
byte-identical. VRAM static weights (spec off → on): 13.20 → 15.95 GiB (P1), 14.75 (P3), 16.71 (P2); KV 17.51 →
14.65 GiB at max-model-len 40960, 8 seqs.

1. **DFlash2 off = today:** golden-search27c-c-cut13 (7,451 positions) KLD rows sha bb6b5a8a8dd77831 for w147's home
   on P0, this branch on P0, P1 spec 0, P1 at depth 7, and the rebased build on P0 / P1 spec 0 / P1: all
   byte-identical (KLD 0.07696, top-1 94.75 %).
2. **Exactness:** spec on vs off, greedy: 6 / 8 text and 6 / 9 image prompts identical to the end (82.5 % / 83 % of
   tokens before a split); every split is a near-tie on both sides (top-1 − top-2 margin 0.013–0.030 nats), i.e.
   the M = 8 verify's numerics vs M = 1 decode, not a wrong token. IQR-V4 S splits the same way (7 / 9, margins
   0.013–0.077). The drafter does not change outputs: P1 on = P2 on = P3 on token for token (1024 / 1024). The
   rebased build = the pre-rebase build bit for bit (logprobs, KL 0, text + images).
3. **Acceptance and speed** (tokens a step, client tok/s; two rounds each):

   | | ctx 0 tok/step | ctx 0 tok/s | 8k tok/step | 8k tok/s | step ms (on / off) |
   |---|---|---|---|---|---|
   | V3 P1 (MXFP4 + int8 rerank) | 2.75 / 2.99 / 2.86 (rebased) | 63.6 / 69.0 / 67.4 | 5.72 / 5.93 / 5.93 | 117.9 / 125.1 / 106.8* | 42.6 / 31.6 |
   | IQR-V4 S | 2.49 / 2.46 | 72.8 / 72.0 | 4.86 / 4.81 | 125.5 / 125.9 | 34.1 / 25.7 |
   | V3 P2 (fp8 + rerank) | 2.84 | 63.3 | 6.37 | 129.6 | 44.3 |
   | V3 P3 (no rerank) | 2.38 | 55.2 | 5.50 | 117.8 | 42.6 |

   *one of three runs at 70.6 ms a step (interference); the other two 43.2 ms. Our drafter accepts more than IQR's
   (+10–20 %); the verify step costs ×1.35 the decode step on both (42.6 / 31.6 vs 34.1 / 25.7), so the e2e gap is
   the trunk's decode (#175): ctx 0 −5 to −12 %, 8k at par to −6 %. Rescoring vs not: +16–26 % tokens a step at
   ctx 0 (P1 vs P3). Small samples (3 × 256 / 3 × 32 tokens): differences under ~10 % are noise.
4. **Vision:** 9 image prompts (hopper, gull, lecture, dashboard, quad, red-wide) answer sensibly on our pack
   (e.g. quad-colors: red / green / blue / white by position; gull-what: "a seagull ... Herring Gull"). Tower bytes =
   IQR's; image outputs ours vs IQR differ as the trunks do (no prompt identical to the end, per-step KL 0.0195,
   first splits at margins 0.07–1.2: a different trunk, not the tower).
