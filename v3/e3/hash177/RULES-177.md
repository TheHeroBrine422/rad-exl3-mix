# #177 stage 3 rules: real 27B KL of the hash codebook (w139, written 2026-11-13 ~10:35 AM CST, before any stage-3 data)

Brief: `rad-local/workers/instr3/w139.md` stage 3 + lead notes `notes/w139.md`. Stages 1–2 (speed, proxy) are in the LOG.
This file fixes what stage 3 measures and how it is read before any hash level is quantized. It is committed before the
first stage-3 job starts; amendments get a date and go at the end, never above.

## What is compared

- **Arm H (hash):** IQR-style computed code on EXL3's 16-bit trellis state (L = 16; L 17 / 18 have no GPU quantizer):
  q(s) = bytesum(((s · A + B) mod 2³²) & 0x3F3F3F3F) − 126, value q · 4 k_inv (mul1i8's int8 grid and step), **A = 0xd0474b,
  B = 0x2545F491** (w27's searched global A; the constant #100c used at 0.8B, so the two runs are comparable). Quantizer:
  `v3/e3/exl3hash` (w48) through `rxq` with `rxq-hash.patch` (engine / deq / mix only).
- **Arm I (control):** the existing mul1i8 levels of box2's ladder, code stamp 43e08d796b6bcfbf
  (`stage/code-old-8f73ee4`), settings stamp 144dd6e18dcda787.
- **Levels:** K2.625 and K4.125 (the box2 ladder's i8 levels with `out/` kept; the brief's K2.5 / K4 have no level dir
  with this code on box2, and 2.625 / 4.125 are V3's own fractional rates). Arm H runs **the ladder level's own job
  specs** (`specs/L*.json`, 256 a level) with `codebook` = hash and the level name `…-hash`, config = the patched tree's
  `v3.toml` (settings stamp must be 144dd6e18dcda787), same inputs (ckpt, calib Hessians, H_out bn1920). Steps as the
  ladder's DONE: jobs, assemble, post (in-job C9), recipe. No swj.
- **Gate before any H job (identity):** two i8 jobs (K4.125 `L00-g0`, K2.625 `L05-g1`) re-run with the patched tree must
  give trellis / suh / svh tensors byte-identical to the ladder's `out/` (only the code-stamp metadata may differ). Fail:
  stop, no H data is read.
- **Packs:** both arms packed from their `final.recipe` with the same command (hostd's pack_cmd: rad10 home-pack,
  `--no-error`); the H level's `final/stamp.json` args.codebook is shimmed to `i8` for the importer (hash codes in mul1i8's
  slot; the original kept as `stamp-hash.json`). Arm I's final is re-made from its `out/` with the unpatched tree
  (assemble / post / recipe), since box2's ladder deleted its finals. Mixes: radmix over the pack itself with
  `act <w> a8g128` on the 336 trunk weights (A8, as V3 ships).
- **Confounder check (AGENTS.md, before reading any ratio):** every non-trunk tensor (embedding, head, norms, GDN params)
  byte-identical between the two packs of a level (sha256 per tensor from the container index), trunk byte counts equal,
  container sizes equal.
- **Scoring:** radiance from `/data/q27/rad10-perf/install` (bf16 KV, max-model-len 16384, max-num-seqs 1). Arm I with
  home `w139/home-base3` (= rad10-perf's exl3i, device code checked byte-identical), arm H with `w139/home-hq`
  (EXL3I_BENCH_IQRH 1, A 0xd0474b: the hash in mul1i8's slot, values bit-equal to exl3hash's: f16 q · step one rounding,
  int8 q + 128). Sanity: arm H's KLD within ×1.5 of arm I's and top-1 within 2 points, else a decode mismatch: stop.

## Slices

- **Deciding:** `golden-search27c-c` (26 calib docs of the merged fit + search pool, 67,932 scored positions, #122).
- **Check (once, verify pool, never iterated):** `golden-ver3-8k` (156,030 positions).
- Ledger: uses `hash177-search27c-c` (role select) and `hash177-ver3-8k` (role check).

## Statistic and readout

- Per (K, slice): ratio R = mean KLD(H) / mean KLD(I), paired per-document bootstrap 95 % CI (2,000 resamples,
  `mix/paired.py`).
- Equal-quality Δbpw per K: ln R / s, s = the arm-I local slope ln(KLD(2.625) / KLD(4.125)) / 1.5 per bit on the same
  slice (one slope from the two control scores, reported).
- Proxy → KL calibration: R at each K against the settle proxy ratio (w46's side table: A 0xd0474b vs mul1_i8, K2 ×1.010 /
  K3 ×1.008 / K4 ×1.010; this run's stage-2 real-tensor proxy at L16 where available) and #100c's 0.8B (×1.026 K2 / ×1.040 K3 / ×1.016
  K4). The multiplier (ln R / ln proxy) is what prices hash + L18 from its proxy gain.
- **No automatic adopt / reject** (the user decides from the options table). Labels for the table, by the deciding
  slice: **"at par"** if the CI contains 1.00 and R ≤ 1.01 at both K; **"costs"** if the CI lies above 1.00 at a K;
  **"better"** if it lies below. A K where search27c-c and ver3-8k disagree in sign (point estimates beyond ±1 %) is
  reported as "slice-dependent".

## Cost of this stage

Quant: 2 levels × 256 jobs on box2's free cards (~1 h on 2 cards); packs 2 × ~3 min (conv.lock + ram.lock); scores
4 mixes × 2 slices × ~4–5 min. Runtime cost of what it measures: decode −4 to −10 % (stage 1), A8 prefill none.

## Amendment 1 (2026-11-13 ~1:25 PM CST; lead note 12:15 PM CST from the w141 overfit audit): a fresh xcorp part

**Disclosure:** written AFTER the stage-3 rows on golden-search27c-c and golden-ver3-8k of both K were read (w139 saw
the note late: stage-3 scoring began 12:45 PM CST). Those rows stay as reported; this amendment adds the part the two-part
rule (`docs/design/eval-data.md`, 11-08) requires, and it is fixed before the x part is built or scored.
- **x part:** `golden-h177-x` = `eval/slices/pickx.py` 130 pieces of night1105's `xc8k-b2-c2` (ds8
  `/data/q27/goldens/xcorp/`), seed 17701, heavy categories (agentic, tools, long, reasoning) cut to 2,048 scored
  positions, excluding the xcorp documents of every earlier x part on ds8 (`w129/excl/*.json` = 27b–27f-x,
  `heads/slices/golden-hd122-x`, `w129/slices/golden-s129-x`, `w131/slices/golden-s171{a,b,c,d}-x`, both `golden-t-x`);
  built as `search/v3/run171/mkslices171.sh` builds its x parts (CPU, ram.lock shared; model field = the calib golden's).
- **Scoring:** the four existing A8 mixes (unchanged, checks passed), ds8, the same radiance command (s3score-ds8.sh).
- **Readout per K:** R_x (document bootstrap, 2,000, seed 0), and the **pooled two-part ratio** over search27c-c (c) + h177-x
  (x) scored positions, CI by bootstrap within each part. Labels on the pooled ratio as above ("at par": CI contains 1 and
  R ≤ 1.01; "costs" / "better": CI excludes 1 **and both parts' R on that side of 1**, eval-data's same-sign rule;
  otherwise "inconclusive"). ver3-8k stays a report-only check (spent for selection, w141 audit). No automatic adoption.
- Ledger: `hash177-h177-x` (role select) on the xcorp documents; `hash177-search27c-c` as before.
