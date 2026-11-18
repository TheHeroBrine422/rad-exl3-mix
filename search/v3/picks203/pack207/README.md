# pack207 on box2 — release containers, text-only (pack207.sh, rel207.sh, pub208.sh) and with vision + drafter (vd203.sh)

`pack207.sh` makes `/data/q27/release-v3/vram/<size>.rad`: the text-only container (trunk pack, radmix over `base-T4.rad`,
head / embedding donors, act lines, census). IQR-V4's release files also carry the bf16 vision tower and the z-lab
DFlash2 drafter, so the release containers must too. `vd203.sh` (w203, 2026-11-18) makes them from the text-only ones:

1. **Base** (`vd203.sh base` → `/data/q27/release-v3/base-T4vd.rad`). The BASE of every container from here on, built by the
   v3/df2 rule: a base with the drafter lines and `--draft-model z-lab/Qwen3.8-27B-DFlash2` makes every radmix over it inherit
   the tower, the drafter and their metadata (`add` from a donor alone would miss the `draft.*` / vision metadata).
   - `rad-convert` of the bf16 checkpoint with `dflash-lines.recipe` (the eight `dflash.*` lines of v3/df2's P3-mx: MXFP4
     linears, u2 draft head; no `dflash.rerank_head` line: the drafter's candidates are rescored against the trunk's own exl3i
     head by `exl3i_rerank`, #180-02, when the line is absent), `--draft-model`, and `--defer` on every block / output /
     embedding / output-norm weight, with the 48 `exl3.act.blk.N.ssm_ab.weight` keys of base-T4 as `--set`.
   - `radmix` over that container with one `add <weight> base-T4.rad` line per base-T4 weight (739): a byte copy.
   - Check (`whash.py`): all 739 base-T4 weights byte-identical; the only extra weights are 333 `v.*` and 77 `dflash.*`.
   - Why not a plain `rad-convert` of the whole model (the recipe base-T4 was made with): its exl3 rules name
     `/data/q27/ladder-mad/…/k1-mad/final` and `/data/q27/mad/ab/absrc-K4`, which are not on box2; the exl3 declare opens
     each source's `config.json` even with `--reuse`. The copy route needs neither.
2. **Per size** (`vd203.sh SIZE...`): `radmix` → `vramvd/<size>.rad.part` over base-T4vd.rad with `release-vd.mix` = the
   release mix of `pack207` where the trunk donor `work/<size>/trunk.rad` (deleted) is `vram/<size>.rad` (same bytes; its
   336 trunk weights). Head, embedding and act lines unchanged. Census as pack207.sh (`--expect-act a8g128 --max-version 4`,
   no `--expect-emb`) → `vramvd/<size>.rad` on PASS.
3. **Verify**: final750 with the rel207.sh command (card `$VCARD`, default 0, ram shared, `--kv-cache-dtype bf16`,
   `--max-model-len 16448`). Rows must be byte-identical to `release-v3/verify/<size>-final750.json.rows` (else
   `mad205/scores/<size>-mad-final750.json.rows`); marks `release-v3/state/<size>.vd-ok` or `.vd-bad`.

Restartable: finished containers and verifies are skipped; `loop203.sh` (in `vramvd/`, not committed) runs the script over the
sixteen sizes as the packs finish them. Locks as pack207 (conv.lock + ram shared around rad-convert / radmix; card + ram shared
around radiance). Stops if `/data` has < 100 GB free.

Smoke (`smoke203/smoke203.sh SIZE`, not committed): one server with the drafter off (`--num-speculative-tokens 0`), one with it
on (default depth), greedy; 8 text prompts (tokens per decode step from the counters) and 2 image prompts; `dcmp.py` /
`tiemargin.py` comparison.

Files on box2: `/data/q27/release-v3/{vd203.sh,base-T4vd.rad,vramvd/,smoke203/}`; the drafter in `/data/q27/models/qwen38-27b-dflash2`.
