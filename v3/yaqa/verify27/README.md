# verify27: swj K4 / K2.5 gains on a fresh calib-verify slice (#104 amendment 2, w59, 2026-11-06)

The rule, written before data, is in `v3/yaqa/SWJ27.md`, amendment 2. This file covers the build and the run.

## The confound found first

Amendment 1's K4 pairs (base4 → base4swj 0.7906, bn4 → bn4swj 0.7667 on golden-search27) compare different heads.

| arm | head | embedding | from |
|---|---|---|---|
| base4, bn4 (also base4s1, bn4s1 in q2) | exl3 K4, 0.636 GB | 0.497 GB | **V2 3.00** |
| base4swj, bn4swj | exl3 K5, 0.795 GB | 0.656 GB | V2 4.00 |

Source: `census-*.txt` in `/data/swap27` and `/data/swj27`.

- **Cause:** `swap27k.sh` line 6 sets `V2=$3`; line 7 sources swap27.sh's settings, which set
  `V2=/data/release/...-3.00bpw.rad`.
- **Fix, not applied** (shared code): `swap27k-v2.patch`.
  - Applying it mid-series would put bn4s1 on a different head from base4 / bn4 / base4s1.
  - Better: leave the K4 series as it is (internally consistent), and re-mix with radmix (seconds, reflinks) whenever a
    swap27k K4 arm is paired with a V2 4.00-based arm, as here.
- **Head-matched controls:** `mix-base4v4.rad` / `mix-bn4v4.rad` (`/data/verify27`) = the 336 trunk weights of
  `/data/swap27/mix-base4.rad` / `mix-bn4.rad` cloned into V2 4.00.
  - Census passes: 336 exl3i K4, plus exactly base4swj's head, embedding and plain lines.
  - Same bytes as base4swj (13,702,406,144).
  - Files: `data/census-*.txt`, `data/*.mix`.

## The slice: golden-verify27

- **golden-search27's recipe, reproduced bit for bit** (no bf16 pass: both are cut from the published calib golden):

  ```
  POOLS=pools-v3.json calib_slice.py calib.jsonl 200000 8192 27 search
  golden_pick.py golden/calib calib.jsonl golden-search27 golden-search27.jsonl <docs>
  ```

- **golden-verify27:**
  - Command: `EXCLUDE_GOLDEN=excl POOLS=pools-v3.json calib_slice.py calib.jsonl 250000 8192 20261107 verify`, then
    golden_pick.py the same way.
  - Size: **71 docs, 171,876 tokens, 171,805 rows, 127,261 scored positions** (gen.u8).
  - Frozen list: `golden-verify27.docs` (v27.sh refuses to score if the slice comes out different). Header:
    `golden-verify27/kld.json`.
- **250,000 tokens asked:** the smallest ask on a 25k grid that reaches ≥ 120k scored positions. The candidates are
  long documents, so ~100 docs is out of reach at this size (150k → 51 docs / 85k positions, 200k → 60 / 106k,
  225k → 66 / 116k).
- **Excluded: `excl-kld.json`, 609 of the 856 verify docs.** Each doc lists why.
  - ver3 (V3 monitor) and ver4 (V3 final), with their long traces 2300 and 4723;
  - the V4 reserve (`eval/pools/ver-2026-11-01.json`: not before V4);
  - @yaqa and @headsens (trained on);
  - g08-verify1 / g08-verify2 (0.8B verify goldens that already read swj, #93).
- **What that leaves.** There are no untouched verify docs outside the V4 reserve. **All 71 docs were read before,
  only by V1 / V2-line goldens:** calBbig 42, ver2 / ver2full 38, ver1 14, calB 11, long2 2.
  - None was used to judge a V3 setting, and none was trained on, so a paired V3 swj contrast isn't biased by them.
  - The two long-agentic pieces are the first 8k of agent-trace rich (905) and express (2675), read by calB / calBbig
    and long2.
  - The pool mix, from calib_slice's stderr: agentic 24,576, books 7,296, chat 10,289, code 10,062, encyclopedic 5,662,
    legal 1,620, long 8,192, long-agentic 16,384, math 7,431, multilingual 15,272, qa 8,590, reasoning 16,384,
    science 19,270, tools 5,369, web 15,479 tokens.
  - Category shares follow the remaining candidates, as calib_slice does. agentic is short (3 docs left).
- **Overlap checks:** golden-search27 0 docs; excl 0 docs; every doc is in the verify pool.

## Run (gpubox card 0, q2 insert)

- `/data/q1106/insert-40-verify27.sh` runs `/data/verify27/tools/v27.sh`. It writes one line to `q.log`, either the
  verdict or `verify27 FAIL exit N`; the existing watch rule `fail` catches FAIL.
- v27.sh stages:

| stage | does | status |
|---|---|---|
| golden | builds golden-verify27 (CPU) | done in the dry run |
| mix | builds the head-matched mixes (CPU) | done in the dry run |
| score | 12 scores: base4v4, base4swj, bn4v4, bn4swj (gating), base4, bn4, base, baseswj, s1, s1swj on golden-verify27; base4v4, bn4v4 on golden-search27 | runs in the insert |
| report | `v27report.py` → `report.txt` / `report.json` / `result.txt` | runs in the insert |

- **Scoring settings:** exactly swap27.sh's `score` (radiance 1.16.0 + plugins home `/data/swap27/home`, bf16 KV,
  max-model-len 16384, card0.lock then ram.lock shared). A score is trusted only at rc 0 with the full rows file
  (`.part` until then).
- **Restartable.** q2 touches `.done` whatever the exit, so after a FAIL rerun by hand:
  `bash /data/q1106/insert-40-verify27.sh`. It resumes per arm; stages can also run one at a time
  (`v27.sh score report`).
- **Report.**
  - Per arm: mean KLD, p99 KLD and top-1 on the scored rows. radiance 1.16 rows are `{kl, nll_q, nll_p, top1}`;
    checked against the json's ppl and top-1.
  - Per pair: ratio, 95% CI and z (paired.py's bootstrap), p99 ratio, top-1 difference; then the verdict.
  - `v27report.py --selftest` reproduces amendment 1's base4 → base4swj line exactly (0.79056, z −5.05).
- **Dry run (no card), 2026-11-07 00:29 UTC:** all stages with `fakeradiance.sh` as `bin/radiance`, in a scratch
  directory with its own lock files. All 12 outputs resolved to the right mixes and the report rendered (fake
  numbers). A rerun skipped every arm. The scratch was deleted afterwards.
- **Timing:**
  - Inserted 6:29 PM CST. q2 runs inserts after base4s1 scores, expected ~7:00–7:15 PM CST (insert-40 before
    insert-50-cd27).
  - Card 0 ~35 min (12 × ~140–160 s), so the result is due ~7:35–7:50 PM CST. Below 30 min of silence, the q2out
    freshness rule stays happy: each step logs to `/data/q1106/insert.log`.
- **Disk:** golden 0.7 GB; the mixes are reflinks. Delete `/data/verify27/mix-*v4.rad` once the result is logged and no
  further pairing needs them.

## For the lead (not done: shared files)

- **Ledger** (golden-search27 is not in it either):
  1. Add to `eval/pools/ledger-roles.json`:
     `"golden-verify27": {"role": "check", "line": "v3", "decisions": 1, "note": "#104 amendment 2: swj K4 / K2.5 confirmation (w59)"}`.
  2. Run `eval/pools/ledger_add.py eval/pools/ledger-roles.json eval/pools/pools-v3.json eval/pools/ledger.jsonl v3/yaqa/verify27/golden-verify27`.
- **Raw results after the run:** copy `/data/verify27/{result.txt,report.txt,report.json}`, `verify/*.json` and
  `search/*.json` to `data/swj27/verify27/`.
- **Amendment 1's K4 line in LOG / INDEX** (0.7906 / 0.7667, "swj adopted at every level") includes the head change.
  The search-pool head-matched pairs in this run's report give the corrected search numbers.

## Runtime cost

None. Only the svh vectors change, and the controls are measurement-only.
