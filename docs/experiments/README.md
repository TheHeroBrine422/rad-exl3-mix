# Experiments: how this folder works

Every experiment we run, plan or drop is tracked here, whether it worked or not, from 2026-11-02 on. Three files, three jobs:

| file | job | how it changes |
|---|---|---|
| `INDEX.md` | one row per experiment: number, title, area, status, last date, verdict | edited in place; **the only place status lives** |
| `LOG.md` | dated entries, newest first, one template | **append-only**: old entries are never rewritten, only followed by newer ones |
| `BACKLOG.md` | ideas and plans not yet started | items move to `INDEX.md` (with a number) when they start |
| `LOG-legacy.md` | the archive's log, dead ends and topic sections rewritten into the template (166 entries, numbers checked verbatim) | frozen |
| `archive/` | the trackers before 2026-11-02, verbatim | frozen |

## What gets a number

Anything that ends in a measurement or a pass/fail check: experiments, and also engineering work such as kernel
optimization passes (speedup + identity gate), ports (KLD parity), new tools or pipelines with a validation, data
builds with a check. Chores with no result to look up (restarts, uploads, doc edits, cleanup) stay out: they belong
in TODO and the git log. If a chore turns up something worth knowing, log that finding as a `#—` entry.

## Numbers

- Every experiment gets the next free number when it is planned or started: `#57`. Numbers are never reused or
  renumbered, even when an experiment is dropped.
- A multi-part experiment (several gates, arms or reads under one question) is a parent `#58` with parts `#58-01`,
  `#58-02`, ... in the order they start. A simple experiment can grow parts later (`#57` → `#57-01`).
- Events and decisions that are not experiments (a competitor publishes, the user sets a rule) are logged with `#—`.
- Experiments before 2026-11-02 were numbered in bulk (#1–#73; V2 wins are #63–#73); their entries are in `LOG-legacy.md`, the originals in `archive/`.
  Old names (E1b, D3, B11, gate 2d, ...) are listed as aliases in the index.

## Status words

`idea` (in BACKLOG only) · `planned` (pass rule written, not started) · `running` · `blocked` (say on what) ·
`done` (answered, positive or as expected) · `null` (no measurable effect) · `negative` (measured worse) ·
`dropped` (stopped without an answer: say why) · `superseded` (by #n).

## Log entry template

```
### 2026-11-02 06:00 CST · #58-08 · YAQA Sketch B on 0.8B: 1K read negative
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/yaqaB.py` @ abc1234
- **Question:** ...
- **Pass rule** (written 2026-11-02 04:10, before data): ...
- **Setup:** model, data slice and pool, arms, seeds
- **Result:** numbers, paired with SE / z or a bootstrap interval
- **Decision / next:** ...
- **Files:** results, logs
```

Short entries are fine: drop the lines that do not apply, but keep the header line exactly in this form
(date, time with zone, number, title). Times in CST (UTC−6). Commits as short hashes.

## Rules (from `/workspace/AGENTS.md`, repeated because every entry depends on them)

- **Pass rules are written before the run**, in the entry that starts it; changes are recorded before any data is
  used ("amendment 1, written ...").
- Tuning only on calib pools (fit / search / qad / verify); public only to report; say which pool a slice came from.
- Report paired comparisons with uncertainty; differences under ~1.5% on 100k-position slices are noise.
- **Two questions, two criteria** (user, 2026-11-02): a pass rule says separately (1) **is it real** — judged against
  measured noise with the least noisy design available (paired by document; by seed or with fixed codes where the
  arms can share rounding noise; seed replicates otherwise: 0.8B per-seed SD ~1% K3 / ~2% K2, #33-16) — and (2) **is
  it worth it** — a threshold set from the idea's cost (quant time, kernel work, format, complexity). A small effect
  that is consistent goes to a tighter confirmation, not straight to "fail"; a real but tiny effect is judged on
  cost. No fixed "≥ 2%" bars copied from idea lists.
- bf16 KV cache for every KLD.
- **Every experiment's raw results go into the repo** (`data/<topic>/`, with a README mapping files to numbers) in the
  same commit as its LOG entry, and the entry's Files line points there. Results living only on a GPU box are lost
  when it is wiped.
- A dropped or null experiment says what was tried, the evidence, and what we do instead, so nobody repeats it.
