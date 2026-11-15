# #192 — driver inbox for the #183 search driver (w171)

User 11-15 1:00 AM CST: hand-made allocations judged by the normal rule (decide), and fit-only probes (probe). Built on HEAD
`a1db482c` (driver.py = `fad1cba9`: strata, pool K = 3, cross, two-stage accept #189, re-selection #189-02).
**Nothing is applied to the live run.** Experiment: `docs/experiments/LOG.md` #192, INDEX row #192.

## Files

| file | what |
|---|---|
| `driver-inbox.patch` | `search/v3/driver.py` against `a1db482c`; header = the choices (`git apply -p1`). Only driver.py changes; no registry, config or host change |
| `test_inbox192.py` | 9 tests: decide tentative → confirm, probe fit rows + extra read, decide waits for the scored incumbent, 9 rejects with reasons + bad JSON, restart mid-inbox, `inbox_max`, oldest first + status, identity (recorded #183 replay; live-state copy) |
| `dryrun192.py` | the inbox on a copy of a live run's state (`--inbox FILE ...`, `--conf-set inbox_max=20`); nothing written to the run |
| `tests.txt` | the run on a fresh `git archive a1db482c` + the patch (all five suites) |
| `consensus192.py` | (lead's, 11-15) the consensus allocations that produce the inbox files |

## Inbox file format

`ROOT/inbox/<name>.json` (ROOT = the run's `drv/`; only `*.json` names are read; write elsewhere and `mv` in):

```json
{"size": 2.5,
 "mode": "decide",
 "assign": {"blk.0.ssm_qkv.weight": "q27v3-635bac02-k4-mad", "...": "...", "token_embd.weight": "t5-..."},
 "note": "free text, kept in the state",
 "extra_slices": []}
```

| key | rule |
|---|---|
| `mode` | `decide` or `probe` (else rejected) |
| `size` | a number. `decide`: in CONF `sizes`, not stopped (`STOP-<size>`), feasible. `probe`: any size |
| `assign` | `{weight: option}` in the drv/assign format, **every** menu weight (338 here), each option a real option of its weight (else rejected: unknown weight / option / unassigned weight) |
| bytes | `decide`: menu bytes in the size's window (`driver.window`: bpw in [size − `size_tol`, size]); `probe`: bpw within ±0.02 of `size` (else rejected) |
| `extra_slices` | optional `[{"name": "t183", "golden": "/data/.../golden-t183"}]`: a whole read of a probe on each golden; never a decision, never a fit row |
| `note` | optional |

**Behaviour**
- Oldest file first (mtime, then name). A file that cannot go yet (inbox cap, or a decide size whose incumbent is not scored on the
  active slice) stops the scan: later files wait behind it.
- `decide` → kind `manual`, tag `<tag_prefix>-<size>-manual-<n>` (e.g. `v3s183-2.5-manual-01477`), submitted **before** the driver's own
  moves, judged exactly like a generated candidate: stage 1 on the active slice with the usual early stop (or the audit draw, as
  generated: `audit_frac_by_size`), reference = the worst pool member once full else the incumbent; tentative → confirmation (the
  size's pair) → accept / join / reject; the pool rules are unchanged. Its stage-1 decision counts towards `rotate_after` as any decision.
- `probe` → kind `probe`, tag `<tag_prefix>-<size>-probe-<n>`, whole on the active slice, no early stop: its rows feed the fit like any
  result (labels by slice). No decision, no pool, no incumbent, no registry count, not a frontier point (frontier.json feeds re-seed
  and the final picks).
- `extra_slices` → kind `read`, tag `<inbox_prefix>-<slice>-<probe tag>` (default prefix `ibx`: e.g. `ibx-t183-v3s183-2.25-probe-01484`),
  whole on the golden. Never a decision, never a fit row. The tracker's `trk183-` tags are not used (`track183.py` would count
  these assignments as tracked).
- `CONF inbox_max` (default 6) bounds the inbox's evals outstanding (probes, decides and reads; a probe with one read takes two). A
  file needing more evals than that is rejected. Inbox evals are queue evals: they count in `queue_depth`.
- **Rejects:** the file moves to `inbox/rejected/`, and `driver.log` gets `inbox <name>.json rejected: <reason>`. No repair.
  Bad JSON is rejected only once it is 30 s old (a file still being written waits).
- **Restart:** an `inbox` state event (`file`, `sha`, `mode`, `size`, `tag`, `tags`, `note`) is written after the evals are queued and
  before the file moves to `inbox/done/`. Replay never resubmits a recorded name. A file still in `inbox/` with a recorded name and the
  same bytes is just moved to `done/` (crash between the record and the move). The same name with other bytes is rejected (`name
  already used by a different file`): a silent move would hide a new file.
- **Status:** the line gets `| inbox <n> (<k> outstanding)` (n = files waiting in `inbox/`, k = inbox evals outstanding) only while
  one of them is non-zero. An empty inbox leaves status.txt as today.
- `inbox/`, `inbox/done/`, `inbox/rejected/` are created when the driver starts (no events).

## Switch (lead)

1. **Code** (VM, repo driver.py clean): `git apply --check -p1 search/v3/inbox192/driver-inbox.patch && git apply -p1 search/v3/inbox192/driver-inbox.patch`;
   then `python3 -I search/v3/inbox192/test_inbox192.py` (9 / 9; the identity tests need the pre-patch HEAD: inside the repo `git archive HEAD`
   is used, so run it before committing the patch, or set `RXM_POOL_BASE`). Commit `search/v3/driver.py` after review.
2. **Stop** the driver between decisions: `touch <run>/drv/STOP` (it exits at its next loop), wait for its PID to exit
   (`while kill -0 <pid> 2>/dev/null; do sleep 5; done`), then `rm <run>/drv/STOP`.
3. **Start** with the same command: `python3 search/v3/driver.py <run>/driver.json` (from the patched repo). **No config change**: the
   empty inbox needs none; add `"inbox_max"` only to change the cap.
4. **Check** `drv/driver.log`: `replayed:` with the same counts as before, no traceback; `status.txt` unchanged (no inbox segment).
5. **Submit** the consensus set: `mkdir -p <run>/drv/inbox`, then copy the files in (any order: the driver orders by mtime, ties by name);
   e.g. `cp -p consensus192/*.json <run>/drv/inbox/` (copy into a temp name in the same folder and `mv` if the driver is reading).
   The driver logs `INBOX <file>: decide|probe ...` per submit and `inbox <file> rejected: ...` for any reject.

Rollback: **drain the inbox first** (wait until the status line has no inbox segment), then stop, `git apply -R -p1` (or revert), start.
The old code ignores the `inbox` events but has no rule for kinds `probe` and `read`: it would judge their results as candidate
decisions. A file still in `inbox/` is left there (the old code never reads it).

## Tests (w171, 11-15)

- `test_inbox192.py` 9 / 9 on the fresh `git archive a1db482c` + patch; the existing suites on the same tree: `test_confirm189.py` 15 / 15,
  `test_reselect189.py` 12 / 12, `pool188/test_pool188.py` 13 / 13, `test_picks.py` 18 / 18 (`tests.txt`).
- Identity, inbox empty: the recorded #183 replay (w165 snapshot) 346 events identical to HEAD (275 decisions); a copy of the live run
  (`dryrun192.py`, 3 steps, confirm + pool + re-selection state): identical events, queued evals, log, status, incumbents and pools (the live state moves between runs: 5 events / 1 queued in one run, 3 / 0 in the `tests.txt` run).
- **Real data:** the consensus set (15 files: decide 2 … 6 and probe 2.25 … 5.5, all with `extra_slices` t183 for the probes) on a copy of the
  live state. Default `inbox_max` 6: the six decides 2 … 4.5 are submitted (`v3s183-2-manual-01476` …), 9 files wait. With `inbox_max` 20 on the copy:
  all 14 non-waiting files pass the checks (0 rejects): 8 manual (2 … 6; decides 3, 4, 6 with the audit draw), 6 probes with their t183
  reads; `5.5-probe.json` waits behind the cap (20 outstanding). The live state's `bpw` of each decide is inside its window.

## Choices and deviations (for review)

1. Extra-read tag: `<prefix>-<slice>-<probe tag>` (the slice keeps several extras of one probe unique); default prefix `ibx`, not the
   tracker's `trk183`.
2. Extra reads are not fit rows: the golden is not a registry slice (no parts, strata or slice offset). One line in `add_obs` if you want them.
3. Probes are not frontier points. They are in the fit and in the state, nowhere else.
4. A decide needs its size's incumbent scored on the active slice (so decides wait after a rotation, as a generated candidate would), but
   not the fit: `pred_logr` is written only once a fit exists. Generation still waits for the fit, as before.
5. Same name, other bytes → rejected (the spec said "just moved"; a silent move would discard a new file without a reason).
6. The assignment must cover all menu weights (not in the spec): a partial assignment would make `M.bytes` wrong without an error.
7. The file is checked when it is reached; a file that waits holds back the later ones (oldest first, as asked).
8. Manual candidates draw the audit as generated ones do, so their early stop can be off at the audit rate.
9. Pool rules for manual candidates run through the generic `pool_judge` / `reference` path (the same code as generated candidates); the
   inbox suite does not test a join separately (the w166 / w167 suites cover the join through the same code).

## Not done / limits

- Not applied to the live run; the lead switches (above). No commit.
- Probe rows carry se 0 (no incumbent anchor), as an incumbent's own rows do.
- A read on a golden a host cannot load fails like any eval (a `fail` result, logged); the driver does not check that the golden exists.

## Round-robin and pending cap (#192-02, w174, 11-15)

Patch: `driver-inbox-rr.patch` (driver.py only, against HEAD `30448bb0`; the #192 inbox is already in HEAD). Candidate keys
(**not set anywhere**; the lead decides):

| CONF key | meaning | candidate |
|---|---|---|
| `inbox_share` | 0 < x ≤ 1. With inbox files waiting, the free slots (`queue_depth` − outstanding) are shared so the inbox holds about this share of the outstanding evals (1 = inbox first) | `0.5` |
| `inbox_pending_max` | N ≥ 1. Files beyond the first N waiting in `inbox/` (oldest first) move to `inbox/overflow/` with a driver.log line; a file under 30 s old waits for a later pass | `200` |

`inbox_max` (6) is still the hard cap on inbox evals outstanding. **Both keys absent = the #192 behaviour** (events, queue, log,
status, folders identical to HEAD; `test_inbox192b.py` checks this on a scenario with inbox files and on a pinned live copy).

**Choices (for review):**
1. Slot rule: each free slot goes to the side further below its share (inbox: inbox evals / share; own moves: the rest / (1 − share)).
   A tie goes to the side that did not get the previous slot (round robin; the inbox first). 0.5, depth 26, 100 probes: 13 inbox / 13 own.
2. A side that cannot use a slot (inbox at its cap, a decide waiting for its incumbent, no own candidate) passes it on.
3. Hold: if the oldest inbox file fits the cap but needs more free slots than are free (a probe with an extra read takes two), the slot
   is held: no own move takes it until enough evals answer. Without the hold, a stream of one-eval moves would starve the two-eval file.
4. One free slot: a waiting decide goes before an older one-eval probe, once. The probe is remembered in memory (`inbox_passed`) and goes
   at the next one-slot turn, so decides cannot starve it. A restart resets the memory: one probe may be passed over once more; nothing is
   lost or repeated.
5. Otherwise oldest first. A decide waiting for its incumbent holds back the later inbox files (as in #192) but takes no slot.
6. The share path runs only while inbox files wait. With none, `generate()` fills the queue as before (`room` is None: the same loop bound).
7. Status: with either new key, the segment is `| inbox waiting n, outstanding k / max` (only while non-zero); absent keys keep `| inbox n (k outstanding)`.
8. Restart: nothing new is persisted; the `inbox` events replay as before. Overflow is an atomic file move (no state).
9. Validation at start: `inbox_share` outside (0, 1] and `inbox_pending_max` < 1 raise ValueError.

**Tests:** `test_inbox192b.py` 13 / 13 (share 0.5 of 26 with 100 waiting; keys absent = inbox first; own moves when the inbox is full; share 1;
odd depth; a decide waiting for its incumbent holds no slot; the two-eval hold; decide before probe once; overflow + young file; restart in
the share path; key validation; identity on `scenario192b.py` (HEAD vs tree); identity on a pinned live copy with the 15 consensus files).
Mutants (deliberately broken copies) are each caught by the tests named in `tests-192b.txt`.

**Limits:**
- The own-move side is a stand-in in the share tests (the fake hosts give no fits). The real `generate()` changed only by its `room` cap.
  `dryrun192.py` drops `fit_done` on purpose (generation off), so the live-copy identity does not exercise the real generator.
- `test_inbox192.py::empty_inbox_identity_live` copies the live run separately for each tree. While the live driver moves it can fail
  (one run at about 8:04 UTC: events differed, 8 / 9; the same state pinned once gave identical events). Fix: pin one copy, as `test_inbox192b` does.
- The pinned live copy's coverage depends on the live state at run time (at about 2:05 AM CST one of the 15 files went out: the oldest decide's
  size was not scored on the active slice, and the existing rule holds the later files back).

**Procedure (lead):** `git apply -p1 search/v3/inbox192/driver-inbox-rr.patch`; run `python3 -I search/v3/inbox192/test_inbox192b.py` **before
committing** (its identity tests compare against `git archive HEAD`); commit `search/v3/driver.py`. Turning it on = a restart with
`"inbox_share"` / `"inbox_pending_max"` added to the CONF (the same stop / start as the #192 switch).
