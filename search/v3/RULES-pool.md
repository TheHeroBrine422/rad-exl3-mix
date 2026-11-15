# RULES-pool — elite pool, crossover and re-seed in the #183 search driver (#188, w166, 11-14 evening CST)

Written and committed before any pool code is run on data, before any replay is scored, and before any slice is
switched. Builds on RULES-picks (the statistic (b) + guard (c)) and RULES-strata (the stratified statistic, s186*).
Nothing here changes the statistic: the accept / reject decision is the driver's current rule, applied to a
different reference.

**Why.** The driver keeps one incumbent per size. A candidate is judged only against it, and once an incumbent is
replaced its predecessor is lost, even when a later slice or a later candidate would have favoured it. An elite
pool keeps the K best measured assignments per size and proposes children of more than one parent.

## 1. Scope and the K = 1 identity

- CONF `"pool": K` (default 1). **K = 1 is today's driver, exactly:** the same decisions, the same incumbents, the
  same state events, the same random-number stream (no extra draws). The identity is tested on a recorded replay of
  the #183 state (`search/v3/pool188/test_pool188.py`, `replay_identity`).
- Unchanged for every K: the fit rows (every result is a row, as now), the tracking (monitor reads only on a change
  of incumbent; t183 / t186n read the incumbent, never the pool), the size adjustment, the decode term, the audit
  rule, the early-stop block's form (the hosts are unchanged), the rotation rule.

## 2. Members and the incumbent

- `pool[size]` = an ordered list of at most K distinct assignments (by hash). Every member has a whole-slice score on
  the active slice before it counts for a decision. The **incumbent** is one member and is always a member.
- **Reference.** For a size with a full pool (K members), the **worst** member W is the one with the largest
  size-adjusted paired statistic against the incumbent, d_m + off_m (off_m = b (bpw_m − bpw_inc) + the decode term, as
  now; the incumbent has 0); a tie with the incumbent resolves to the non-incumbent. For a pool that is not full, and for
  K = 1, the reference is the incumbent.
- The reference is the early-stop `ref` of every candidate submitted while it is in force (the hosts stop against its
  per-document sums, as now). The candidate is judged at its result against the reference in force at that time.

## 3. Join rule (whole slice only; a stopped candidate is rejected, as now)

Let accept(X, Y) be the driver's accept rule on the active slice: the size-adjusted, capped statistic (RULES-picks (b),
stratified pmean if the slice is stratified) < 0 and every part < 0, and the (c) guard. For a whole-slice candidate c:

- **Pool not full** (fewer than K members): if accept(c, inc), c becomes the incumbent and joins; the old incumbent
  stays a member.
- **Pool full:** if accept(c, W) or accept(c, inc): c joins, replacing W (the worst). If accept(c, inc), c also becomes the
  incumbent.
- An assignment equal (by hash) to a member never joins.
- **K = 1:** W = inc, so the rule is today's: accept → c replaces the incumbent.

## 4. Generation (K > 1)

- Parent: the incumbent with probability 0.6, else uniform over the other members. Mutation kinds (solve, partial,
  pair, block, thompson, explore) run from that parent, inside the same byte window as today.
- **cross** (new kind): parent a (as above), other parent b uniform over the other members. Units: one unit per
  trunk layer (all tensors of the layer, the head as its own unit), and the embedding as a unit. Each unit takes a's
  option or b's, a fair coin per unit. The child is repaired into the byte window by `moves.repair`. A cross child that
  is outside the window or already seen is a failed try (up to 6, as the other kinds).
- Kind shares: the K = 1 shares are unchanged. For K > 1 the share table gets `cross` at CONF `cross_share0` (default
  0.15) and the other shares are renormalised by the same rule as today.
- Generation waits until every member is scored on the active slice and not in flight (the reference and every
  member must be on the same slice for the join check).
- The RNG draws for the parent choice happen only when K > 1.

## 5. Rotation

- At each rotation every member without a score on the new slice is rescored in full (kind "incumbent", as now: logged,
  not decided). No decision is made during rescoring.
- **Re-selection:** when the last member is scored on the new slice, the incumbent is re-chosen once: among the other
  members, the one with the lowest size-adjusted paired statistic against the current incumbent, if it passes accept(·, inc).
  Otherwise the incumbent stays. No new evaluation; the decision uses only the active slice's scores.

## 6. Re-seed at the switch (once)

- CONF `"reseed"`: `{"slice": "s186a", "sizes": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0], "top": 3,
  "source": "drv/frontier.json"}`. Triggered when the named slice becomes active (the first stratified slice).
- For each listed size: the candidate set is the size's current incumbent (its score on the slice) plus the top
  `top` assignments of `frontier.json[size].top` (by score, whole-slice measurements), excluding any with the
  incumbent's hash. Each candidate is scored in full on the slice (kind "reseed", no early stop).
- When all of a size's candidates are scored: rank them by the size-adjusted paired statistic against the incumbent (the
  stratified statistic when the slice is stratified; the incumbent is 0). **Incumbent** := the first-ranked candidate
  that passes accept(·, incumbent) (the old incumbent if none does). **Pool** := the top K of the ranking; if the new
  incumbent is not among them, it replaces the last.
- Sizes not listed start with the pool [incumbent] and fill by joins (section 3).

## 7. Replay of the #183 history (sanity check, not a gate)

- The replay (`search/v3/pool188/replay188.py`) runs on the frozen snapshot `workers/w165/snap/` (state.jsonl 1915
  lines, the snapshot's own sha256): per size and per slice, the whole-slice results only (stopped runs carry only a
  prefix, so they are excluded), in result order, with K = 3, starting from the incumbent scored on that slice.
- Reported: joins, promotions (a member that was not the incumbent at its join and later became incumbent on the same
  slice), the rate per join, and how many of the real K = 1 incumbent changes had a pool member that would have
  anticipated them. Reported with the bootstrap over sizes and slices. Not a gate for the live switch.

## 8. Tests (before any run on the live search)

- `test_pool188.py`: (a) K = 1 identity on a recorded replay of the #183 state: the same decisions, incumbents and events as the HEAD driver;
  (b) K = 3: join when not full, join-and-replace of the worst, join that also replaces the incumbent, rejection of a
  candidate that beats neither; (c) rotation: members rescored, generation gated until all are scored, re-selection
  picks the best accepted member; (d) reseed: ranking, incumbent = first accepted, pool = top K with the incumbent in it,
  frontier source read by hash; (e) cross: the child's bytes are inside [lo, hi] for every returned child, the child's
  units come from one parent or the other, and None is returned when no child fits.
- The shared-code change is delivered as a patch against HEAD (`search/v3/pool188/driver-pool.patch`). The live run
  imports the repo tree; nothing here is applied to it.

## 9. Runtime and what the lead decides

- **Runtime:** neither (search allocation and search cost; no change to the format, kernels or prefill / decode).
- Cost: a K = 3 search evaluates each member once per slice rotation (about 3 × 8 sizes extra rescores, plus the
  re-seed set, about 30 full evals on s186a) and the same number of candidate evals per decision. The lead decides K
  and the switch after the replay; this rule does not start a run.
- Not decided here: whether the pool is kept for the final picks (RULES-picks' finalists are unchanged).
