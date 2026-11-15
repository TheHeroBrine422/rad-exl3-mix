# RULES-slices195 — the slice program for the allocator phase (#195)

Written 2026-11-15 4:40 PM CST, before any data it describes (user: "cycle through ~5 search slices but have more
confirm slices … prep 10–15 slices of 2× with 10 % non-Latin … we need to make more non-Latin stuff no matter what").

1. **Fresh non-Latin supply first:** a new xcorp batch of non-Latin pieces (xcorp v1's multilingual and other
   categories' non-Latin text not in any earlier golden; if those pieces carry too few model-written (gen) rows, the
   model writes the answers itself with `xcorp/selfgen.py`-style generation), bf16 goldens made by us, gen masks,
   script labels. xcorp rules: allowed for search / confirm; never calib verify / public. Target ≥ 400 pieces over
   ≥ 8 scripts.
2. **Slice set:** 15 slices, each 2× s186's size (Latin gen positions ≥ 1.9× the matching s186 part) with **≈ 10 % of
   gen positions non-Latin** (9–12 %), ≥ 5 scripts, no script > 1/3 of the slice's N gen.
   - **5 search slices** (`s195s1`–`s195s5`): Latin may reuse retired search / V2-era documents of the fit + search
     pool, least-used first by V3 decision count (ledger); N from the new batch; the 5 are cycled in order at each
     rotation (a search slice may be used again after the other four).
   - **10 confirm slices** (`s195c1`–`s195c10`): the freshest documents available (calib Latin least-used, unused xcorp
     Latin, the new batch's N), disjoint from every search slice; **each confirm slice is used for one rotation period
     only, then retired.** Pairing: search s195s((k−1) mod 5 + 1) with confirm s195ck, k = 1…10.
   - Never in any of them: t183, t186n, golden-final750, final-x, verify, public, live slices.
3. Until the set exists, rotations use the s190 pairs (RULES-cooldown amendments).
4. t183 stays the fixed tracking slice (consistency across the run). It is a monitor, never a selector: allocator
   adoption or final-pick choices are not made on it (final picks: golden-final750 + ver4 + fresh xcorp part).
