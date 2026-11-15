# RULES-cooldown — larger search / confirm slices as the search converges (#190)

Written 2026-11-14 11:15 PM CST, before any data on larger slices (user: "up the size of the search corpus to cut
through some of the noise … similar to learning rate cooldown").

## Why
Near the optimum the true effect of a move (≲ 1 % at 3–4.5 bpw) is below the paired SE of a ~180-document slice
(~±1.5 %); 8 of the first 10 tentative accepts failed confirmation. SE falls as 1/√(positions), so doubling the slices
cuts it by ~30 %. Early stopping keeps most of the extra cost on near-null candidates.

## Rule
1. **New slice set s190a–f, 2× s186's size**, built like s186 (RULES-strata, w164's builder, seeded): c parts ~90k scored
   positions (2× the 45k target, same repair to 15.5–19 % non-Latin gen positions, ≥ 6 scripts, no script > 1/3 of N),
   x parts 260 pieces (2× 130) from unused xcorp pieces, same N rules and category stratification. No document or piece
   of any earlier ready / active / confirm / reselect / tracking slice (s17*, s186a–f, t183, t186n, final-x, golden-final750).
   If supply cannot give six 2× slices, build as many full pairs as it can (≥ 2 pairs) and report; never shrink a slice.
2. **Pairs:** (s190a, s190b), (s190c, s190d), (s190e, s190f); b / d / f role confirm. Same RULES-confirm rules (z < −1).
3. **When:** at the next rotation after the set is built and checked (s186c reaching rotate_after), all active sizes
   (2.0–6.0) move to s190a. rotate_after stays 200 decisions. s186d keeps confirming s186c's pending tentatives.
4. **Stopping (the cooldown's end), judged at the lead's check-in ≥ 2 h after the switch:** a size with no confirmed
   accept on the 2× slices since the switch, or whose confirmed accepts sum to < 1 % on t183, is stopped (`STOP-<size>`);
   its cards go to other work (QAD at 2.5 first, by the user's decision). Sizes with confirmed t183 gains continue.
5. Final picks unchanged (golden-final750).

## Amendment 2026-11-14 11:40 PM CST (user OK): one pair, 2× Latin only (supply, w169)
w169's supply check: 15 free non-Latin calib documents, 52 unused non-Latin xcorp pieces, so six (even two pairs of)
2× slices at a 15.5–19 % N share cannot be built. Replaces rule 1's sizes and N floor and rule 2:
1'. **One pair, s190a (search) and s190b (confirm).** Each part keeps about s186's non-Latin amount (c: the 15 free N
    documents split between the two slices, ≥ 6 per c part, distinct within the pair; x: the unused N pieces split evenly,
    distinct) and doubles the Latin part (c ≈ 2× s186's L positions, x ≈ 2× s186's L pieces). The N share then falls to
    whatever results (~8–12 %); ≥ 4 scripts per slice, no script > 1/2 of a slice's N gen. The stratified statistic still
    weights the strata 0.961 / 0.039, so the N share only sets the N stratum's precision. Size check: each part's L gen
    positions ≥ 1.9× the matching s186 part's.
2'. Pair (s190a, s190b), s190b role confirm. Rules 3–5 unchanged. A later cooldown step (4×, or more pairs) needs a new
    N-rich xcorp golden batch first.

## Amendment 2026-11-15 5:00 AM CST (user): rule 4's stops suspended
At the 5 h check 3.5, 4.0, 5.0 and 6.0 qualified for STOP (LOG 11-15 5:00 AM). The user keeps every active size
running: the #193 alternative allocators (the xgboost regressor above all) are on live trial through the inbox at all
sizes, and a stop makes the inbox refuse a size. Rule 4 is not applied until the user says so.
