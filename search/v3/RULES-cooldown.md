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
