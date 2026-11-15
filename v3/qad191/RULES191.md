# RULES191 — QAD combination screen, fractional factorial 2^(5-1) at 2.5 bpw (#191)

Written 2026-11-15 12:30 AM CST, before any run (user: "pull some cards away from search and move them to QAD … the
factorial search"). Design from BACKLOG "QAD combination screen (user 11-14)"; base recipe and judging from #181 (w154,
`v3/qad181/`, RULES181.md).

## Model and budget
- **Student:** the #183 search's 2.5 incumbent at the start (its assignment hash, recorded in the run plan; later search
  changes do not move it). Codes frozen. Every arm trains the scales (S, always on, #181 arm A).
- **Budget:** 1M gen rows per arm, its own lr decay to 0, seed fixed per arm, lr 1e-3 for the end-to-end stage (#181's
  winner). Same data stream order across arms with the same D level.

## Factors (two levels each; "−" = #181's recipe)
| | factor | − | + |
|---|---|---|---|
| L | low-rank adapter (rank 16) on every trunk linear, trained with the scales | off | on |
| B | block-wise stage first: per-block output MSE vs bf16, 25 % of the budget, then end-to-end | off | on |
| T | DFlash2-tap hidden-state MSE added to the end-to-end loss (weight fixed in the plan before data) | off | on |
| Q | A8 fake-quant (a8g128, as served) in the student forward | off | on |
| D | data mix | #181 mixed (80 % xcorp, 20 % qad pool) | 50 % qad pool, 50 % xcorp |

16 runs: full 2^4 in L, B, T, Q; **D = L·B·T·Q** (resolution V: main effects and all two-way interactions estimable,
aliased only with three-way terms).

## Data discipline
Training: calib **qad pool** + xcorp (+ our pi sessions, allowed as xcorp). Never fit / search / verify / public. Judge
sets disjoint from every training document: #181's held-out monitor sets (the qad-pool documents #181 held out, and
golden-x181), gen rows only. No verify, no public in this screen; the winner gets one verify read later (lead).
New calib uses go into the ledger.

## Response and analysis
- **Response:** each arm's mean KL vs bf16 on the judge sets (gen rows), on the **student's torch path with A8**
  (tscore181.py), since L cannot be served by radiance yet. Check on the 8 arms with L off that the torch path agrees
  with the served path (radiance, rows) within 1e-5 per the equivalence bar or report the gap; if it does not agree,
  analyse L-off arms on the served path and say so.
- **Effects:** main effects and two-way interactions as contrasts of log KL over the 16 arms; 95 % CIs by a per-document
  bootstrap (documents resampled jointly across all 16 arms, 2,000 replicates). Also every arm vs the S-only arm (#181
  recipe, which is arm (L−,B−,T−,Q−,D−)… note: with D = LBTQ that arm has D+; add the D− S-only arm as a 17th
  reference run).
- **Reading:** a factor helps if its main effect's CI is below 0. Q is expected to join every final arm regardless (we
  ship A8); the screen measures its size. **L adds inference work** (extra low-rank GEMMs in prefill and decode) and a
  format / kernel change: report its KL effect and an estimated prefill / decode cost; adopting L is the user's call.
- **Follow-up (not in this screen):** the best combination re-run at 4M and at 4.0 bpw.

**Runtime:** S, B, T, Q, D: none (scales fold into the format; A8 already shipped). L: prefill + decode (adapter GEMMs).
