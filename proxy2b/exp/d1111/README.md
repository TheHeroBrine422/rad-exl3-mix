# d1111 · Input Hessians from broader text: calib + xcorp, calib + raw agentic sessions (0.8B, 4070) (#101)

Question (user, 2026-11-05): instead of self-generated text (#91: helps own text, costs calib), does blending broader
text into the input Hessians help real-world text (xcorp, agentic) without costing calib?
Arms (on the `dygm` stack: calib Σdy² H_out, K3 / K2, seeds 0 / 1): H_in = calib (published-style, base) blended with
H_in over **xcorp** (~2.3M tokens, 2,048-token windows of xcorp v1 docs, stratified) at w 0.25 / 0.5, and with H_in
over the user's **raw sessions** (#91's `mB`, 4.7M tokens) at w 0.25 (`blend.py hin`). Scored on calib search
(g08-search1), **g08-xcorp2** (fresh xcorp, out of calib) and g-agent (own agentic text), paired.
**Rule (written 2026-11-05 ~11:15 AM CST, before data):** a blend is adopted (27B arm next) if calib ≤ 1.005 at both
K **and** g08-xcorp2 ≤ 0.98 at both K (or g-agent ≤ 0.97 for the sessions blend). Otherwise H_in stays calib-only.
Runtime cost: none.

## Amendment 1 (2026-11-05 ~12:20 PM CST, before data): public Qwen3.8-27B agent traces (ukisai)
HF `ukisai/Qwen3.8-27B-multi-turn-agent-sft` (Apache-2.0; Qwen3.8-27B FP16, Terminus-2 harness, nl2bash +
InferredBugs; no overlap with calib / public: 0 hits for its prompt / task names), rendered with the chat template
(`xcorp/ukisai.py`): H_in over 339 sessions / 2.3M tokens, blended into calib H_in at w 0.25 (arm `U0.25`); a fresh
golden `g08-ukisai` from 54 other sessions (0.4M tokens, scored on the assistant turns). All arms (incl. X / B) also
scored on `g08-agent-calib` (calib's own 27B agentic text, search pool).
**Rule:** U0.25 is adopted (27B arm next) if calib ≤ 1.005 at both K and g08-agent-calib or g08-ukisai ≤ 0.97 at both K.
(Caveat: the text is the 27B's, off-policy for the 0.8B.)

## Amendment 2 (2026-11-05 ~9:20 PM CST, before 27B data): the 27B arm
H_in over the same xcorp META (`mX`, 1,148 docs / 1.14M tokens) on one card (`eval/gold27/mkhin27.py`, w33's gates),
blended into the published calib Hessians at w 0.5 (`blend.py hin`). K2.5 swap arms (H27 setup: trunk swapped into
V2 3.00, `golden-search27`, paired), on the bn H_out stack as on the 0.8B (dygm ↔ bn): **xbn** (seed 0) vs **bn**,
**xbns1** vs **bns1** (seed offset 1). Since the 27B quant-noise floor is ~2.5% per pair (H27 amendment 4), two seed
pairs. **Rule:** adopted if both pair ratios < 1 and their geometric mean ≤ 0.98; if the geo mean is within ±2%:
inconclusive at this noise (log it; a third seed pair or a lower-noise golden next); worse beyond 2%: no.
Secondary (not deciding): the same arms on the night1105 xcorp goldens, if their documents are disjoint from `mX`.
