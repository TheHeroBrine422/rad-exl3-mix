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
