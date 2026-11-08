# #105 asym27 raw results (w78, 2026-11-08)

The writers asym ρ0 arm on the K2.5 s1 level (rule `v3/yaqa/ASYM27.md`, code `v3/yaqa/asym27/`). The base arm was stopped
by the user after its moments reached layer 24 (no base pair).

| file | what |
|---|---|
| `s1/pair.json`, `pair-meta.json` | s1 → s1a paired KLD on golden-search27: 0.057570 → 0.057957, r 1.0067, z +0.59; no-op gate PASS |
| `s1a.json` | the s1a score summary (per-document rows stay on gpubox: `/data/asym27/s1a.json.rows`) |
| `s1/dn-l1-r0/stats.jsonl` | mkdenoise per group: clip, hsub, F_dev, ‖WF‖/‖W‖, solve route / residual |
| `s1/dn-l1-r0-*.log`, `dn-l1-r0-half.log` | mkdenoise chunks and the N/2 dose run |
| `s1/asym27.log`, `queue.log`, `result.txt` | driver / queue logs (dose line, gate, noop, pair) |
| `s1/gate.log`, `s1/mom-tail.log` | the gate and the tail of the moments log (earlier faulting runs above the final one) |
| `s1/q-noop.log`, `s1/q-asym.log` | ourq logs of the 128 writers (no-op and asym) |
| `census-s1a.txt` | the mix census |
