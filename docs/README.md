# Docs

| path | what | state |
|---|---|---|
| [`ROADMAP.md`](ROADMAP.md) | plans by release (V3, V3 extras, V4 = QAD and after), research method and infrastructure, data | current |
| [`design/v2.md`](design/v2.md) | how EXL3-V2 was made: options, proxy error, sensitivity model, solver, search, data discipline | final (V2 released) |
| [`design/v3.md`](design/v3.md) | the EXL3-V3 design: int8 / int4 compute, ladder and codebooks, quantizer stack, production pipeline, embedding, head, solver, search, order of work | current (2026-11-06), open decisions marked **[decide]** |
| [`design/eval-data.md`](design/eval-data.md) | V3's evaluation and statistics data: roles of calib / xcorp / own traces, goldens, self-generated text, xcorp as a statistics source | current (2026-11-06) |
| [`design/kernel-opts.md`](design/kernel-opts.md) | V3 kernel work: prefill (A8, A4) and decode optimizations from IQR-V4's kernels, the hash-codebook (#100) trade-off, branch status, measurements | current (2026-11-06) |
| [`design/lsa.md`](design/lsa.md) | D-LSA (latent sparse attention) in radiance: facts, interface contract, the two work packages, gate | draft brief; both packages built on branches, not integrated (2026-11-06) |
| [`experiments/`](experiments/) | every experiment and measured piece of work: `INDEX.md` (numbered, status), `LOG.md` (append-only entries), `BACKLOG.md`; conventions in its README | current |
| [`research/`](research/) | literature briefs: PTQ state of the art, mix allocation, RCO, NVIDIA Model Optimizer overlap (`modelopt.md`, 2026-11-03), IQR-V4 (`iqr-v4.md`, 2026-11-04) | snapshots (2026-10-29/30), each with a status header saying what became of its recommendations |
| [`ideas/`](ideas/) | LLM-generated research-idea passes (lenses + a ranked merge), one folder per pass, with the paper lists | **untested**; status of anything tried is in `experiments/` |

Related repositories (2026-11-06): `rad-exl3-quantizer` (the production quantizer pipeline, package `rxq`; its
`config/v3.toml` pins the V3 freeze settings with their evidence from `experiments/`) and `rad-quant-at-home` (the
job system for donated compute that runs the ladder's per-tensor jobs). This repository stays the research record.
