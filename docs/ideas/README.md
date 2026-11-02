# Research ideas (2026-11-02): generated, untested

These files are the output of an LLM research-idea run, published early so others can use them. **Nothing here is a
result.** Ideas were written by Claude Opus 5.5 (extended thinking, "xhigh") agents and then merged, spot-checked
and ranked by a second agent of the same model. Expected gains are guesses with wide error bars; some cited numbers
were checked against the papers (marked in `ranked.md` §0.3), most were not.

What actually happens to an idea is tracked in [`../experiments/INDEX.md`](../experiments/INDEX.md) (numbered
experiments, status, verdict) and [`../experiments/LOG.md`](../experiments/LOG.md).

## Setup

- **Context given to every agent:** our V3 design, the experiment index, log and archives, the roadmap, the V2 method,
  the YAQA proxy study (`proxy2b/YAQA08.md`), and a corpus of recent papers ([`papers.md`](papers.md): 448 papers from
  May–Nov 2026, of which 187 had structured digests, plus reference sheets of 32 older core papers).
- **Ten lenses, one agent each:**

  | file | lens |
  |---|---|
  | [`composable.md`](composable.md) | ideas that fit the current pipeline (per-level library + knapsack, our hardware) |
  | [`open.md`](open.md) | no pipeline or format constraints |
  | [`crossarea.md`](crossarea.md) | borrowed from outside LLM weight quantization |
  | [`lowbpw.md`](lowbpw.md) | the 2.0–2.5 bpw regime only |
  | [`measure.md`](measure.md) | sensitivity measurement, allocation, search, evaluation statistics |
  | [`hout.md`](hout.md) | output-side Hessian information (after the 2026-11-02 H_out result, experiment #33) |
  | [`kernels.md`](kernels.md) | speed: int8 prefill, decode, eval throughput |
  | [`heal.md`](heal.md) | QAD and scale / norm healing |
  | [`codes.md`](codes.md) | the weight format: trellis, codebooks, rotations, fractional K |
  | [`redteam.md`](redteam.md) | adversarial review of the plan and the H_out result |

- **[`ranked.md`](ranked.md):** the critic's merge (~950 raw ideas → 120), code and paper fact-checks, a top 25, the
  cheapest diagnostics, and rejected ideas with reasons. Start here.

## Notes for readers

- The setting is one model (Qwen3.8-27B, a Gated DeltaNet hybrid), EXL3-style trellis codes after a 128-point
  Hadamard, AMD R9700 cards, and a mean-KL-vs-bf16 score; many ideas are specific to that.
- "#n" refers to experiment numbers in `../experiments/INDEX.md`. Machine names are generic ("GPU box": 2× R9700;
  "CPU box": a 36-core server; "4070": the RTX 4070 that runs the 0.8B / 2B proxies).
- Results measured after the run (e.g. #33-16) are not reflected in the lens reports; the critic saw them.
