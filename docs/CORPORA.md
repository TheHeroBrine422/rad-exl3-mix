# Corpora

What each corpus is used for. Contents and locations of the private ones are not in this repo.

| corpus | size | what | allowed uses |
|---|---|---|---|
| challenge calib (`StillDeadcode/qwen3.8-27b-quant-challenge`) | 10.26M tokens / 4,789 docs | the challenge's calibration split, in fixed pools (fit / search / qad / verify, `eval/pools/`) | per pool (AGENTS.md data discipline) |
| challenge public | 1.69M / 596 docs | held-out split | final reporting only |
| xcorp v1 (`xcorp/`) | 96.1M tokens | built from the challenge corpus's upstream sources at pinned revisions, same categories and rendering; heads of the streams skipped, deduplicated against calib + public | training and statistics (healing, Hessians, QAD); search accept/reject with fresh draws (2026-11-04); never final picks or published numbers |
| private agentic traces (`xcorp/pisess.py`) | 18.5M tokens / 539 docs (v1) | our own coding-agent sessions, rendered as the challenge's agentic traces, secrets redacted | training and statistics; a share of search accept/reject on our own machines only; never published, never sent to donated compute, never final picks |

xcorp v1 by category (tokens): agentic chat 12.1M, code 12.0M, web 10.8M, reasoning 8.4M, chat 8.4M, multilingual 7.5M,
science 7.2M, books 6.6M, math 6.0M, long 4.8M, tools 3.8M, encyclopedic 3.6M, qa 3.6M, legal 1.2M.
Teacher logits for training on xcorp / the private traces are made from the bf16 model by us (no published golden).
