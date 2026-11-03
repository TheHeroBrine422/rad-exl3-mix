# w01-windows: what the YAQA collectors' windows cover (ranked.md A4)

**Question.** `collect.py` (the 27B diagonal-H_out collector, and the 0.8B runs of #33-17/#33-18) uses only qad
documents of ≥ 2048 tokens, cut into 2048-token windows and drawn in a fixed shuffle. Does that mix match what the
challenge scores (gen rows of the calib corpus, whose distribution the private set shares)? Should short documents be
packed in, or windows reweighted?

**Method.** CPU only: `windows.py` on the calib golden's `kld.json` + `gen.u8` and `pools-v3.json`. It reproduces
collect.py's window list and shuffle exactly, and per source category gives the share of scored (gen) rows in:
- the corpus;
- the qad pool;
- all windows;
- the 1,024 drawn windows;
- the drawn loss positions (what today's all-position loss trains on).

Run: `run.sh` (seconds, on the 4070's CPU). Outputs are in `out/`.

## Numbers (L = 2048, SEED 0; `out/windows-L2048-n1024.txt`)

| | value |
|---|---|
| qad windows of 2048 tokens that exist | **1,350**: amendment 10's "continue to 2,048, then 4,096" is impossible (see below) |
| qad docs never used (< 2048 tokens) | **2,353 of 2,799**, holding **38.0%** of the qad pool's gen rows |
| gen share of the drawn loss positions | 74.2% (masked loss, A1, keeps these); **105 of 1,024 windows have no gen row** |
| total variation vs corpus gen mix: qad pool / drawn gen / drawn loss positions | 0.137 / **0.261** / **0.291** |
| corpus gen mass in categories drawn at < 0.5× | **22.5%** |
| same at L = 1024 (first 1,024 of 3,382 windows) | TV 0.210 / 0.235; 20.7% of qad gen never used |
| what-if: qad docs ≥ 256 tokens as their own sequences (all 3,391 seqs) | TV **0.131** / 0.129; 90% of qad gen used |
| what-if: ≥ 512 tokens (2,727 seqs) | TV 0.123 / 0.140; 84% used |

Drawn gen share relative to the corpus gen share, by category (≥ 1% of corpus gen; full table in `out/`):
- **Under-represented:** the categories made of short documents. stackexchange 0.20×, stackv2-edu 0.41×, fineweb-edu
  0.69×, wikimedia 0.74×, nemotron-swe 0.62×, doab 0.61×. smoltalk2/everyday and smoltalk2/systemchats are at 0.13×;
  glaive-fc, when2call, some fineweb2 languages and wikipedia/th are at 0× (all their qad docs are < 2048).
- **Over-represented:** long single documents. arxiv 2.39×, gutenberg 2.32×, openthoughts3 2.04×, peS2o 1.84×,
  pg19 1.66×.
- **Absent from the qad pool altogether:** pg19-long (2.6% of corpus gen) and arxiv-long (1.5%). This is a pool
  property (those docs went to other pools); neither packing nor reweighting can fix it.
- **All-position loss:** it adds the prompts of agentic and chat documents. Drawn loss positions vs corpus gen share:
  swe-rebench 3.3×, swe-smith 3.3×, wildchat 2.65×, smoltalk2/longalign **70×** (2.8% of all positions, 2.3% of
  them scored). That is A1's case.

## Findings
1. **Hard cap: 1,350 windows.**
   - `collect.py` indexes `wins[s]` with no bound, so a run asked for more sequences dies with an IndexError at
     sequence 1,350, about 11 h into a 27B night.
   - Amendment 10 (YAQA08.md: "otherwise continue to 2,048, then 4,096") and `hdiag27.sh` (SNAPS
     256,512,1024,**2048**,N) both assume more windows than exist.
   - `../w01-failloud/collect-check.patch` makes this fail at start-up.
   - The real ceiling for "more data" on the 27B is 1,350 windows unless short documents join (finding 3).
2. **The critic's rule fires.** Several categories with ≥ 1% of corpus gen are off by more than 2×
   (stackv2-edu, stackexchange, openthoughts3, arxiv, gutenberg; plus the absent -long ones). TV is about twice the
   qad pool's own (0.26–0.29 vs 0.14). The bias is systematic: short-document categories (web Q&A, code snippets,
   function calling, chat) are missing, and long single texts are over-weighted.
3. **Short documents as their own sequences, not packing and not reweighting.**
   - With sampled labels, the cross-position terms of G_b G_bᵀ have zero mean (E over the model's own label is 0
     given the context). So Sketch B's diagonal is additive over positions in expectation, and variable-length
     sequences are unbiased.
   - That removes the need for packing with GDN state resets. Each short document is simply one sequence.
   - Reweighting cannot recover the zero-coverage categories, and it inflates noise.
   - Taking every qad document ≥ 256 tokens brings TV down to the pool's own floor (0.131), raises the usable gen
     rows from 2.05M to 3.45M, and gives 3,391 sequences instead of 1,350.
   - Cost: more, shorter sequences. Per-sequence overhead dominates for short ones, so expect roughly 1.5–2× the time
     per gen token on the 27B (not measured).
4. **Interaction with A1:** the masked loss removes the prompt-position distortion (positions TV 0.291 → gen TV
   0.261), but not the document-length bias. They are independent fixes.

## Proposed follow-up (not built; the lead decides)
- A collect.py option `YAQA_SHORT=MIN`: qad docs of MIN..L-1 tokens join the window list as single sequences.
  Variable length; the shuffle stays; snapshots count sequences.
- A 0.8B test like `../w01-genmask`: 1K sequences from the extended list vs c18 (same seed; the lists differ, so not
  label-paired), 3 seeds K3/K2.
- Draft rule: adopt if the K3 3-seed mean ratio ≤ 0.98 with every seed better, and K2 ≤ 1.00.
- GPU ~1 h.
- It matters most if the 27B halves rule (amendment 10) fails at 1,024 windows: then this is the only way to add data.

## What it changes
- Now, no GPU needed:
  - fix amendment 10 and hdiag27.sh's 2048 snapshot (the pool ends at 1,350);
  - merge the window-count guard (A8 patch).
- The ranking of A4 rises from "0 to −1.5%, P 0.3" to a likely real but small effect: the composition is measurably
  off, and it also caps the dose.
- Cite: ranked.md A4, A5 (dose), INDEX #33-13 / #33-17 / #33-18.

## Runtime cost (user rule, 2026-11-03)
n/a (analysis of collection windows).
