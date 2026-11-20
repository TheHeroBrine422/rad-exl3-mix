# Claude 5.5 / Fable 5.2 on Artificial Analysis (pasted by the user 2026-11-20)

Source (user's reference link): https://artificialanalysis.ai/models/comparisons?compare=claude-fable-5-2%2Cclaude-opus-5-5%2Cclaude-sonnet-5-5%2Cclaude-haiku-5-5
Supersedes `model-costs-aa-2026-11-15.md` for the settings below. Cost = AA's USD per Intelligence-Index task (short tasks: Haiku's
5× long-context price is NOT in it — our Haiku workers measured 4.2–4.7× that, so a long Haiku worker ≈ $1.1 / task-equivalent).

| model (reasoning) | index | Engineering | Terminal-Bench 4.0 | FrontierCode 2.0 | TB-Science | $/task | min/task |
|---|---|---|---|---|---|---|---|
| Fable 5.2 xhigh | 62 | 70 | 61.6 | 47.6 | 67.6 | 14.83 | 24.5 |
| Fable 5.2 high | 62 | 70 | 59.6 | 49.1 | 70.0 | 9.08 | 15.1 |
| Opus 5.5 xhigh | 57 | 63 | 59.6 | 43.8 | 61.9 | 6.05 | 16.0 |
| Fable 5.2 medium | 56 | 65 | 55.1 | 41.3 | 60.5 | 3.80 | 6.5 |
| Fable 5.2 low | 54 | 62 | 54.5 | 38.4 | 53.8 | 3.10 | 5.5 |
| Opus 5.5 high | 52 | 58 | 56.6 | 35.9 | 49.0 | 3.46 | 9.3 |
| Sonnet 5.5 xhigh | 50 | 54 | 57.1 | 35.6 | 52.4 | 4.09 | 14.4 |
| Opus 5.5 medium | 48 | 54 | 52.5 | 33.1 | 43.3 | 2.45 | 6.8 |
| Sonnet 5.5 high | 42 | 46 | 43.9 | 25.7 | 34.3 | 1.64 | 7.1 |
| Opus 5.5 low | 35 | 43 | 31.3 | 15.8 | 24.3 | 0.98 | 3.1 |
| Haiku 5.5 xhigh | 35 | 41 | 29.3 | 29.9 | 18.1 | 0.25 | 9.5 |
| Sonnet 5.5 medium | 31 | 37 | 29.8 | 11.0 | 16.2 | 0.78 | 3.7 |
| Haiku 5.5 high | 30 | 37 | 21.7 | 23.2 | 10.0 | 0.16 | 6.5 |
| Sonnet 5.5 low | 26 | 33 | 20.7 | 7.1 | 10.5 | 0.51 | 2.6 |
| Haiku 5.5 medium | 25 | 32 | 15.2 | 16.1 | 7.1 | 0.09 | 4.0 |

Reading for our work (agentic shell + code = Terminal-Bench, FrontierCode, TB-Science):
- **Haiku xhigh** is the code bargain (FrontierCode 29.9 > Sonnet high 25.7 at 1/6.5 the price) but weak at long agentic shell work
  (TB 29.3) — short, well-specified coding tasks only.
- **Sonnet high** ($1.64, TB 43.9) is the cheap long build-and-test worker (no long-context surcharge).
- **Opus medium** ($2.45, TB 52.5) and **Fable low** ($3.10, TB 54.5, FC 38.4, TB-Science 53.8) are close; Fable low is the better
  buy for hard code / research, only +27 % cost. Opus low, Sonnet medium / low, Sonnet xhigh are off the frontier (dominated).
- Fable high ($9.08) only for the rare hardest runs.
