# v3/qad230: teacher data sets t1 + t2 (w230, 11-25)

Two sets of IQ6R-V4-written, bf16-labelled chat data, **public sources only** (xgap-v1 / xcorp-v1 rendered Qwen3.8 documents; no
user sessions), so the sets can be published. Same recipe as `v3/qad228` (raw `/completion` with token-id prompts, thinking-mode
sampling, golden rows = generated tokens via `score_from`), but WHOLE prefixes and a chat-settings variant.

| step | script | what |
|---|---|---|
| 1 | `cuts230.py XGAP TOK OUT` (colo) | cut lists `t1`, `t1-judge`, `t2`, `t2-judge`: `cuts.jsonl` + `sess/<sidx>.npz` (prompt-prefix tokens). `selftest`-free: counts in `data/qad230/cuts-stats.json`. |
| 3 | `gen230.py RENDER OUT --ports P --slots N --target T` | generation on radiance TP2 servers; sessions claimed by mkdir (several processes / lanes share one cut list); `gen.tok`, `closed`; per-cut restartable. |
| 4 | `docs230.py NAME RENDER GEN DOCS --chunks K` | whole-prompt + generation documents (`score_from` = first generated token - 1), K chunks of ~equal tokens, deepest first. |
| 5 | `merge230.py`, `check230.py` | chunk goldens -> one golden (tokens / rows / top.bin concatenated, `kld.json` rewritten, `gen.u8` ones, `corpus.meta.jsonl`), checked against the corpus. |
| driver | `run230.sh` -> `hold.sh` (4 card locks) + `chain230.py` | two TP2 lanes (A cards 0,1 / B cards 2,3) that work-steal: generation, then TP2 bf16 recording per chunk; a chunk that fails on TP2 is recorded on TP4 at the end. |
| tests | `selftest230.py`, `chaintest230.py`, `fake_radiance230.py` | no-GPU tests (mock `/completion`, fake goldens) of the data path and of the whole chain incl. the TP4 fallback. |

Set 1 (agent cuts): xgap `agent-sessions` + xcorp agentic SWE trajectories; cut = start of an assistant turn after a user / tool turn,
prefix 4,096..96,000 tokens (bands 4-16k / 16-32k / 32-64k / 64-96k = 40 / 30 / 20 / 10 %), <= 4 anchors a session + the next assistant
turn after each; thinking-on header iff the doc's system turn has the effort line, else the thinking-off header; `max_new` 32,768 (16,384 off).
Set 2 (chat): one cut a doc (random assistant turn, prefix <= 32k), settings xhigh / low / medium / off round robin, system turn edited as the
template would. Judges (never trained): xgap `holdout` docs + a seeded 10 % of xcorp (by session / doc).

Box2 layout: `/data/q27/teach230/{render-*,out-*,docs,rec,logs,chain.log}`; finals `queue/slices/golden-jt{1,2}` (+ `.jsonl`),
`queue/ext/gold/tset{1,2}` (+ `queue/ext/genmask/tset*.u8`, corpus `queue/ext/teach230/tset*.jsonl`).
Rerun: `cd /data/q27/teach230; setsid nohup code/v3/qad230/run230.sh < /dev/null > /dev/null 2>&1 &` (restartable).
