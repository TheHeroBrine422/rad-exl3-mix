# v3/qad225 — two-card layer-split ("pipeline") QAD training for 64k–256k context (w225)

The switches live in the trainer itself (`../qad220/qad220.py`); every one is off by default and then the trainer is the old one
(CPU check `testhf225.py` with `Q220_OLD=` the previous copy; GPU check: a 2-step `B` run = I25C's first steps, see the report
`rad-local/workers/out/w225.md`). Raw results: `../../data/qad225/`.

| flag | what |
|---|---|
| `--split-layers K` | one process, two visible cards: the token table and layers 0..K-1 on the first, layers K..63 + final norm + head on the second (`qmodel.Student(split=K)`). The hidden state crosses once a segment forward, its gradient once back. Each layer's carried state (attention KV, GDN / conv state) lives on its own card. Alone it is the **naive split** (one connected autograd graph over both cards). Plain KL training only (no `--prompt-weight` / `--local-mse` / taps / LoRA / block stage) |
| `--pipeline` | with `--split-layers`: the **filled pipeline**. Card A (main thread) runs segment j's first half while card B (a worker thread) runs segment j-1's second half forward + loss + backward; A then runs j-1's first-half backward from the hidden-state gradient B sends back. The step's 24 streams' segments are the micro-batches (bubble ~1/25). Same math as the naive split, bit for bit (each card accumulates its parameters' gradients in segment order). No `--clip-doc` |
| `--state-card-gb G` | keep each stream's carried state on its cards while every card's kept states stay within G GiB, park the rest on the CPU (0 = park all, as before) |
| `--attn-gqa` | segment attention with sdpa `enable_gqa` instead of `repeat_interleave` of the 4 KV heads to 24: the attention transient at 256k drops from ~12 GiB to ~2 GiB a layer (`attn225.py`); forward bitwise, backward rounding-level. Needed for 256k |
| `--depth-test D` | test switch: every document starts after D synthetic positions (zero K / V in the attention layers, positions offset by D) — memory / speed at depth D on any data. Not for training |

Any of `--split-layers` / `--state-card-gb` / `--depth-test` turns on the **buffered KV** (`SegCacheBuf`): each attention layer's carried
K / V sits in one buffer per document (capacity = the document's length), a segment writes its own K / V there in place and attends to
`cat(buffer[:P], k)` (a one-layer transient). The old `SegCache` concatenated, so the carried prefix was held twice (the previous state and the
new one) and the whole KV was copied every segment; the values the attention sees are the same bits.

## Command line (long context, box2, two cards)
```
# a T / B line's flags (qrun.sh passes the data / student paths); the trainer needs both cards visible (ROCR_VISIBLE_DEVICES=A,B)
--split-layers 32 --pipeline --attn-gqa --state-card-gb 6 --cap 262144 --cap-keep /data/q27/queue/slices/jlong-ids.json \
  --lr 3e-4 --compile-scales --compile-ew --a8-int8
```
The queue has no generic two-card training kind yet: `gate225.sh A B LIST` (holds both card locks, runs `qrun.sh` with `CODE_PY` = w225's copy; `NOWIPE=1` resumes)
ran the gates; a queue kind is a follow-up (README of `../qadq`).

## Files
- `testhf225.py` — CPU checks on a tiny real Qwen3.5 stack (8 layers, two "cards" = two CPU device entries): buffered KV, naive split, pipeline,
  frozen first half, `--accum 2` at V = 1, gqa, depth test; `Q220_OLD=/path/qad220.py` also checks the default loop against an older copy.
  Run on box2: `HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= QAD_PIPELINE=/data/q27/w225/code/pipeline /data/q27/venv/bin/python testhf225.py` (no FLA on the path).
- `attn225.py` — one attention layer at depth (1,024 queries after P carried tokens): repeat vs gqa, time and transient memory (dad1 9070 XT).
- `doclens225.py` — document lengths of the training data at a cap and the plan's concurrent depth (how many streams are deep at once).
- `ship225.sh` (VM) — w225's code copy on box2 (`/data/q27/w225/code`). `gate225.sh` + `gates1.txt` (G0–G2), `gates2.txt` (G3 depth), `gates3.txt` (G4 cap 256k).

## Results (report `rad-local/workers/out/w225.md`, raw `../../data/qad225/`)
- One card and split / pipeline match bitwise at 8k (train KL and gnorm, 3–4 steps).
- Pipeline 46–50 s / step vs 72–75 s on one card = 1.5×. The single process is host / GIL-bound; GPU work is 37 / 40 s a card.
- Peak per card with pipeline + gqa: 14.9 / 17.7 GiB at 64k, 17.3 / 19.7 at 128k, 22.0 / 24.3 at 256k.
- 0.25M rows at `--cap 262144`: 51.7 s / step.
- The training data has no document deeper than 32k (`deepdocs225.py`).
