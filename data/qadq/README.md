# data/qadq — results of the global QAD queue (w222)

`results-box2.tsv`, `results-iceman.tsv`: the machines' `QROOT/results.tsv` (one row per served read on j217; columns in `v3/qadq/README.md`), refreshed by `rad-local/bin/qreadout.sh`.
`gate-iceman-train.jsonl`, `gate-box2-25C-first2-train.jsonl`: the iceman gate (2.50, 25C flags, 2 steps) vs box2's w220 25C: bit-identical (`python3 v3/qadq/readout.py --gate A B`).
`digest-box2.txt`, `digest-iceman.txt`: md5 digests of every input on both machines (`v3/qadq/inputs-digest.sh`); the only differences are the `.done` marker and `map214.json`'s `rad` path.
Ledger: training rows are w214's (`qad214`), the judge slice j217 is `qad217-judge`; no new golden slice was cut, nothing else read.
