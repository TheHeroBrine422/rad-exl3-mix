# v3/qad214 — V3.1 QAD pilot on the released 2.50 / 3.00 / 4.00 picks (#214, w214)

Rules: `RULES214.md` (committed before any training). Report: `rad-local/workers/out/w214.md`. Raw results: `../../data/qad214/`. Runtime: none.

Scale-only QAD (arm A of #165 / #181 / #196 / #200: the trunk's suh / svh log multipliers on frozen trellis codes, the quantized head frozen;
`--a8` = arm Q) trained on the released EXL3-V3 containers themselves and judged on the served path (radiance `--kld-ref` on the container with the
trained scales folded in) on a fresh calib-verify slice. box2 (`/data/q27/w214`), HIP cards 0-3.

| file | what |
|---|---|
| `rad214.py` | RAD2 container directory reader (entries, schemes, plane offsets) |
| `rad2level214.py` | released .rad -> the student's inputs: trunk level (trellis / suh / svh per linear), quantized R16 head + permutation, GDN a / b (K4 `ssm_ab`) decoded to bf16, `map214.json`; `--check N` decodes N tensors vs the bf16 checkpoint |
| `swap214.py` | packer: copy of the .rad with a checkpoint's multipliers folded into the f16 suh / suh.1 / svh planes (`--identity`, `--verify`) |
| `score214.sh` | the judge: plain radiance `--kld-ref` on one card (w206's release install, ROCm 10.1 libs, A8 as released, host-emb) |
| `cmp214.py` | paired readout: ratio of gen-row KL vs the control, per-document bootstrap CI, documents better / worse, step vs the previous arm |
| `chkt214.sh`, `chkstu214.py` | loader / packer check on the torch path: the student (A8 on; `tscore191.py`) vs radiance's rows on the same gen rows |
| `mkjudge214.py` | the judge slice j214 (verify-pool documents of the calib golden, ledger-filtered); cut with `eval/golden_pick.py` |
| `mkpools214.py` | training pool: seeded subset of calib qad + fit + search documents (`train214`) at ~20 % of the mixed gen rows + the xcorp / trace chunks' counts |
| `env214.sh`, `run214.sh`, `q214.sh`, `serve214.sh`, `reader214.sh` | box2 runners: one training job a card (qad200.py, WSD branches 0.25 / 0.5 / 1 M), a retrying queue, swap + score of a checkpoint, a card's read queue |

## Rerun (box2 `/data/q27/w214`; code = this dir shipped to `code/v3/qad214`, `code/v3/qad{,181,191,200}`, `code/pipeline`)
```
hf download Hero422/qwen3.8-27b-exl3-v3-rad host-emb/qwen3.8-27b-exl3-v3-<2.50|3.00|4.00>bpw.rad --local-dir rel
python code/v3/qad214/rad2level214.py rel/host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad lv/2.50 --model $MODEL --check 6
python code/v3/qad214/mkjudge214.py $CALIB_GOLDEN $CALIB_JSONL pools-v3.json ledger.jsonl 100000 8192 214   # -> golden_pick.py -> slices/golden-j214
# gold/<chunk> from Thor's backup (xc8k-b2-c0, xc8k-b2-c1, tr32k-b2-c0), genmask/<chunk>.u8 from v3/qad181/genmask181.py make, chunks214.txt, pools214.json (mkpools214.py)
code/v3/qad214/score214.sh ctl-2.50 rel/host-emb/qwen3.8-27b-exl3-v3-2.50bpw.rad 0                       # control (unhealed release)
code/v3/qad214/q214.sh 25A 0 2.50 2.5e5,5e5,1e6                                                           # training (30A: card 1, 3.00; 40Q: card 2, 4.00, --a8)
code/v3/qad214/reader214.sh 3 reads214.txt                                                                 # served reads of every finished checkpoint
python code/v3/qad214/cmp214.py slices/golden-j214 scores/ctl-2.50.json.rows scores/25A-ckpt-2e+05.json.rows ...
```
