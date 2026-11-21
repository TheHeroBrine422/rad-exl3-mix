# mik225 — DS4.1-F baseline: mikoskinen's EXL3 2.25 bpw container on bakon's tree (w212b, 11-21)

Container: HF `mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad` rev `57e019d8535f09f311ccc8f9b68c8c23549766be`, joined sha256
`985ea224…cf3d` (verified = published SHA256SUMS; `smoke/answer-config-sha-parts.txt`). Engine: bakon's tree `/data/radiance-ds41` @ 9a963c2 on box2.
Slice `ds41-verify-v1` (`eval/ds41/slice-ds41-verify-v1.manifest.json`, golden on box2 `/data/q27/w212/gold-v1`): 184 docs, 606,116 rows, **150,008 scored (gen.u8 = 1)**.

| file | what |
|---|---|
| `score-mik225.json` | `score41.py` output: both runs + paired run-to-run difference |
| `mik1/`, `mik2/` | two engine starts: `config.yaml`, `report.json`, `report.json.rows` (606116 × 4 f32: kld, lp_ref, lp_cand, top1), `serve.log`, `time.txt` |
| `gold-meta/` | the golden's `kld.json` (document table) and `gen.u8` (mask) — all `score41.py` needs besides the rows |
| `smoke/` | greedy smoke answer + config + sha check + part sizes |

Rerun (box2, `/data/q27/w212/`; scripts in `eval/ds41/w212/`):
```
./join-mik.sh /data/q27/w212/mik225          # after: hf download <repo> --local-dir /data/q27/w212/mik225  (needs ~540 GB peak: parts + joined)
./smoke-mik.sh smoke-mik1                    # serve (port 8011), one greedy chat request, stop
./score-mik.sh mik1 ; ./score-mik.sh mik2    # take locks card0-3; writes runs/TAG/{config.yaml,report.json,report.json.rows,serve.log,time.txt}
python3 -I score41.py GOLD_DIR MANIFEST runs/mik1/report.json runs/mik2/report.json --out score-mik225.json   # GOLD_DIR needs kld.json + gen.u8
```
Config (`mik1/config.yaml`): TP4 cards 0-3, `tp-wire: exact`, `placement: expert_tiered`, `host-pool-mib: 49152`, `ngram-placement: disk`, `kv-cache-dtype: bf16`,
`max-model-len 9216`, `max-num-seqs 1`, `max-num-batched-tokens 2048`, `gpu-headroom-mib 96`, thinking off; even expert split (no `RADIANCE_EXPERT_SPLIT`), no env file.

Headline numbers and caveats: `rad-local/workers/out/w212b.md`.
