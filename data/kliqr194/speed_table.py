#!/usr/bin/env python3
"""speed_table.py SPEED_DIR -- w177 (#194-02): one row per served container from bench.py's per-request JSON lines
(SPEED_DIR/<TAG>.jsonl, warm-ups excluded) and rocm-smi snapshots (SPEED_DIR/smi-<TAG>-loaded.json): decode ms per token
(median of the 5 runs per context, client side), decode tok/s, TTFT (median) and prefill tok/s = prompt tokens / TTFT.
Contexts: 0 (short prompt), 4k, 16k. Writes speed.md and speed.json into this file's directory. Reads raw copies only."""
import glob, json, os, statistics, sys

SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
rows, meta = [], {}
for f in sorted(glob.glob(os.path.join(SRC, "*.jsonl"))):
    tag = os.path.basename(f)[:-6]
    if tag.startswith(("smi", "prof")):
        continue
    recs = [json.loads(l) for l in open(f) if l.strip()]
    recs = [r for r in recs if not r.get("warmup")]
    if not recs:
        continue
    cap = None; load_s = None
    p = os.path.join(SRC, f"smi-{tag}-loaded.json")
    if os.path.exists(p):
        d = json.load(open(p))
        load_s = d.get("load_s")
        for card, v in (d.get("loaded") or {}).items():
            if isinstance(v, dict) and "Max Graphics Package Power (W)" in v:
                cap = float(v["Max Graphics Package Power (W)"])
    per = {}
    for ctx in ("0", "4k", "16k"):
        rs = [r for r in recs if r["ctx"] == ctx]
        if not rs:
            continue
        ms = [r["client_ms_per_token"] for r in rs if r.get("client_ms_per_token")]
        tt = [r["client_ttft_ms"] for r in rs if r.get("client_ttft_ms")]
        pt = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens")]
        per[ctx] = {"n": len(rs), "decode_ms_per_token": statistics.median(ms) if ms else None,
                    "decode_tok_s": 1000 / statistics.median(ms) if ms else None,
                    "ttft_ms": statistics.median(tt) if tt else None,
                    "prompt_tokens": statistics.median(pt) if pt else None,
                    "prefill_tok_s": (statistics.median(pt) / (statistics.median(tt) / 1000)) if (pt and tt) else None}
    meta[tag] = {"power_cap_w": cap, "load_s": load_s, "bench_file": os.path.basename(f),
                 "build": recs[0].get("meta", {}).get("build") if isinstance(recs[0].get("meta"), dict) else None}
    rows.append((tag, per))
json.dump({"rows": {t: p for t, p in rows}, "meta": meta}, open(os.path.join(HERE, "speed.json"), "w"), indent=1)
L = ["# Speed of the KL-IQR entries, IQR-V4 neighbours (#194-02, w177)", "",
     "Card: box2 card 0 (rocm-smi GPU[3]); one radiance serve per row; bench.py over the chat API: batch 1, greedy, 512 generated "
     "tokens, ignore_eos, 2 warm-ups + 5 runs per context, medians. Prefill tok/s = prompt tokens / TTFT. Power cap from rocm-smi.", "",
     "| row | decode ms/tok ctx0 | ctx4k | ctx16k | decode tok/s ctx0 | prefill tok/s 4k | prefill tok/s 16k | TTFT 4k (ms) | TTFT 16k (ms) | power cap (W) | load (s) |",
     "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


for tag, per in rows:
    c0, c4, c16 = per.get("0", {}), per.get("4k", {}), per.get("16k", {})
    m = meta[tag]
    L.append(f"| {tag} | {f(c0.get('decode_ms_per_token'))} | {f(c4.get('decode_ms_per_token'))} | {f(c16.get('decode_ms_per_token'))} | "
             f"{f(c0.get('decode_tok_s'), 1)} | {f(c4.get('prefill_tok_s'), 0)} | {f(c16.get('prefill_tok_s'), 0)} | "
             f"{f(c4.get('ttft_ms'), 0)} | {f(c16.get('ttft_ms'), 0)} | {f(m['power_cap_w'], 0)} | {f(m['load_s'], 1)} |")
L += ["",
      "Notes:",
      "- Entries (Nano, Tiny, Small) and IQR-V4 rows run on THEIR build (Perire qc/qwen38-kl-guided-iqr 3eec7eb6); the IQR-V4 rows are the same engine, so the comparison is the containers'. The V3 proxy row runs on our build (rad10-e157b).",
      "- The V3 row is NOT an incumbent and NOT size-matched: `q27v3-635bac02-k4-mad-a8.rad` (w150 pack, uniform K4, A8 act) is 18.25 GB: 12.23 GB trunk, 2.54 GB bf16 token embedding, 2.54 GB bf16 head, 0.92 GB bf16 vision tower (rad-info tensor sums); 5.15 text-plane bpw, 5.43 file bpw. Its decode and prefill are confounded by the bf16 head and embedding. Our 3.0 / 4.0 / 5.0 / 6.0 incumbents could not be built on box2 (levels K4.625-K7.5 exist only as finals). Read it as a V3 kernel reference only.",
      "- All rows: card 0 (rocm-smi GPU[3]), power cap 210 W (rocm-smi Max Graphics Package Power), spec decoding off (IQR-V4's drafter off), --max-num-seqs 8 (the decode176 harness; the KLD reads used 16), --max-model-len 40960, bf16 KV.",
      "- gemm_us (the board metric) is not measured: rad-kbench on their build declines every gemm_nt_q case (no operand recipe for that op), so there is no bench for their GEMM. End-to-end only, as the lead allowed.",
      "- Prefill tok/s = prompt tokens / client TTFT (includes the first decode step); 4k prompts are about 4,146 tokens, 16k about 16,435."]
open(os.path.join(HERE, "speed.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
