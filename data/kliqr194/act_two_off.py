#!/usr/bin/env python3
"""act_two_off.py -- w178 (#194-02): the KL-IQR-V1 entries with two-pass int8 activations OFF (RADIANCE_ACT_TWO=0, a
load-time override of the container's quantization_config.act_two=1) against the same containers as w177 measured them
(act_two on), on t183 quality and on decode / prefill speed. Reads raw/ only (this directory's raw/ holds copies of the
w177 and w178 outputs); writes act_two_off.md and act_two_off.json next to this file.

Quality: mean KL over the golden's gen rows (kl_mean.py), its 95 % per-document bootstrap (kl_boot.py), and the paired
per-document change off vs on (paired_boot.py). Ratios to IQR-V4 and to our incumbents use w177's curves at the same
text-plane bpw (table.json: the same interpolation as make_table.py).
Speed: bench.py per-request JSON (warm-ups excluded), medians over 5 runs per context, as speed_table.py.
"""
import glob, json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "raw"); A2 = os.path.join(RAW, "act2off")
W177 = json.load(open(os.path.join(HERE, "table.json")))
ENTRY = {"nano": "nano", "tiny": "tiny", "small": "small"}
NEAREST = {"nano": "iq3r-s", "tiny": "iq4r-s", "small": "iq5r"}   # the IQR-V4 row each entry is compared with on speed


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def by_rows(path):
    return {os.path.basename(r["rows_file"]).replace(".json.rows", ""): r for r in jl(path)}


kl_off = by_rows(os.path.join(A2, "kl-t183.jsonl"))
boot_off = by_rows(os.path.join(A2, "boot-t183.jsonl"))
paired = {os.path.basename(r["off_rows"]).replace(".json.rows", ""): r for r in jl(os.path.join(A2, "paired-t183.jsonl"))}
tails = {os.path.basename(r["on_rows"]).replace(".json.rows", "").replace("-t183", ""): r
         for r in jl(os.path.join(A2, "tails-t183.jsonl"))}
IQ183 = sorted((r["bpw"], r["kld"]) for r in jl(os.path.join(RAW, "refs-t183.jsonl")) if r["name"].startswith("iq"))
OURS = sorted((r["bpw"], r["kld_t183"]) for r in jl(os.path.join(RAW, "incumbents-t183.jsonl")))


def ll(points, b):
    pts = sorted(points)
    if b < pts[0][0] or b > pts[-1][0]:
        return None
    import math
    for (b0, k0), (b1, k1) in zip(pts, pts[1:]):
        if b0 <= b <= b1:
            return math.exp(math.log(k0) + (math.log(k1) - math.log(k0)) * (b - b0) / (b1 - b0))


def speed_rows(folder, tag):
    """decode ms/token and prefill tok/s per context from <folder>/<tag>.jsonl (warm-ups excluded) + power cap / load time."""
    p = os.path.join(folder, f"{tag}.jsonl")
    if not os.path.exists(p):
        return None
    recs = [r for r in jl(p) if not r.get("warmup")]
    cap = load_s = None
    s = os.path.join(folder, f"smi-{tag}-loaded.json")
    if os.path.exists(s):
        d = json.load(open(s)); load_s = d.get("load_s")
        for _, v in (d.get("loaded") or {}).items():
            if isinstance(v, dict) and "Max Graphics Package Power (W)" in v:
                cap = float(v["Max Graphics Package Power (W)"])
    out = {"power_cap_w": cap, "load_s": load_s, "n": len(recs)}
    for ctx in ("0", "4k", "16k"):
        rs = [r for r in recs if r["ctx"] == ctx]
        ms = [r["client_ms_per_token"] for r in rs if r.get("client_ms_per_token")]
        tt = [r["client_ttft_ms"] for r in rs if r.get("client_ttft_ms")]
        pt = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens")]
        out[ctx] = {"decode_ms": statistics.median(ms) if ms else None,
                    "ttft_ms": statistics.median(tt) if tt else None,
                    "prefill_tok_s": (statistics.median(pt) / (statistics.median(tt) / 1000)) if (pt and tt) else None,
                    "prompt_tokens": statistics.median(pt) if pt else None}
    return out


W177_SPEED = os.path.join(RAW, "speed")
A2_SPEED = os.path.join(A2, "speed")
res = {"quality": {}, "speed": {}}
for e in ENTRY:
    t = W177["entries"][e]
    on = t["kl_t183"]; onci = t["kl_t183_ci95"]
    off = kl_off[f"{e}-t183-a2off"]["kl_mean_gen"]; offci = boot_off[f"{e}-t183-a2off"]["ci95_doc_boot"]
    if e == "nano":
        ctl = kl_off[f"{e}-t183-ctl"]["kl_mean_gen"]
    else:
        ctl = None
    pr = paired.get(f"{e}-t183-a2off")
    bpw = t["text_plane_bpw"]
    iq = t["iqr_v4_t183_at_text_bpw"]; ours = t["ours_t183_at_text_bpw"]
    res["quality"][e] = {"label": t["label"], "text_plane_bpw": bpw, "kl_on": on, "kl_on_ci95": onci,
                         "kl_off": off, "kl_off_ci95": offci, "kl_on_control_same_day": ctl,
                         "paired_delta_off_vs_on_pct": pr["delta_pct"] if pr else None,
                         "paired_delta_ci95_pct": pr["delta_pct_ci95_doc_boot"] if pr else None,
                         "iqr_v4_at_bpw": iq, "ours_at_bpw": ours,
                         "vs_iqr_on_pct": (on / iq - 1) * 100 if iq else None,
                         "vs_iqr_off_pct": (off / iq - 1) * 100 if iq else None,
                         "vs_ours_on_pct": (on / ours - 1) * 100 if ours else None,
                         "vs_ours_off_pct": (off / ours - 1) * 100 if ours else None}
    # speed
    sp = {"on_w177": speed_rows(W177_SPEED, e), "on_same_day": speed_rows(A2_SPEED, f"{e}-ctl"),
          "off": speed_rows(A2_SPEED, f"{e}-a2off"), "iqr_v4": speed_rows(W177_SPEED, NEAREST[e]),
          "iqr_v4_row": NEAREST[e]}
    res["speed"][e] = sp

out_json = os.path.join(HERE, "act_two_off.json")
json.dump(res, open(out_json, "w"), indent=1)


def pct(x):
    return "n/a" if x is None else f"{x:+.1f} %"


def ci(c):
    return "" if not c else f" [{c[0]:.5f}, {c[1]:.5f}]"


def pci(c):
    return "" if not c else f" [{c[0]:+.1f}, {c[1]:+.1f}]"


def g(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


L = ["# KL-IQR-V1 entries with two-pass activations OFF (#194-02, w178)", "",
     "Each container is read as shipped, with one load-time change: `RADIANCE_ACT_TWO=0` overrides the container's "
     "`quantization_config.act_two=1` (the override is printed at load: `radiance: RADIANCE_ACT_TWO=0 overrides ...`). Nothing "
     "else changes: same files (w177's), build, golden (`golden-t183`), bf16 KV, `--tp 1`, `--max-num-seqs 16`. `act_group` "
     "stays as the container says (Nano 32, Tiny and Small 128). Speed: card 0, 210 W, batch 1, 512 tokens (bench.py).", "",
     "## Headline", ""]
q = res["quality"]
for e in ENTRY:
    d = q[e]
    L.append(f"- **{d['label']}**: act_two off changes t183 mean KL by {pct(d['paired_delta_off_vs_on_pct'])} "
             f"(paired 95 % CI{pci(d['paired_delta_ci95_pct'])} %) against act_two on.")
L.append("")
L.append("## Quality (t183, gen rows; text-plane bpw)")
L.append("")
L.append("| entry | text-plane bpw | act_two on (w177) KL [95 % doc CI] | act_two off KL [95 % doc CI] | off vs on, paired (95 % CI) | on vs IQR-V4 / off vs IQR-V4 (same bpw) | on vs ours / off vs ours (same bpw) |")
L.append("|---|---:|---:|---:|---:|---:|---:|")
for e in ENTRY:
    d = q[e]
    L.append(f"| {d['label']} | {d['text_plane_bpw']:.4f} | {d['kl_on']:.5f}{ci(d['kl_on_ci95'])} | {d['kl_off']:.5f}{ci(d['kl_off_ci95'])} | "
             f"{pct(d['paired_delta_off_vs_on_pct'])}{pci(d['paired_delta_ci95_pct'])} | "
             f"{pct(d['vs_iqr_on_pct'])} / {pct(d['vs_iqr_off_pct'])} (IQR-V4 {d['iqr_v4_at_bpw']:.5f}) | "
             f"{pct(d['vs_ours_on_pct'])} / {pct(d['vs_ours_off_pct'])} (ours {d['ours_at_bpw']:.5f}) |")
L.append("")
L.append("Guards (README rule: tails and top-1 next to the mean; gen rows, 500 document resamples, 95 % intervals on off / on):")
L.append("")
L.append("| entry | KL99 off / on (ratio [CI]) | KL99.9 off / on (ratio [CI]) | top-1 on → off (diff, pp [CI]) |")
L.append("|---|---:|---:|---:|")
for e in ENTRY:
    tl = tails[e]
    L.append(f"| {q[e]['label']} | {tl['kl99_off']:.4f} / {tl['kl99_on']:.4f} ({tl['kl99_ratio']:.3f} [{tl['kl99_ratio_ci95'][0]:.3f}, {tl['kl99_ratio_ci95'][1]:.3f}]) | "
             f"{tl['kl999_off']:.3f} / {tl['kl999_on']:.3f} ({tl['kl999_ratio']:.3f} [{tl['kl999_ratio_ci95'][0]:.3f}, {tl['kl999_ratio_ci95'][1]:.3f}]) | "
             f"{tl['top1_on'] * 100:.2f} → {tl['top1_off'] * 100:.2f} ({tl['top1_diff_pp']:+.3f} [{tl['top1_diff_pp_ci95'][0]:+.3f}, {tl['top1_diff_pp_ci95'][1]:+.3f}]) |")
L.append("")
L.append("Same-day control: Nano with the container's own setting, re-read on 11-15 with the same harness: KL "
         f"{q['nano']['kl_on_control_same_day']:.9f} vs w177's {q['nano']['kl_on']:.9f} (identical to the printed digits: the harness "
         "reproduces w177's reads, so the off-vs-on difference is the setting alone).")
L.append("")
L.append("Sanity gate (lead): no entry's act_two-off KL is above 3× its act_two-on value (ratios: "
         + ", ".join(f"{e} {q[e]['kl_off'] / q[e]['kl_on']:.3f}" for e in ENTRY) + "). The load did not fail.")
L.append("")
L.append("## Speed (card 0, 210 W, batch 1, 512 tokens; medians over 5 runs; prefill = prompt tokens / TTFT)")
L.append("")
L.append("| row | decode ms/tok ctx0 | decode ms/tok ctx16k | prefill tok/s 4k | prefill tok/s 16k | TTFT 4k (ms) | power cap (W) | load (s) |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|")


def line(name, s):
    if not s:
        return f"| {name} | not run | | | | | | |"
    c0, c4, c16 = s["0"], s["4k"], s["16k"]
    return (f"| {name} | {g(c0['decode_ms'])} | {g(c16['decode_ms'])} | {g(c4['prefill_tok_s'], 0)} | "
            f"{g(c16['prefill_tok_s'], 0)} | {g(c4['ttft_ms'], 0)} | {g(s['power_cap_w'], 0)} | {g(s['load_s'], 1)} |")


for e in ENTRY:
    sp = res["speed"][e]
    L.append(line(f"{q[e]['label']}, act_two on (w177)", sp["on_w177"]))
    L.append(line(f"{q[e]['label']}, act_two on (same-day control)", sp["on_same_day"]))
    L.append(line(f"{q[e]['label']}, act_two OFF", sp["off"]))
    L.append(line(f"IQR-V4 {sp['iqr_v4_row']} (w177, nearest size)", sp["iqr_v4"]))
L.append(line("Nano repeat, act_two on (control, run first)", speed_rows(A2_SPEED, "nano-rep-ctl")))
L.append(line("Nano repeat, act_two OFF (run second)", speed_rows(A2_SPEED, "nano-rep-a2off")))
L.append("")
L.append("Share of the gap to IQR-V4 at the nearest size that act_two accounts for (same-day control as 'on'; a share is the part of the gap that switching act_two off would remove):")
L.append("")
L.append("| entry | decode ms ctx0: gap to IQR-V4 | act_two share | prefill tok/s 4k: gap to IQR-V4 | act_two share |")
L.append("|---|---:|---:|---:|---:|")
for e in ENTRY:
    sp = res["speed"][e]
    on = sp["on_same_day"] or sp["on_w177"]; off = sp["off"]; iq = sp["iqr_v4"]
    if not (on and off and iq):
        L.append(f"| {q[e]['label']} | not run | | | |"); continue
    dg = on["0"]["decode_ms"] - iq["0"]["decode_ms"]
    da = (on["0"]["decode_ms"] - off["0"]["decode_ms"]) / dg * 100 if dg else None
    if off["0"]["decode_ms"] > on["0"]["decode_ms"]:
        da_txt = "off is slower (act_two helps decode)"
    else:
        da_txt = f"{da:.0f} %"
    pg = iq["4k"]["prefill_tok_s"] - on["4k"]["prefill_tok_s"]
    pa = (off["4k"]["prefill_tok_s"] - on["4k"]["prefill_tok_s"]) / pg * 100 if pg else None
    L.append(f"| {q[e]['label']} | {g(dg, 2)} ms | {da_txt} | {g(pg, 0)} tok/s | {g(pa, 0)} % |")
L.append("")
L.append("")
L.append("Off vs same-day control, per entry (negative = act_two off is faster):")
L.append("")
L.append("| entry | decode ms ctx0 | decode ms ctx16k | prefill tok/s 4k | prefill tok/s 16k |")
L.append("|---|---:|---:|---:|---:|")
for e in ENTRY:
    sp = res["speed"][e]; c = sp["on_same_day"]; o = sp["off"]
    if not (c and o):
        L.append(f"| {q[e]['label']} | not run | | | |"); continue
    def dp(a, b):
        return "n/a" if not (a and b) else f"{(b / a - 1) * 100:+.1f} %"
    L.append(f"| {q[e]['label']} | {dp(c['0']['decode_ms'], o['0']['decode_ms'])} | {dp(c['16k']['decode_ms'], o['16k']['decode_ms'])} | "
             f"{dp(c['4k']['prefill_tok_s'], o['4k']['prefill_tok_s'])} | {dp(c['16k']['prefill_tok_s'], o['16k']['prefill_tok_s'])} |")
L.append("")
L.append("## Method and caveats")
L.append("")
L.append("- Quality: `kl_mean.py` (gen rows), `kl_boot.py` (per-document 95 % CI, B = 2000, seed 194), `paired_boot.py` (off vs on, "
         "the same resampled documents for both). Rows files on box2 `/data/q27/w178/rows/`, summaries in `raw/act2off/`.")
L.append("- Speed: `speed178.sh` (w177's `speed.sh` harness with an env column). Each row loads its own serve; the same-day "
         "control row is the container as shipped. The IQR-V4 rows are w177's (11-15, 2:47–3:59 PM CST), not re-run.")
L.append("- Not done: a second speed day, the t186n split (not asked), gemm_us (no bench for their GEMM).")
L.append("- Other per-container switches looked at: see the report (`out/w178.md`).")
open(os.path.join(HERE, "act_two_off.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
