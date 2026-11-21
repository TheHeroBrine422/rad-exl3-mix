#!/usr/bin/env python3
"""agg.py -- the 11-21 kernel pass's tables (fp16 tile search = qtf2 in libquant) from the raw files here; the first
cut's numbers (f32 search) come from ../ (prod-new.*.jsonl, old-K*/proxy.jsonl)."""
import glob, json, math, os
H = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: [json.loads(l) for l in open(p) if l.startswith("{")] if os.path.exists(p) else []
ti = {str(d["K"]): d for d in rd(f"{H}/stage1-tiles.jsonl")}
qc = {d["K"]: d for d in rd(f"{H}/stage1-qtf2.jsonl")}
qs = {d["K"]: d["old_tiles_per_s"] for d in rd(f"{H}/stage1-qtf2-speed.jsonl")}
print("### tile search (4096 tiles a rate: 3840 Gaussian + 256 tie tiles)\n")
print("| K | fp16 tiles/s | old encoder (qtf2 in torch) tiles/s | fp16 / old | f32 reference tiles/s | fp16 vs old encoder: identical tiles | MSE fp16 / old | fp16 vs f32: identical tiles | MSE fp16 / f32 | exl3i oracle decode diffs | rerun diffs | f32 vs host def. / viterbi3 diffs |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for K, d in ti.items():
    o = qc.get(K, {}); sp = qs.get(K)
    ident = f'{o["identical_tiles"]} / {o["tiles"]}' if "identical_tiles" in o else o.get("old", "—")
    print(f'| {K} | {d["f16_tiles_per_s"]:.0f} | {sp or "—"} | {"%.2fx" % (d["f16_tiles_per_s"] / sp) if sp else "—"} | {d["f32_tiles_per_s"]:.0f} | {ident} | {o.get("mse_ratio", "—")} | {d["f16_identical_tiles"]} / {d["tiles"]} | {d["f16_mse_over_f32"]:.5f} | {d["f16_decode_diff"]} | {d["f16_rerun_diff"]} | {d["host_state_diff"]} / {d.get("viterbi3_tile_diff", "—")} |')
def load(pat, key):
    r = {}
    for f in glob.glob(pat):
        for d in rd(f): r[(d["key"], str(d["K"]))] = d
    return r
new = load(f"{H}/prod-new.*.jsonl", 0); first = load(f"{H}/../prod-new.*.jsonl", 0)
old = load(f"{H}/old-K*/proxy.jsonl", 0); old0 = load(f"{H}/../old-K*/proxy.jsonl", 0)
if new:
    print("\n### production mode, layer 3 (whole tensors), fp16 search\n")
    print("| tensor | n x k | K | new proxy | record via exl3i oracle | ourq proxy | new / ourq - 1 | first cut (f32) proxy | new s | search | gscale | factor | refit | LDLQ feedback | host prep + pack | first cut s | ourq s (this session / 11-20) | new / ourq time |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {}
    for (key, K), d in sorted(new.items(), key=lambda x: (float(x[0][1]), x[0][0])):
        o = old.get((key, K)); o0 = old0.get((key, K)); f = first.get((key, K)); s = d["sec"]; oo = o or o0
        t = tot.setdefault(K, {"n": 0, "v": 0, "o": 0, "o0": 0, "f": 0, "r": [], "c": 0})
        t["n"] += s["quantize"]; t["v"] += s["viterbi"]; t["c"] += 1
        if o: t["o"] += o["s"]
        if o0: t["o0"] += o0["s"]
        if f: t["f"] += f["sec"]["quantize"]
        if oo: t["r"].append(d["proxy_after_refit"] / oo["proxy_err"])
        print(f'| {key.split("layers.")[1]} | {d["n"]}x{d["k"]} | {K} | {d["proxy_after_refit"]:.4e} | {d["proxy_record"]:.4e} | {"%.4e" % oo["proxy_err"] if oo else "—"} | {"%+.2f%%" % (100 * (d["proxy_after_refit"] / oo["proxy_err"] - 1)) if oo else "—"} | {"%.4e" % f["proxy_after_refit"] if f else "—"} | {s["quantize"]:.1f} | {s["viterbi"]:.1f} | {s["gscale"]:.1f} | {s["factor"]:.1f} | {s["refit"]:.1f} | {s["feedback"]:.2f} | {s["prep"] + s["pack"]:.1f} | {"%.1f" % f["sec"]["quantize"] if f else "—"} | {o["s"] if o else "—"} / {o0["s"] if o0 else "—"} | {"%.2fx" % (s["quantize"] / oo["s"]) if oo else "—"} |')
    print("\n| K | new layer s | of which search | first cut (f32) layer s | ourq layer s (this session / 11-20) | new / ourq | geo-mean proxy new / ourq |\n|---|---|---|---|---|---|---|")
    for K, t in sorted(tot.items(), key=lambda x: float(x[0])):
        ob = t["o"] or t["o0"]; g = math.exp(sum(map(math.log, t["r"])) / len(t["r"])) if t["r"] else None
        print(f'| {K} | {t["n"]:.0f} | {t["v"]:.0f} | {t["f"]:.0f} | {"%.0f" % t["o"] if t["o"] else "—"} / {"%.0f" % t["o0"] if t["o0"] else "—"} | {"%.2fx" % (t["n"] / ob) if ob else "—"} | {"%+.2f %%" % (100 * (g - 1)) if g else "—"} |')
    # ---- extrapolations
    keys = sorted({k for k, _ in new}); tiles = sum(new[(k, "4")]["n"] * new[(k, "4")]["k"] for k in keys) / 256
    print(f"\n### extrapolations (layer 3 = {tiles / 1e6:.3f}M tiles, {tiles * 256 / 1e6:.0f}M weights; 64 layers ~ the 27B trunk)\n")
    for K in ("2", "4"):
        t = tot[K]; ob = t["o"] or t["o0"]
        print(f"- 27B trunk, one level at K{K}: new {64 * t['n'] / 60:.0f} min (first cut {64 * t['f'] / 60:.0f} min), ourq plain settings {64 * ob / 60:.0f} min; in-pipeline search rate {tiles / t['v'] / 1e3:.0f}k tiles/s vs {ti[K]['f16_tiles_per_s'] / 1e3:.0f}k on 4096-tile batches")
    et = 15360 * 3 * 2304 * 5120 / 256; tt = 7e9 / 256
    for K in ("2", "3", "4"):
        r = ti[K]["f16_tiles_per_s"]
        print(f"- DS4.1-F at K{K} on 4 cards, search alone ({et / 1e9:.2f}B expert tiles + {tt / 1e6:.0f}M trunk tiles) at the full-batch rate {r / 1e3:.0f}k tiles/s a card: {(et + tt) / r / 4 / 3600:.1f} h (first cut: {(et + tt) / ti[K]['f32_tiles_per_s'] / 4 / 3600 * 1.1:.1f} h)")
