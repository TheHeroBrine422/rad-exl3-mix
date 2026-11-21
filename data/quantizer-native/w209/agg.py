#!/usr/bin/env python3
"""agg.py -- w209 tables from the raw files in this directory (harness.jsonl, prod-new.*.jsonl, old-K*/proxy.jsonl,
cpuref.jsonl) and the C3 reference ../../cptc/rerun-1120/c3v-*.jsonl."""
import glob, json, math, os, random
H = os.path.dirname(os.path.abspath(__file__))
def rd(p): return [json.loads(l) for l in open(p) if l.startswith("{")]
ref = {}
for f in glob.glob(f"{H}/../../cptc/rerun-1120/c3v-*.jsonl"):
    for d in rd(f):
        for K, v in d.get("per_K", {}).items():
            if "mad" in v: ref[(d["layer"], d["w"], K)] = v["mad"]
cpu = {(d["layer"], d["w"]): d for d in rd(f"{H}/cpuref.jsonl")} if os.path.exists(f"{H}/cpuref.jsonl") else {}
def geo(xs):
    random.seed(0); m = lambda v: math.exp(sum(map(math.log, v)) / len(v))
    bs = sorted(m([random.choice(xs) for _ in xs]) for _ in range(4000))
    return m(xs), bs[100], bs[3899]
for hf, label, usecpu in (("harness-tb1.jsonl", "TB1 (the stored reference's setting)", False), ("harness.jsonl", "TB16 (tensor_check.py main()'s default; the stored reference is TB1)", True)):
  if os.path.exists(f"{H}/{hf}"):
      hs = rd(f"{H}/{hf}"); rat = {"2": [], "3": [], "4": []}
      print(f"\n### harness mode, {label}: new / stored reference mad - 1\n\n" "| tensor | K2 new | K2 ref | Δ | K3 new | K3 ref | Δ | K4 new | K4 ref | Δ | new s (K2+3+4) | new vs CPU rerun at TB16 (K2/3/4) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
      for d in sorted(hs, key=lambda d: (d["w"], d["layer"])):
          row = f'| {d["layer"]}/{d["w"]} '
          for K in "234":
              n = d["per_K"][K]["mad"]; r = ref.get((d["layer"], d["w"], K))
              if r: rat[K].append(n / r)
              row += f'| {n:.6f} | {r if r else "—"} | {"%+.2f%%" % (100 * (n / r - 1)) if r else "—"} '
          c = cpu.get((d["layer"], d["w"])) if usecpu else None
          cs = " / ".join("%+.3f%%" % (100 * (d["per_K"][K]["mad"] / c["per_K"][K]["mad"] - 1)) for K in "234") + f' ({c["total_sec"]:.0f} s)' if c else "—"
          print(row + f'| {d["total_sec"]:.0f} | {cs} |')
      print()
      for K in "234":
          if rat[K]:
              g, lo, hi = geo(rat[K]); print(f"- K{K}: n = {len(rat[K])}, geo-mean new / ref = {100 * (g - 1):+.2f} % [95 % CI {100 * (lo - 1):+.2f} %, {100 * (hi - 1):+.2f} %]; range {100 * (min(rat[K]) - 1):+.1f} % .. {100 * (max(rat[K]) - 1):+.1f} %")

new = {}
for f in glob.glob(f"{H}/prod-new*.jsonl"):
    for d in rd(f): new[(d["key"], str(d["K"]), d["scaled_hessian"])] = d
old = {}
for f in glob.glob(f"{H}/old*-K*/proxy.jsonl"):
    for d in rd(f): old[(d["key"], str(d["K"]), d["scaled_hessian"])] = d
if new:
    print("\n### production mode (whole tensors)\n\n| tensor | n x k | K | new proxy | record via exl3i oracle | ourq proxy | new / ourq - 1 | g_scale new / ourq | new s | viterbi | LDLQ feedback | factor + gscale + refit | host prep + pack | ourq s | new / ourq time |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rat = {}
    for (key, K, sh), d in sorted(new.items(), key=lambda x: (x[0][2], float(x[0][1]), x[0][0])):
        o = old.get((key, K, sh)); s = d["sec"]
        if o: rat.setdefault((K, sh), []).append((d["proxy_after_refit"] / o["proxy_err"], s["quantize"], o["s"]))
        print(f'| {key.split("layers.")[1]}{" (scaled H)" if sh else ""} | {d["n"]}x{d["k"]} | {K} | {d["proxy_after_refit"]:.4e} | {d["proxy_record"]:.4e} | {"%.4e" % o["proxy_err"] if o else "—"} | {"%+.2f%%" % (100 * (d["proxy_after_refit"] / o["proxy_err"] - 1)) if o else "—"} | {d["g_scale"]:.3f} / {"%.3f" % o["g_scale"] if o else "—"} | {s["quantize"]:.1f} | {s["viterbi"]:.1f} | {s["feedback"]:.2f} | {s["factor"] + s["gscale"] + s["refit"]:.1f} | {s["prep"] + s["pack"]:.1f} | {o["s"] if o else "—"} | {"%.1fx" % (s["quantize"] / o["s"]) if o else "—"} |')
    print()
    for (K, sh), v in sorted(rat.items(), key=lambda x: (x[0][1], float(x[0][0]))):
        g = math.exp(sum(math.log(a[0]) for a in v) / len(v))
        print(f"- K{K}{' scaled H' if sh else ''}: n = {len(v)}, geo-mean proxy new / ourq = {100 * (g - 1):+.2f} %; layer time new {sum(a[1] for a in v):.0f} s vs ourq {sum(a[2] for a in v):.0f} s")
