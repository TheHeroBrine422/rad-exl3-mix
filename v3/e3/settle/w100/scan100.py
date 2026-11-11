#!/usr/bin/env python3
"""scan100.py OUT.json DIR [DIR ...] -- w100 (#128, ../w100.md): zero / subnormal / non-finite entries in the stored
f16 input / output channel scales (<key>.suh, <key>.svh) of every *.safetensors directly in each DIR or one level
below (ourq q-LNN shards, rxq t-files). Reads the header and those tensors' bytes only (no torch); read-only.
f16 subnormal: 0 < |x| < 2^-14 (6.1e-5). Also per tensor: min |x| / median |x| (flagged below 1e-3, R14's clamp).
One record per DIR (level K from stamp.json / metadata when present), plus the flagged tensors."""
import glob, json, os, struct, sys
import numpy as np

out, dirs = sys.argv[1], sys.argv[2:]
TINY = 2.0 ** -14
res = []
for d in dirs:
    files = sorted(set(glob.glob(f"{d}/*.safetensors") + glob.glob(f"{d}/*/*.safetensors")))
    r = {"dir": d, "files": len(files), "tensors": 0, "entries": 0, "zero": 0, "subnormal": 0, "nonfinite": 0,
         "neg": 0, "min_abs": None, "min_ratio": None, "flag": [], "K": set(), "codebook": set()}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hd = json.loads(fh.read(n)); base = 8 + n
            md = hd.pop("__metadata__", {}) or {}
            if "K" in md: r["K"].add(md["K"])
            if "codebook" in md: r["codebook"].add(md["codebook"])
            for name, v in hd.items():
                if not (name.endswith(".suh") or name.endswith(".svh")) or v["dtype"] != "F16": continue
                a, b = v["data_offsets"]; fh.seek(base + a)
                x = np.frombuffer(fh.read(b - a), dtype=np.float16).astype(np.float32)
                ax = np.abs(x); fin = np.isfinite(x)
                z = int((x == 0).sum()); sn = int(((ax > 0) & (ax < TINY)).sum()); nf = int((~fin).sum())
                med = float(np.median(ax[fin])) if fin.any() else 0.0
                mn = float(ax[fin & (ax > 0)].min()) if (fin & (ax > 0)).any() else 0.0
                ratio = mn / med if med > 0 else 0.0
                r["tensors"] += 1; r["entries"] += x.size; r["zero"] += z; r["subnormal"] += sn; r["nonfinite"] += nf
                r["neg"] += int((x < 0).sum())
                r["min_abs"] = mn if r["min_abs"] is None else min(r["min_abs"], mn)
                r["min_ratio"] = ratio if r["min_ratio"] is None else min(r["min_ratio"], ratio)
                if z or sn or nf or ratio < 1e-3:
                    r["flag"].append({"file": os.path.relpath(f, d), "t": name, "zero": z, "subnormal": sn, "nonfinite": nf,
                                      "min_abs": mn, "median": med, "min_ratio": ratio})
    st = os.path.join(d, "stamp.json")
    if os.path.exists(st):
        try:
            s = json.load(open(st)); r["K"].add(str(s.get("K", s.get("args", {}).get("K", "")))); r["codebook"].add(str(s.get("codebook", s.get("args", {}).get("codebook", ""))))
        except Exception: pass
    r["K"] = sorted(x for x in r["K"] if x); r["codebook"] = sorted(x for x in r["codebook"] if x)
    print(json.dumps({k: v for k, v in r.items() if k != "flag"}), "flagged", len(r["flag"]), flush=True)
    res.append(r)
json.dump(res, open(out + ".part", "w"), indent=1); os.replace(out + ".part", out)
