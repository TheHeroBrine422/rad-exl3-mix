#!/usr/bin/env python3
"""c2hash115.py LV OUT.json -- amendment 1's c2 (../w115.md): per-tensor sha256 of w114's K2.5 / K4 level t-files of
layers 10, 11, 39 (gpubox, CPU, read-only), so box2's N seed-0 units can be compared without moving the files.
OUT = {"<K>": {key: {"seed_offset": S, "stamp": settings stamp, "t": {tensor: [dtype, shape, sha256]}}}}."""
import glob, hashlib, json, os, struct, sys

LV, OUT = sys.argv[1:3]
STAMP = "4f8319840e59c5d0"
LAYERS = {"10", "11", "39"}


def tensors(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
        meta = hdr.pop("__metadata__", {}); out = {}
        for k, v in sorted(hdr.items()):
            a, b = v["data_offsets"]; f.seek(base + a)
            out[k] = [v["dtype"], v["shape"], hashlib.sha256(f.read(b - a)).hexdigest()]
    return out, meta


res = {}
for K in ("2.5", "4"):
    lv = f"{LV}/q27v3-{STAMP[:8]}-k{K}-i8/out"
    for tf in sorted(glob.glob(f"{lv}/*/t-*.safetensors")):
        key = os.path.basename(tf)[2:-len(".safetensors")]
        if key.split(".")[3] not in LAYERS: continue
        spec = json.load(open(f"{os.path.dirname(tf)}/job.json"))
        t, meta = tensors(tf)
        res.setdefault(K, {})[key] = {"seed_offset": spec.get("seed_offset", spec.get("spec", {}).get("seed_offset")),
                                      "stamp": meta.get("rxq_settings"), "t": t}
    print(f"K{K}: {len(res.get(K, {}))} tensors", file=sys.stderr)
json.dump(res, open(OUT + ".part", "w"), indent=0); os.replace(OUT + ".part", OUT)
