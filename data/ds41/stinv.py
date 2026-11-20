import json, struct, glob, os, re, collections
D = "/data/ds41ck-full"
rows = collections.defaultdict(lambda: [0, 0, set()])   # group -> [elements, bytes, dtypes]
DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "F8_E8M0": 1, "I32": 4, "I64": 8, "BOOL": 1}
names = []
for f in sorted(glob.glob(f"{D}/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        el = 1
        for d in v["shape"]: el *= d
        b = v["data_offsets"][1] - v["data_offsets"][0]
        g = re.sub(r"\.\d+\.", ".N.", k); g = re.sub(r"\.\d+\.", ".N.", g)
        rows[g][0] += el; rows[g][1] += b; rows[g][2].add((v["dtype"], tuple(v["shape"]) if len(names) < 0 else len(v["shape"])))
        names.append(k)
out = {g: [r[0], r[1], sorted(r[2])] for g, r in rows.items()}
json.dump(out, open("/tmp/stinv.json", "w"))
print(len(names), "tensors", len(out), "groups")
