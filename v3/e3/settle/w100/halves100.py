#!/usr/bin/env python3
"""halves100.py SNAPROOT OUT [BNDIR] -- w100 (#129, ../w100.md): the 27B Σdy² collection's cumulative snapshots
(SNAPROOT/hdiag-snapN-dy, running means over N sequences, N = 128 .. 1920) differenced into its 15 blocks of 128
sequences, each block scaled to the same per-tensor total (MKHALF_NORM=1 / mkbn.py's block normalization), then averaged
over block sets, written as ourq --hout dirs (OUT/<set>/<key>.safetensors, tensor "hout_diag"):
  F   all 15 blocks (= mkbn's bn1920 up to one constant per tensor: the production H_out)
  E   even blocks 0, 2, .., 14 (8 blocks, 1,024 sequences): what the proxy's quantizer sees
  O   odd blocks 1, 3, .., 13 (7 blocks): held out
  O1  odd blocks 1, 5, 9, 13 (pick half);  O2  odd blocks 3, 7, 11 (judge half)
Gate G2 (BNDIR = the bn1920 dir the ladder uses): F / mean(F) == bn / mean(bn) for every key to 1e-5 (float32 files),
15 blocks, every h of every set finite and > 0. Writes OUT/halves.json; OUT/.done last. Restartable (a finished set
file is not rewritten)."""
import glob, json, os, re, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

R, O = sys.argv[1], sys.argv[2]
BN = sys.argv[3] if len(sys.argv) > 3 else None
SETS = {"F": list(range(15)), "E": list(range(0, 15, 2)), "O": list(range(1, 15, 2)), "O1": [1, 5, 9, 13], "O2": [3, 7, 11]}
ns = sorted(int(m.group(1)) for p in glob.glob(f"{R}/hdiag-snap*-dy") if (m := re.search(r"hdiag-snap(\d+)-dy$", p)))
assert ns == [128 * (i + 1) for i in range(15)], ns
keys = sorted(os.path.basename(f) for f in glob.glob(f"{R}/hdiag-snap1920-dy/*.safetensors"))
assert len(keys) == 400, len(keys)
for s in SETS: os.makedirs(f"{O}/{s}", exist_ok=True)


def get(n, k):
    with safe_open(f"{R}/hdiag-snap{n}-dy/{k}", "pt") as h: return h.get_tensor("hout_diag").double()


rep = {"snaps": ns, "sets": SETS, "keys": len(keys), "gate_max_rel": 0.0, "min_h": {}, "bad": []}
for k in keys:
    blocks, prev, pn = [], None, 0
    for n in ns:
        cur = get(n, k); b = cur * n - (prev * pn if prev is not None else 0); m = n - pn
        blocks.append(b / b.sum().clamp_min(1e-300) * m)
        prev, pn = cur, n
    for s, ix in SETS.items():
        h = sum(blocks[i] for i in ix) / (128 * len(ix))
        if not (bool(torch.isfinite(h).all()) and float(h.min()) > 0): rep["bad"].append([s, k, float(h.min())])
        rep["min_h"][s] = min(rep["min_h"].get(s, 1e300), float(h.min() / h.mean()))
        f = f"{O}/{s}/{k}"
        if not os.path.exists(f):
            save_file({"hout_diag": h.float().contiguous()}, f + ".part", metadata={"est": "dy", "blocknorm": "1", "blocks": ",".join(map(str, ix))})
            os.replace(f + ".part", f)
        if s == "F" and BN:
            with safe_open(f"{BN}/{k}", "pt") as hh: bn = hh.get_tensor("hout_diag").double()
            a, b = h.float().double(), bn
            rel = float(((a / a.mean()) - (b / b.mean())).abs().max() / (b / b.mean()).abs().max())
            rep["gate_max_rel"] = max(rep["gate_max_rel"], rel)
rep["G2"] = (not rep["bad"]) and (BN is None or rep["gate_max_rel"] <= 1e-5)
json.dump(rep, open(f"{O}/halves.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "bad"}), "bad", len(rep["bad"]))
if rep["G2"]: open(f"{O}/.done", "w").close()
else: raise SystemExit("G2 FAIL")
