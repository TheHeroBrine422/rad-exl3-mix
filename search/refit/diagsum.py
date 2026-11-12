#!/usr/bin/env python3
"""diagsum.py OUT.jsonl LEVELDIR... -- per-tensor summary of the ladder jobs' #139 diag files (w120, 11-12): one line per
(level, tensor) with the sums kfit3's Fisher prior needs. A LEVELDIR is an rxrun level root (out/<job>/diag-*.safetensors)
or any dir holding diag-*.safetensors files (searched two deep). Per tensor (rxq/refit.py's definitions, H_e = the
undamped calib H_in, h = the raw H_out diagonal):
  R  = sum_n r1_n   (= W H_e W^T: the weight's energy in the input metric)
  E1 = sum_n e1_n   (one-sided error; E1 / R = the logged proxy_err)
  E2 = sum_n e2_n   (= sum_n h_n e1_n: the two-sided error, ~2x the Fisher price of the tensor's error, unnormalised)
  hbar = E2 / E1    (the error-weighted output sensitivity; the per-tensor prior's s_w ~ hbar * R)
  E1pre / E2pre (before the refit), n (output channels), K, level, settings stamp.
CPU only, numpy + the safetensors header (no torch); ~1 s a level-file on gpubox."""
import glob, json, os, struct, sys

import numpy as np

DT = {"F32": np.float32, "F16": np.float16}


def read_st(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        out = {}
        for k, v in h.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]; f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), DT[v["dtype"]]).reshape(v["shape"])
    return h.get("__metadata__", {}), out


def summarize(p):
    md, t = read_st(p)
    key = md.get("key") or os.path.basename(p)[5:-12]
    g = lambda s: t.get(f"{key}.{s}")
    r1, e1, e2, e1p, e2p = g("r1"), g("e1"), g("e2"), g("e1_pre"), g("e2_pre")
    S = lambda a: None if a is None else float(np.asarray(a, np.float64).sum())
    o = {"key": key, "level": md.get("level"), "K": float(md["K"]) if md.get("K") else None, "stamp": md.get("rxq_settings"),
         "n": int(r1.shape[0]) if r1 is not None else None, "R": S(r1), "E1": S(e1), "E2": S(e2), "E1pre": S(e1p), "E2pre": S(e2p)}
    o["hbar"] = o["E2"] / o["E1"] if o["E2"] and o["E1"] else None
    return o


def main(argv):
    out, dirs = argv[0], argv[1:]
    n = 0
    with open(out + ".part", "w") as f:
        for d in dirs:
            fs = sorted(set(glob.glob(f"{d}/diag-*.safetensors") + glob.glob(f"{d}/*/diag-*.safetensors")
                            + glob.glob(f"{d}/out/*/diag-*.safetensors")))
            for p in fs:
                f.write(json.dumps(summarize(p)) + "\n"); n += 1
    os.replace(out + ".part", out)
    print(f"{out}: {n} tensors from {len(dirs)} dirs")


if __name__ == "__main__":
    main(sys.argv[1:])
