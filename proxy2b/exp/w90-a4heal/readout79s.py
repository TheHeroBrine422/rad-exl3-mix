#!/usr/bin/env python3
"""readout79s.py O PCMP GOLDEN -- #89-06c on the student path (w79s.sh): w79's K4 rule (../w79-heal45/README.md) applied
to K4 at the held-out-picked lr and to K5 at lr 1e-4: per seed, healed / stack KL ratio on g08-search1 (+ pcmp z) and on
g08-verify1 + verify2 pooled (gen rows of both). Heal if both seeds <= 0.95 on search and <= 0.97 on verify; small if
some seed <= 0.97 on search; otherwise do not heal."""
import json, os, subprocess, sys
import numpy as np

O, PCMP, S = sys.argv[1:4]
pick = open(f"{O}/pick").read().strip()


def kl(t):
    f = f"{O}/{t}.f32"
    if not os.path.exists(f): return None
    a = np.fromfile(f, np.float32).reshape(-1, 4); g = a[:, 2] > 0
    return float(a[g, 0].sum()), int(g.sum())


def z(b, h):
    try:
        r = json.loads(subprocess.run(["python3", PCMP, S, f"{O}/{b}.f32", f"{O}/{h}.f32"], capture_output=True, text=True).stdout.strip().splitlines()[-1])
        return r["z"]
    except Exception: return float("nan")


print(f"#89-06c on the student path (heal.py's wrappers, no fold): K4 lr {pick} (held-out pick), K5 lr 1e-4")
out = {}
for K, lr in ((4, pick), (5, "1e-4")):
    ok_all, small = True, False
    for s in ("", "_s1"):
        b, h = f"m-k{K}stack{s}", f"m-k{K}edyswjasym_r0_lr{lr}{s}"
        rs = kl(h)[0] / kl(b)[0] if kl(h) and kl(b) else None
        v = [(kl(b + x), kl(h + x)) for x in (".v1", ".v2")]
        rv = sum(y[0] for _, y in v) / sum(x[0] for x, _ in v) if all(x and y for x, y in v) else None
        zs = z(b, h) if rs is not None else float("nan")
        print(f"K{K} seed {s or '_s0'}: search {'-' if rs is None else f'{rs:.4f} (z {zs:+.1f})'}, verify1+2 {'-' if rv is None else f'{rv:.4f}'}")
        out[f"K{K}{s or '_s0'}"] = {"search": rs, "z": zs, "verify": rv}
        if rs is None or rv is None: ok_all = None; continue
        if ok_all and not (rs <= 0.95 and rv <= 0.97): ok_all = False
        if rs <= 0.97: small = True
    if K == 4:
        for s in ("", "_s1"):
            o = "1e-4" if pick == "3e-4" else "3e-4"; b, h = f"m-k4stack{s}", f"m-k4edyswjasym_r0_lr{o}{s}"
            if kl(h) and kl(b): print(f"  (report) K4 lr {o} seed {s or '_s0'}: search {kl(h)[0] / kl(b)[0]:.4f}")
    v = ("incomplete" if ok_all is None else f"HEAL K{K} (both seeds <= 0.95 on search and <= 0.97 on verify)" if ok_all
         else f"small (some seed <= 0.97 on search): K{K} healing worth <= ~5%, lead decides with its cost" if small
         else f"do not heal K{K}")
    out[f"verdict_K{K}"] = v
    print(f"W79S VERDICT K{K}: {v}")
json.dump(out, open(f"{O}/readout79s.json", "w"), indent=1)
