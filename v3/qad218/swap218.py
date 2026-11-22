#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""swap218.py RAD LV CODES OUT.rad -- a copy of a released container with the probe tensors' re-encoded records (#218): for every trunk
weight whose level keys all have CODES/<key>.rec (encode218.py's records, exl3i plane 1 layout [n/16][k/16][m/2]), its `record` plane
becomes those records (ffn_gate_up: the gate's tile rows, then the up's). Nothing else changes (same size, planes, offsets, scales);
checked: only the replaced record planes differ from the source. LV = rad2level214.py's OUT (map214.json). Written OUT.part, renamed."""
import argparse, json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
from rad214 import Rad  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rad"); ap.add_argument("lv"); ap.add_argument("codes"); ap.add_argument("out")
    a = ap.parse_args()
    rad = Rad(a.rad); mp = json.load(open(f"{a.lv}/map214.json"))["weights"]
    part = a.out + ".part"; shutil.copyfile(a.rad, part)
    spans, changed, swapped = [], 0, []
    with open(part, "r+b") as f:
        for w, d in mp.items():
            recs = [os.path.join(a.codes, k + ".rec") for k in d["keys"]]
            if not all(os.path.exists(r) for r in recs): continue
            assert not any(os.path.exists(r) for r in recs) or all(os.path.exists(r) for r in recs), w
            new = b"".join(np.fromfile(r, "<u4").tobytes() for r in recs)
            po, pn = rad.entries[w].planes["record"]
            assert len(new) == pn, (w, len(new), pn)
            f.seek(po); old = f.read(pn)
            if old != new: changed += 1; f.seek(po); f.write(new)
            spans.append((po, po + pn)); swapped.append(w)
    st = np.array(sorted(s for s, _ in spans)); en = np.array(sorted(e for _, e in spans))
    bad = 0
    with open(a.rad, "rb") as f1, open(part, "rb") as f2:
        pos, CH, size = 0, 1 << 26, os.path.getsize(a.rad)
        while pos < size:
            b1, b2 = f1.read(CH), f2.read(CH)
            if b1 != b2:
                o = pos + np.nonzero(np.frombuffer(b1, np.uint8) != np.frombuffer(b2, np.uint8))[0]
                k = np.searchsorted(st, o, side="right") - 1
                bad += int(((k < 0) | (o >= en[np.maximum(k, 0)])).sum())
            pos += len(b1)
    assert bad == 0, f"{bad} bytes differ outside the swapped record planes"
    os.replace(part, a.out)
    print(f"SWAP218 DONE {a.out}: {len(swapped)} weights swapped ({changed} records differ), only their record planes changed")


if __name__ == "__main__":
    main()
