# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Input-statistics prep stages (run once per model, before any quant job; their outputs are the jobs' input dirs):

blend_hin(A, B, w, OUT)   the xcorp H_in blend (#101): per-token means mixed, (1 - w) A / rows_A + w B / rows_B, stored
                          as a sum over rows_A tokens (A's scale), metadata A's. Port of rad-exl3-mix
                          proxy2b/exp/d1108/blend.py "hin" (byte-identical files: tests/check_cpu.py).
blend_hout(A, B, w, OUT)  the same for H_out diagonals (each normalised to mean 1 first): blend.py "hout".
hout_blocks(SNAPROOT, OUT, est="dy", norm=True)
                          the block-normalized H_out estimator (#33-27): the collection's cumulative snapshots split
                          into 128-sequence blocks, each block scaled to the same per-tensor total, averaged. Port of
                          rad-exl3-mix v3/yaqa/hconv/mkhalf.py (MKHALF_NORM=1); with split=False every block goes into
                          one mean (the production estimator: mkhalf's train/test split is for convergence tests only).

python -m rxq.stats blend-hin A B W OUT [ONLY] | blend-hout A B W OUT [ONLY] | hout SNAPROOT OUT [EST] [--plain] [--split]"""
import glob, json, os, re, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def blend(mode, A, B, W, O, only=None):
    only = re.compile(only) if isinstance(only, str) else only
    os.makedirs(O, exist_ok=True); n = 0
    for f in sorted(glob.glob(f"{A}/*.safetensors")):
        name = os.path.basename(f); fb = f"{B}/{name}"
        if not os.path.exists(fb) or (only and not only.search(name)):
            if not os.path.exists(f"{O}/{name}"): os.symlink(os.path.realpath(f), f"{O}/{name}")
            continue
        with safe_open(f, "pt") as x, safe_open(fb, "pt") as y:
            if mode == "hout":
                a, b = x.get_tensor("hout_diag").double(), y.get_tensor("hout_diag").double()
                v = (1 - W) * a / a.mean() + W * b / b.mean()
                save_file({"hout_diag": v.float().contiguous()}, f"{O}/{name}", metadata={"blend": f"{W} on {B}"})
            else:
                ma, mb = x.metadata(), y.metadata(); ra, rb = float(ma["rows"]), float(mb["rows"])
                assert ma["k"] == mb["k"], name
                u = ((1 - W) * x.get_tensor("hessian_upper").double() / ra + W * y.get_tensor("hessian_upper").double() / rb) * ra
                save_file({"hessian_upper": u.float().contiguous()}, f"{O}/{name}", metadata=ma)
        n += 1
    open(f"{O}/.done", "w").close()
    return n


def blend_hin(A, B, W, O, only=None): return blend("hin", A, B, float(W), O, only)
def blend_hout(A, B, W, O, only=None): return blend("hout", A, B, float(W), O, only)


def hout_blocks(R, O, est="dy", norm=True, split=True):
    """split=True: mkhalf.py's output exactly (OUT/train-N even blocks up to snapshot N, OUT/test the odd blocks,
    mkhalf.json); split=False: OUT/all = every block, the estimator a quant job reads"""
    suf = f"-{est}" if est else ""
    ns = sorted(int(m.group(1)) for p in glob.glob(f"{R}/hdiag-snap*{suf}") if (m := re.search(rf"hdiag-snap(\d+){re.escape(suf)}$", p)))

    def load(n):
        out = {}
        for f in glob.glob(f"{R}/hdiag-snap{n}{suf}/*.safetensors"):
            with safe_open(f, "pt") as h: out[os.path.basename(f)] = h.get_tensor("hout_diag").double()
        return out

    def save(d, acc, cnt, meta):
        os.makedirs(d, exist_ok=True)
        for k, v in acc.items(): save_file({"hout_diag": (v / cnt).float().contiguous()}, f"{d}/{k}", metadata=meta)
    prev, pn, ev, od, ne, no = {}, 0, {}, {}, 0, 0
    for i, n in enumerate(ns):
        cur = load(n); blk = {k: v * n - prev.get(k, 0) * pn for k, v in cur.items()}; m = n - pn
        if norm: blk = {k: v / v.sum().clamp_min(1e-300) * m for k, v in blk.items()}
        tgt = ev if (i % 2 == 0 or not split) else od
        for k, b in blk.items(): tgt[k] = tgt.get(k, 0) + b
        if i % 2 == 0 or not split: ne += m
        else: no += m
        if split:
            if not os.path.exists(f"{O}/train-{n}/.done"):
                save(f"{O}/train-{n}", ev, ne, {"seqs": str(ne), "half": "even", "upto": str(n), "est": est})
                open(f"{O}/train-{n}/.done", "w").close()
        prev, pn = cur, n
    if split:
        if no: save(f"{O}/test", od, no, {"seqs": str(no), "half": "odd", "upto": str(ns[-1]), "est": est})
        json.dump({"snaps": ns, "train_seqs_last": ne, "test_seqs": no}, open(f"{O}/mkhalf.json", "w"))
    else:
        save(f"{O}/all", ev, ne, {"seqs": str(ne), "half": "all", "upto": str(ns[-1]), "est": est, "norm": str(int(norm))})
        open(f"{O}/all/.done", "w").close()
    return ns, ne, no


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if a[0] in ("blend-hin", "blend-hout"):
        n = blend(a[0][6:], a[1], a[2], float(a[3]), a[4], a[5] if len(a) > 5 else None)
        print(f"BLEND DONE {a[0][6:]} W={float(a[3])}: {n} blended -> {a[4]}")
    elif a[0] == "hout":
        pos = [x for x in a[1:] if not x.startswith("--")]
        ns, ne, no = hout_blocks(pos[0], pos[1], pos[2] if len(pos) > 2 else "dy", norm="--plain" not in a, split="--split" in a)
        print(f"snapshots {ns}: {ne} + {no} seqs")
    else: raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
