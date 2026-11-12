#!/usr/bin/env python3
"""sub1ratios.py OUT.json PROXY.jsonl... [--ref-k 1] [--err proxy_err|proxy_err_2s] [--scale SCALES.json] [--cmp CPU.json]
-- knap5 KN_SUB1 prices for the V3R menu's sub-1 i8 options from the 27B levels' own proxies (w120, 2026-11-12).

PROXY.jsonl: a level's level/proxy.jsonl (rxq assemble; one row per tensor: key, K, proxy_err, proxy_err_2s), one
file per level; the K1 level (--ref-k) is the reference. Per tensor and sub-1 K: ratio = err(K) / err(K1) (gate / up:
geometric mean of the two, as knap5's KN_LEVELS); per family: the geometric mean over its tensors. Written:
  {"ratios": {family: {k: geo}}, "tensors": {knap name: {k: ratio}}, "kl_scale": {k: s} (from --scale, else none),
   "spread": {family: {k: [p10, p90]}}, "src": [...], "err": ...}
knap5 reads "ratios" (and "kl_scale"); "tensors" is the per-tensor table (knap5 KN_SUB1_TENSORS=1 uses it where a
tensor has every k). The one-sided proxy_err is kfit2's error measure (its s_w were fitted on it); proxy_err_2s is
written for the record only (#139: the two-sided error is part 3's prior, not knap5's input).
--cmp CPU.json (data/e3/sub1/sub1ratios.json, #90's CPU trellis on the 11-02 stack) prints both side by side."""
import argparse, collections, json, math, os, sys

HF2KN = {"self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "linear_attn.in_proj_qkv": "ssm_qkv", "linear_attn.in_proj_z": "ssm_z",
         "linear_attn.out_proj": "ssm_out", "mlp.gate_proj": "ffn_gate_up", "mlp.up_proj": "ffn_gate_up",
         "mlp.down_proj": "ffn_down"}


def kname(key):
    k = key.split(".")
    if k[2] != "layers": return None
    n = HF2KN.get(".".join(k[4:6]))
    return f"blk.{k[3]}.{n}.weight" if n else None


def geo(v): return math.exp(sum(map(math.log, v)) / len(v))


def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


def read(paths, err):
    E = collections.defaultdict(lambda: collections.defaultdict(list))      # knap name -> K -> errors (gate, up)
    for p in paths:
        for l in open(p):
            d = json.loads(l); n = kname(d["key"])
            if n is None or d.get(err) is None or not d[err] > 0: continue
            E[n][float(d["K"])].append(d[err])
    return {n: {K: geo(v) for K, v in m.items()} for n, m in E.items()}


def build(paths, ref_k=1.0, err="proxy_err", scale=None):
    E = read(paths, err)
    T = {}
    for n, m in E.items():
        if ref_k not in m: continue
        r = {k: v / m[ref_k] for k, v in m.items() if 0 < k < 1}
        if r: T[n] = r
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, r in T.items():
        for k, v in r.items(): fam[n.split(".")[-2]][k].append(v)
    out = {"ratios": {f: {f"{k:g}": geo(v) for k, v in sorted(m.items())} for f, m in sorted(fam.items())},
           "tensors": {n: {f"{k:g}": v for k, v in sorted(r.items())} for n, r in sorted(T.items())},
           "spread": {f: {f"{k:g}": [q(v, 0.1), q(v, 0.9)] for k, v in sorted(m.items())} for f, m in sorted(fam.items())},
           "n": {f: {f"{k:g}": len(v) for k, v in sorted(m.items())} for f, m in sorted(fam.items())},
           "err": err, "ref_k": ref_k, "src": [os.path.basename(p) for p in paths]}
    if scale: out["kl_scale"] = scale
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sub1ratios.py")
    ap.add_argument("out"); ap.add_argument("proxies", nargs="+"); ap.add_argument("--ref-k", type=float, default=1.0)
    ap.add_argument("--err", default="proxy_err"); ap.add_argument("--scale"); ap.add_argument("--cmp")
    a = ap.parse_args(argv)
    sc = json.load(open(a.scale)) if a.scale else None
    if sc and "kl_scale_extra" in sc: sc = sc["kl_scale_extra"]
    o = build(a.proxies, a.ref_k, a.err, sc)
    if not o["ratios"]: raise SystemExit(f"no sub-1 tensor with a K{a.ref_k:g} reference in {a.proxies}")
    json.dump(o, open(a.out + ".part", "w"), indent=1); os.replace(a.out + ".part", a.out)
    cpu = json.load(open(a.cmp))["ratios"] if a.cmp else {}
    print(f"{a.out}: {len(o['tensors'])} tensors, families {len(o['ratios'])}, err {a.err}")
    for f, m in o["ratios"].items():
        print(f"  {f:12s} " + "  ".join(f"K{k}: {v:6.3f} [{o['spread'][f][k][0]:.2f}, {o['spread'][f][k][1]:.2f}]"
                                         + (f" cpu {cpu[f][k]:.3f}" if f in cpu and k in cpu[f] else "") for k, v in m.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
