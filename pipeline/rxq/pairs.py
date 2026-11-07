# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Swap-pair hygiene: a swap test compares two models that differ only in its arm (a family, a few tensors, or a
plane of them). These tools refuse a pair that differs anywhere else, so code or build drift, a stale donor or a
wrong level is never read as the arm's effect (the lesson of #104's K4 head mismatch, LOG 11-06 7:50 PM CST).

  python -m rxq.pairs levels A B --arm SPEC [--planes P,..] [--allow-same]
      two ourq-layout levels (q-L*.safetensors + index): every tensor outside the arm torch.equal (dtype, shape,
      values); arm tensors may differ (only in the named planes, e.g. svh, if --planes is given)
  python -m rxq.pairs mixes A.rad B.rad --arm SPEC [--planes P,..] [--allow-same]
      two packed containers (RAD2, read with rad-exl3-mix store/census.py): the same weight set; outside the arm every
      weight identical (shape, quantiser, options but the recipe's src=, encoding struct, every plane's offset / size /
      bytes) and the same act mode; arm weights may differ (only in the named planes if --planes)
  python -m rxq.pairs same BASE JOBDIR
      the no-op gate: every tensor of JOBDIR (rxq job outputs t-<key>.safetensors) bit-identical to BASE's
  python -m rxq.pairs splice BASE JOBDIR DST --arm SPEC
      DST = BASE's level with JOBDIR's tensors in place of exactly the arm's linears (all of them, nothing else);
      provenance in DST/spliced.json; per shard .part -> rename, resumable

SPEC: comma list of families (rxq names: attn_qg attn_k attn_v attn_output ssm_qkv ssm_z ssm_out ffn_gate ffn_up
ffn_down head; container names ffn_gate_up = gate + up), optionally restricted to layers FAM@L1+L2, or checkpoint keys
(model.language_model.layers.N....). Unless --allow-same, a pair whose arm is identical too is refused (a no-op arm).
Last line "PAIR OK {json}" or "PAIR REFUSED {json}"; exit 1 on REFUSED."""
import argparse, hashlib, json, os, struct, sys, time

from . import qwen35

L = qwen35.L
RAD_NAME = {"attn_qg": "attn_qg", "attn_k": "attn_k", "attn_v": "attn_v", "attn_output": "attn_output", "ssm_qkv": "ssm_qkv",
            "ssm_z": "ssm_z", "ssm_out": "ssm_out", "ffn_gate": "ffn_gate_up", "ffn_up": "ffn_gate_up", "ffn_down": "ffn_down",
            "head": "output"}


def parse_arm(spec, n_layers=64):
    """-> set of checkpoint keys (trunk linears; lm_head)"""
    keys = set()
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok.startswith(L + ".") or tok == "lm_head":
            qwen35.locate(tok); keys.add(tok); continue
        fam, _, ls = tok.partition("@")
        fams = {"ffn_gate", "ffn_up"} if fam == "ffn_gate_up" else {fam}
        if not fams <= set(qwen35.FAMILIES): raise SystemExit(f"--arm: unknown family {fam!r} (known: {', '.join(qwen35.FAMILIES)}, ffn_gate_up)")
        layers = {int(x) for x in ls.split("+")} if ls else set(range(n_layers))
        if "head" in fams: keys.add("lm_head")
        keys |= {k for k in qwen35.trunk_keys(n_layers) if qwen35.family(k) in fams and qwen35.layer_of(k) in layers}
    if not keys: raise SystemExit(f"--arm {spec!r}: no tensors")
    return keys


def rad_name(key):
    """checkpoint key -> the container's weight name (blk.L.<name>.weight; gate / up share ffn_gate_up)"""
    if key == "lm_head": return "output.weight"
    return f"blk.{qwen35.layer_of(key)}.{RAD_NAME[qwen35.family(key)]}.weight"


def verdict(ok, st, problems):
    for p in problems[:20]: print("MISMATCH", p)
    if len(problems) > 20: print(f"... {len(problems) - 20} more")
    print(f"PAIR {'OK' if ok else 'REFUSED'} {json.dumps(st)}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------- levels
def _index(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


def _reader(d):
    from safetensors import safe_open
    h = {}

    def get(name, f):
        if f not in h: h[f] = safe_open(os.path.join(d, f), "pt", device="cpu")
        return h[f].get_tensor(name)
    return get


def tensor_key(name):
    """<key>.suh / .svh / .trellis / .mcg|.mul1 ... -> (key, plane)"""
    k, _, p = name.rpartition(".")
    return k, p


def cmp_levels(a, b, arm, planes=None, allow_same=False):
    import torch
    ia, ib = _index(a), _index(b)
    ga, gb = _reader(a), _reader(b)
    problems, same, diff_arm, diff_planes = [], 0, set(), set()
    if set(ia) != set(ib):
        x = sorted(set(ia) ^ set(ib)); problems.append(f"tensor sets differ ({len(x)}: {x[:3]})")
    for n in sorted(set(ia) & set(ib)):
        x, y = ga(n, ia[n]), gb(n, ib[n])
        eq = x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
        key, plane = tensor_key(n)
        if eq: same += 1; continue
        if key not in arm: problems.append(f"{n}: differs outside the arm"); continue
        if planes and plane not in planes: problems.append(f"{n}: plane {plane} differs (arm planes: {sorted(planes)})"); continue
        diff_arm.add(key); diff_planes.add(plane)
    missing = sorted(k for k in arm if not any(n.startswith(k + ".") for n in ia))
    if missing: problems.append(f"arm tensors not in the levels: {missing[:3]}")
    if not diff_arm and not allow_same: problems.append("the arm is identical too: a no-op pair (--allow-same to accept)")
    st = {"a": a, "b": b, "tensors": len(set(ia) & set(ib)), "identical": same, "arm_keys": len(arm),
          "arm_keys_changed": len(diff_arm), "planes_changed": sorted(diff_planes), "problems": len(problems)}
    return not problems, st, problems


# ---------------------------------------------------------------------------------------------------------- mixes
def _census():
    from .engine import rxm_root
    p = os.environ.get("RXQ_CENSUS") or os.path.join(rxm_root(), "store", "census.py")
    import importlib.util
    sp = importlib.util.spec_from_file_location("rxq_census", p)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def _planes(C, R, n):
    """[(role, relative offset, size, sha256)] of weight n"""
    w = R.w[n]; eb = w["enc"]; k = struct.unpack_from("<I", eb, 48)[0]
    roles = [C.PLANE.unpack_from(eb, 56 + 64 * i)[0].split(b"\0")[0].decode() for i in range(k)]
    out = []
    for i, (po, pn) in enumerate(w["planes"]):
        R.f.seek(po); h = hashlib.sha256(); left = pn
        while left:
            b = R.f.read(min(C.CH, left)); h.update(b); left -= len(b)
        out.append((roles[i] if i < len(roles) else f"plane{i}", po - w["offset"], pn, h.hexdigest()))
    return out


def cmp_mixes(a, b, arm, planes=None, allow_same=False):
    C = _census()
    A, B = C.Rad(a), C.Rad(b)
    names = {rad_name(k) for k in arm}
    nosrc = lambda o: ",".join(t for t in o.split(",") if not t.startswith("src="))
    problems, same, diff_arm, diff_planes = [], 0, set(), set()
    if set(A.w) != set(B.w):
        x = sorted(set(A.w) ^ set(B.w)); problems.append(f"weight sets differ ({len(x)}: {x[:3]})")
    for n in sorted(set(A.w) & set(B.w)):
        wa, wb = A.w[n], B.w[n]
        act = (A.meta.get(f"exl3.act.{n}", "f16"), B.meta.get(f"exl3.act.{n}", "f16"))
        if wa["shape"] != wb["shape"]: problems.append(f"{n}: shape {wa['shape']} vs {wb['shape']}"); continue
        hdr_eq = wa["quant"] == wb["quant"] and nosrc(wa["opts"]) == nosrc(wb["opts"]) and act[0] == act[1]
        pa, pb = _planes(C, A, n), _planes(C, B, n)
        dp = [x[0] for x, y in zip(pa, pb) if x[1:] != y[1:]] + (["(plane count)"] if len(pa) != len(pb) else [])
        if hdr_eq and wa["enc"] == wb["enc"] and not dp: same += 1; continue
        if n not in names:
            problems.append(f"{n}: differs outside the arm (header {'equal' if hdr_eq else 'differs'}, encoding "
                            f"{'equal' if wa['enc'] == wb['enc'] else 'differs'}, planes {dp})"); continue
        if planes and (not hdr_eq or wa["enc"] != wb["enc"] or set(dp) - set(planes)):
            problems.append(f"{n}: differs beyond the arm's planes {sorted(planes)} (planes {dp}, header / encoding "
                            f"{'equal' if hdr_eq and wa['enc'] == wb['enc'] else 'differ'})"); continue
        diff_arm.add(n); diff_planes |= set(dp)
    missing = sorted(names - set(A.w))
    if missing: problems.append(f"arm weights not in the containers: {missing[:3]}")
    if not diff_arm and not allow_same: problems.append("the arm is identical too: a no-op pair (--allow-same to accept)")
    st = {"a": a, "b": b, "weights": len(set(A.w) & set(B.w)), "identical": same, "arm_weights": len(names),
          "arm_weights_changed": len(diff_arm), "planes_changed": sorted(diff_planes), "problems": len(problems)}
    return not problems, st, problems


# ---------------------------------------------------------------------------------------------------------- jobs
def _job_tensors(jobdir):
    files = {}
    for f in os.listdir(jobdir):
        if f.startswith("t-") and f.endswith(".safetensors"): files[f[2:-len(".safetensors")]] = os.path.join(jobdir, f)
    return files


def same(base, jobdir):
    import torch
    from safetensors import safe_open
    ib, gb = _index(base), _reader(base)
    problems, n = [], 0
    files = _job_tensors(jobdir)
    for key, p in sorted(files.items()):
        with safe_open(p, "pt", device="cpu") as f:
            for t in f.keys():
                n += 1
                if t not in ib: problems.append(f"{t}: not in {base}"); continue
                x, y = f.get_tensor(t), gb(t, ib[t])
                if not (x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)): problems.append(f"{t}: differs")
        mine = {t for t in ib if t.startswith(key + ".")}
        with safe_open(p, "pt", device="cpu") as f:
            if set(f.keys()) != mine: problems.append(f"{key}: tensor set {sorted(f.keys())} vs the level's {sorted(mine)}")
    st = {"base": base, "job": jobdir, "keys": len(files), "tensors": n, "problems": len(problems)}
    return not problems and bool(files), st, problems or ([] if files else ["no job tensors"])


def splice(base, jobdir, dst, arm):
    from safetensors import safe_open
    from safetensors.torch import save_file
    ib, gb = _index(base), _reader(base)
    files = _job_tensors(jobdir)
    if set(files) != arm:
        raise SystemExit(f"splice: the job holds {len(files)} tensors, the arm {len(arm)}; missing {sorted(arm - set(files))[:3]}, "
                         f"extra {sorted(set(files) - arm)[:3]}")
    new = {}
    for key, p in files.items():
        with safe_open(p, "pt", device="cpu") as f:
            t = {n: f.get_tensor(n) for n in f.keys()}
        mine = {n for n in ib if n.startswith(key + ".")}
        if set(t) != mine: raise SystemExit(f"splice: {key}: tensors {sorted(t)} vs the base level's {sorted(mine)}")
        for n, v in t.items():
            ref = gb(n, ib[n])
            if v.dtype != ref.dtype or v.shape != ref.shape: raise SystemExit(f"splice: {n}: {v.dtype} {tuple(v.shape)} vs {ref.dtype} {tuple(ref.shape)}")
        new.update(t)
    os.makedirs(dst, exist_ok=True)
    shards = sorted(set(ib.values()))
    for sh in shards:
        out = os.path.join(dst, sh)
        if os.path.exists(out): continue
        with safe_open(os.path.join(base, sh), "pt", device="cpu") as f:
            t = {n: (new[n] if n in new else f.get_tensor(n)) for n in f.keys()}; md = dict(f.metadata() or {})
        md["rxq_spliced"] = json.dumps({"base": os.path.abspath(base), "job": os.path.abspath(jobdir)})
        save_file(t, out + ".part", metadata=md); os.replace(out + ".part", out)
    for f in ("model.safetensors.index.json", "config.json", "stamp.json", "proxy.jsonl"):
        if os.path.exists(os.path.join(base, f)) and not os.path.exists(os.path.join(dst, f)):
            open(os.path.join(dst, f), "wb").write(open(os.path.join(base, f), "rb").read())
    js = {}
    jj = os.path.join(jobdir, "job.json")
    if os.path.exists(jj): js = json.load(open(jj))
    json.dump({"base": os.path.abspath(base), "job": os.path.abspath(jobdir), "keys": sorted(arm),
               "job_settings_stamp": js.get("settings_stamp"), "job_code_stamp": js.get("code_stamp"),
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(os.path.join(dst, "spliced.json"), "w"), indent=1)
    return len(shards), len(new)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxq.pairs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("levels", "mixes"):
        p = sub.add_parser(c); p.add_argument("a"); p.add_argument("b"); p.add_argument("--arm", required=True)
        p.add_argument("--planes", default=None); p.add_argument("--allow-same", action="store_true")
        p.add_argument("--layers", type=int, default=64)
    p = sub.add_parser("same"); p.add_argument("base"); p.add_argument("job")
    p = sub.add_parser("splice"); p.add_argument("base"); p.add_argument("job"); p.add_argument("dst"); p.add_argument("--arm", required=True)
    p.add_argument("--layers", type=int, default=64)
    a = ap.parse_args(argv)
    if a.cmd in ("levels", "mixes"):
        planes = set(a.planes.split(",")) if a.planes else None
        f = cmp_levels if a.cmd == "levels" else cmp_mixes
        return verdict(*f(a.a, a.b, parse_arm(a.arm, a.layers), planes, a.allow_same))
    if a.cmd == "same":
        return verdict(*same(a.base, a.job))
    n, t = splice(a.base, a.job, a.dst, parse_arm(a.arm, a.layers))
    print(f"SPLICE DONE {t} tensors into {n} shards -> {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
