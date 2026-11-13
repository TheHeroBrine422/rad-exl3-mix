#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkmenu.py OUT.json --levels LEVELDIR... --base BASE.rad [--diag LEVELDIR] [--heads HCURVE.json --head-donor PATTERN]
-- the search menu (menu.py) from ladder level dirs (w126, #166). Runs on the host with the level dirs; numpy only.

Per level dir (rxrun ladder root/<level>: level/proxy.jsonl, final/q-LNN.safetensors) and trunk weight (rxq.mix's
names; HF parts per weight as kfit3.kname maps them):
  bytes = the parts' stored tensors in final/ (trellis + scales + mul1: what the pack stores for them)
  err   = sum over the parts of proxy_err x R (R = the part's input-metric energy from --diag's diag files, diagsum.py;
          kfit3's absolute error). fam = "v3" + codebook (v3i8 / v3i4 / v3mul1)
hbar per weight (kfit3's Fisher prior) from the --diag level (E2 / E1 summed over its parts).
Head / embedding: the base container's own (one option "base" each, bytes from BASE.rad), or with --heads a head
option per plan of w122's price table (hcurve122.json: avg_bits, kl, donor): bytes = vocab x 5120 x avg_bits / 8 + 1 MiB
(the QUANTIZED head's bytes; the eval splices its bf16-dequantized R16 rows: TODO "Heads before w124's un-permute"),
err = the plan's head-only KL, src.path = --head-donor with {plan} filled (the spliced donor container on the host).
fixed_bytes = BASE.rad's total minus its trunk / head / embedding entries."""
import argparse, glob, json, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "refit"))
import menu as MN       # noqa: E402
import diagsum as DS    # noqa: E402

HF2KN = {"self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "linear_attn.in_proj_qkv": "ssm_qkv", "linear_attn.in_proj_z": "ssm_z", "linear_attn.out_proj": "ssm_out",
         "mlp.gate_proj": "ffn_gate_up", "mlp.up_proj": "ffn_gate_up", "mlp.down_proj": "ffn_down"}


def kname(key):
    k = key.split(".")
    if len(k) < 6 or k[2] != "layers": return None
    n = HF2KN.get(".".join(k[4:6]))
    return f"blk.{k[3]}.{n}.weight" if n else None


def st_sizes(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    return {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items() if k != "__metadata__"}


def rad_entries(p):
    f = open(p, "rb"); h = f.read(4096)
    magic, ver, fbytes = struct.unpack_from("<IIQ", h, 0)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt); w = {}
    for i in range(dir_cnt):
        e = D[i * es:(i + 1) * es]; name = struct.unpack_from("<Q", e, 0)[0]
        w[S[name:S.index(b"\0", name)].decode()] = struct.unpack_from("<Q", e, es - 16 - 8)[0]
    return fbytes, w


def weight_of(entry):
    m = re.match(r"(blk\.\d+\.\w+?\.weight|output\.weight|token_embd\.weight)", entry)
    return m.group(1) if m else None


def level_opts(d):
    name = os.path.basename(d.rstrip("/"))
    prox = {}
    for l in open(f"{d}/level/proxy.jsonl"):
        r = json.loads(l); prox[r["key"]] = r
    sizes = {}
    for p in glob.glob(f"{d}/final/q-L*.safetensors"): sizes.update(st_sizes(p))
    return name, prox, sizes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mkmenu.py")
    ap.add_argument("out"); ap.add_argument("--levels", nargs="+", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--diag"); ap.add_argument("--heads"); ap.add_argument("--head-donor"); ap.add_argument("--vocab", type=int, default=248320)
    a = ap.parse_args(argv)
    trunk = MN.trunk_names(64)
    R, E2 = {}, {}
    dd = a.diag or a.levels[0]
    for p in sorted(glob.glob(f"{dd}/out/*/diag-*.safetensors")):
        s = DS.summarize(p); R[s["key"]] = s["R"]; E2[s["key"]] = (s["E1"], s["E2"])
    T = {w: {"w": w, "layer": MN.layer_of(w), "group": MN.group_of(w), "hbar": None, "opts": []} for w in trunk}
    hb = {}
    for k, (e1, e2) in E2.items():
        w = kname(k)
        if w and e1: hb.setdefault(w, [0.0, 0.0]); hb[w][0] += e1; hb[w][1] += e2
    for w, (e1, e2) in hb.items(): T[w]["hbar"] = e2 / e1 if e1 else None
    for d in a.levels:
        name, prox, sizes = level_opts(d)
        acc = {}
        for key, r in prox.items():
            w = kname(key)
            if w is None: continue
            b = sum(v for k, v in sizes.items() if k.startswith(key + "."))
            x = acc.setdefault(w, {"o": name, "K": float(r["K"]), "bytes": 0, "err": 0.0, "fam": "v3" + (r.get("codebook") or "mul1"), "parts": 0})
            x["bytes"] += b; x["parts"] += 1
            if key not in R: raise SystemExit(f"no diag R for {key} (--diag)")
            x["err"] += r["proxy_err"] * R[key]
        miss = [w for w in trunk if w not in acc]
        if miss: raise SystemExit(f"{name}: {len(miss)} trunk weights missing, e.g. {miss[:3]}")
        for w, x in acc.items(): x.pop("parts"); T[w]["opts"].append(x)
    total, ent = rad_entries(a.base)
    byw = {}
    for e, b in ent.items():
        w = weight_of(e)
        if w: byw[w] = byw.get(w, 0) + b
    fixed = total - sum(byw.get(w, 0) for w in trunk) - byw.get(MN.HEAD, 0) - byw.get(MN.EMB, 0)
    tens = [T[w] for w in trunk]
    if a.heads:
        hc = json.load(open(a.heads))["plans"]
        opts = [{"o": pn, "K": p["avg_bits"], "bytes": int(a.vocab * 5120 * p["avg_bits"] / 8 + 2 ** 20), "err": p["kl"], "fam": "v3head",
                 "src": {"path": a.head_donor.format(plan=pn, donor=p.get("donor", pn))}} for pn, p in hc.items()]
    else:
        opts = [{"o": "base", "K": None, "bytes": byw.get(MN.HEAD, 0), "err": 1.0, "fam": "v3head"}]
    tens.append({"w": MN.HEAD, "layer": 64, "group": "head", "hbar": None, "opts": opts})
    tens.append({"w": MN.EMB, "layer": -1, "group": "emb", "hbar": None, "opts": [{"o": "base", "K": None, "bytes": byw.get(MN.EMB, 0), "err": 0.0, "fam": "emb"}]})
    out = {"params": MN.PARAMS, "fixed_bytes": fixed, "n_layers": 64, "groups": MN.GROUPS, "tensors": tens,
           "built": {"levels": a.levels, "base": a.base, "diag": dd, "heads": a.heads}}
    json.dump(out, open(a.out + ".part", "w")); os.replace(a.out + ".part", a.out)
    M = MN.Menu(out)
    print(f"{a.out}: {len(tens)} tensors, {len(a.levels)} levels, fixed {fixed} bytes; uniform: " +
          ", ".join(f"{n.split('-')[-2]} {M.bpw(M.uniform(n, 'base' if not a.heads else None, 'base')):.3f}" for n in M.levels()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
