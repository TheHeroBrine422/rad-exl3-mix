# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""menu.py -- the search's option menu (w126, #166): per mixable tensor, the options a mix may give it.

File (JSON):
  {"params": 26895998464, "fixed_bytes": B0,          bytes of everything outside the menu (norms, small tensors, ...)
   "groups": [...],                                   kfit3's groups (attn_in, attn_out, ssm_in, ssm_out, ffn_gu, ffn_down)
   "tensors": [{"w": "blk.0.ssm_qkv.weight", "layer": 0, "group": "ssm_in", "hbar": h | null,
                "opts": [{"o": OPTION, "K": k, "bytes": b, "err": e, "fam": "v3i8", "src": {...}}, ...]}, ...]}
  "o" names the option: a ladder level (trunk tensors: the eval line's mix.levels value) or a donor key (head /
  embedding: mix.donors[w] = src["path"]). err = the absolute input-metric error kfit3 uses (proxy_err x R summed over
  the tensor's HF parts); bytes = the option's stored bytes (heads: the quantized head's bytes, even when the eval
  splices its bf16-dequantized rows; TODO "Heads before w124's un-permute"). Options sorted by bytes.
  The embedding / head are tensors like the others (layer = n_layers for the head; group "head" / "emb").
Assignments are {w: option name}; bytes(assign) = fixed_bytes + sum of the picks' bytes; bpw = bytes * 8 / params.

  python3 menu.py synth OUT.json [--layers 64 --levels 1,1.5,2,...]   a synthetic menu (stand-in tests)
  python3 menu.py show MENU.json                                         sizes of the uniform-level mixes"""
import json, math, re, sys

PARAMS = 26895998464
GROUPS = ["attn_in", "attn_out", "ssm_in", "ssm_out", "ffn_gu", "ffn_down"]
TYPE2G = {"attn_qg": "attn_in", "attn_k": "attn_in", "attn_v": "attn_in", "attn_output": "attn_out", "ssm_qkv": "ssm_in",
          "ssm_z": "ssm_in", "ssm_out": "ssm_out", "ffn_gate_up": "ffn_gu", "ffn_down": "ffn_down"}
HEAD, EMB = "output.weight", "token_embd.weight"


def is_full_attention(l): return l % 4 == 3


def trunk_names(n_layers=64):
    """rxq.mix.trunk_names (the container's trunk weight names, in order)"""
    out = []
    for l in range(n_layers):
        if is_full_attention(l): out += [f"blk.{l}.{t}.weight" for t in ("attn_qg", "attn_k", "attn_v", "attn_output")]
        else: out += [f"blk.{l}.{t}.weight" for t in ("ssm_qkv", "ssm_z", "ssm_out")]
        out += [f"blk.{l}.ffn_gate_up.weight", f"blk.{l}.ffn_down.weight"]
    return out


def group_of(w):
    if w == HEAD: return "head"
    if w == EMB: return "emb"
    return TYPE2G[re.match(r"blk\.\d+\.(\w+)\.weight$", w).group(1)]


def layer_of(w, n_layers=64):
    m = re.match(r"blk\.(\d+)\.", w)
    return int(m.group(1)) if m else n_layers


class Menu:
    def __init__(self, d):
        self.d = d; self.P = d.get("params", PARAMS); self.fixed = d.get("fixed_bytes", 0)
        self.T = d["tensors"]; self.W = [t["w"] for t in self.T]; self.WI = {w: i for i, w in enumerate(self.W)}
        for t in self.T: t["opts"].sort(key=lambda o: o["bytes"])
        self.oi = [{o["o"]: j for j, o in enumerate(t["opts"])} for t in self.T]
        self.NL = d.get("n_layers", 1 + max(t["layer"] for t in self.T if t["w"].startswith("blk.")))

    @classmethod
    def load(cls, p): return cls(json.load(open(p)))

    def opt(self, w, name): i = self.WI[w]; return self.T[i]["opts"][self.oi[i][name]]

    def bytes(self, a): return self.fixed + sum(self.opt(w, o)["bytes"] for w, o in a.items())

    def bpw(self, a): return self.bytes(a) * 8 / self.P

    def uniform(self, name, head=None, emb=None):
        """every trunk tensor at option `name` (a level); head / embedding at their given or cheapest option"""
        a = {}
        for t in self.T:
            names = [o["o"] for o in t["opts"]]
            if t["w"] == HEAD: a[t["w"]] = head if head else names[0]
            elif t["w"] == EMB: a[t["w"]] = emb if emb else names[0]
            elif name in names: a[t["w"]] = name
            else: raise KeyError(f"{t['w']} has no option {name}")
        return a

    def levels(self):
        """trunk option names present on every trunk tensor, by bytes"""
        tr = [t for t in self.T if t["w"] not in (HEAD, EMB)]
        common = set(o["o"] for o in tr[0]["opts"])
        for t in tr[1:]: common &= {o["o"] for o in t["opts"]}
        return sorted(common, key=lambda n: sum(self.opt(t["w"], n)["bytes"] for t in tr))

    def eval_mix(self, a, base):
        """the rxrun eval line's "mix" for assignment a: trunk -> levels, head / embedding -> donors"""
        lv, dn = {}, {}
        for w, o in a.items():
            src = self.opt(w, o).get("src") or {}
            if src.get("path"): dn[w] = src["path"]
            elif w in (HEAD, EMB): continue                     # no donor: the base container's own tensor
            else: lv[w] = o
        return {"base": base, "levels": lv, "donors": dn}


def synth(n_layers=16, levels=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0), seed=0, heads=(3.0, 4.0, 5.0, 6.0), embs=(3.0, 4.0, 5.0)):
    """a synthetic menu with the real tensor shapes' proportions (scaled down to n_layers): err(K) ~ R 2^(-2K) x a
    per-tensor jitter, bytes ~ numel K / 8 + overhead"""
    import random
    rng = random.Random(seed)
    numel = {"attn_qg": 5120 * 12288, "attn_k": 5120 * 1024, "attn_v": 5120 * 1024, "attn_output": 6144 * 5120,
             "ssm_qkv": 5120 * 10240, "ssm_z": 5120 * 6144, "ssm_out": 6144 * 5120, "ffn_gate_up": 5120 * 34816, "ffn_down": 17408 * 5120}
    T = []
    for w in trunk_names(n_layers):
        typ = re.match(r"blk\.\d+\.(\w+)\.weight$", w).group(1); n = numel[typ]; R = n * (0.5 + rng.random())
        jit = math.exp(rng.gauss(0, 0.15))
        T.append({"w": w, "layer": layer_of(w), "group": group_of(w), "hbar": math.exp(rng.gauss(0, 0.5)),
                  "opts": [{"o": f"L{k:g}", "K": k, "bytes": int(n * k / 8 + 2 * 5120 * 2), "err": R * jit * 2 ** (-1.6 * k) * math.exp(rng.gauss(0, 0.03)),
                            "fam": "v3i8"} for k in levels]})
    vocab, dm = int(248320 * n_layers / 64), 5120                     # head / embedding scaled with the trunk
    T.append({"w": HEAD, "layer": n_layers, "group": "head", "hbar": None,
              "opts": [{"o": f"H{k:g}", "K": k, "bytes": int(vocab * dm * k / 8), "err": vocab * dm * 0.7 * 2 ** (-1.6 * k), "fam": "v3head",
                        "src": {"path": f"/heads/H{k:g}.rad"}} for k in heads]})
    T.append({"w": EMB, "layer": -1, "group": "emb", "hbar": None,
              "opts": [{"o": f"E{k:g}", "K": k, "bytes": int(vocab * dm * k / 8), "err": 0.0, "fam": "emb", "src": {"path": f"/emb/E{k:g}.rad"}} for k in embs]})
    # params scaled with the trunk: the real model's ratio of trunk to total
    tb = sum(numel[re.match(r"blk\.\d+\.(\w+)\.weight$", t["w"]).group(1)] for t in T[:-2])
    return {"params": tb + 2 * vocab * dm, "fixed_bytes": 2 * 1024 * 1024, "n_layers": n_layers, "groups": GROUPS, "tensors": T}


def main(argv):
    if argv[0] == "synth":
        kw = {}
        for i in range(2, len(argv), 2):
            if argv[i] == "--layers": kw["n_layers"] = int(argv[i + 1])
            elif argv[i] == "--levels": kw["levels"] = tuple(float(x) for x in argv[i + 1].split(","))
        json.dump(synth(**kw), open(argv[1], "w"))
    elif argv[0] == "show":
        M = Menu.load(argv[1])
        for n in M.levels(): print(f"{n:40s} {M.bpw(M.uniform(n)):.3f} bpw")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
