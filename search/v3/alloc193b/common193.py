# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""common193.py -- #193 shared pieces: the RULES-193 slice allow-list, snapshot loaders, menu helpers, tensor groups.

Every loader filters by ALLOWED (RULES-193 §0); rows of any other slice are dropped and counted."""
import json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import menu as MN  # noqa: E402

# w181 (RULES-193 amendment 12): every slice retired at the 11-15 6:26 PM CST rotation + the prior rows. s190f joins
# (env A193_S190F=1) only once its pending confirmations are resolved. Never: s195*, t183, t186n, ver*, final, public.
ALLOWED = {"s173a", "s173b", "s173c", "s173d", "s173e", "s173f", "s186a", "s186b", "s186c", "s186d", "s186e", "s186f",
           "s190a", "s190b", "s190c", "s190d", "s190e", "fit129"} | ({"s190f"} if os.environ.get("A193_S190F") == "1" else set())
SIZES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
BAND = {2.0: "b1", 2.5: "b1", 3.0: "b2", 3.5: "b2", 4.0: "b2", 4.5: "b3", 5.0: "b3", 6.0: "b3"}
TYPES = ["attn_qg", "attn_k", "attn_v", "attn_output", "ssm_qkv", "ssm_z", "ssm_out", "ffn_gate_up", "ffn_down"]
dropped = {}


def ok_slice(s):
    s = (s or "").split("|")[0]
    if s in ALLOWED: return True
    dropped[s] = dropped.get(s, 0) + 1
    return False


def load_state(snap):
    return [json.loads(l) for l in open(f"{snap}/state.jsonl")]


def load_menu(snap):
    return MN.Menu.load(f"{snap}/menu-v2.json")


def fit_rows(snap, groups=("b1", "b2", "b3")):
    """the driver's fit inputs (latest per band) with non-allowed labels dropped; dedup by (tag, label)"""
    out, seen = [], set()
    for g in groups:
        d = json.load(open(f"{snap}/fit-{g}.in.json"))
        for o in d["obs"]:
            k = (o["tag"], o["label"])
            if k in seen or not ok_slice(o["label"]): continue
            seen.add(k); o["band"] = g; out.append(o)
    return out


def assign(snap, tag):
    return json.load(open(f"{snap}/assign/{tag}.json"))


def group_of(w, nl=64):
    """(type, depth band 0..7) of a trunk tensor; ('head', 8) / ('emb', -1)"""
    if w == MN.HEAD: return ("head", 8)
    if w == MN.EMB: return ("emb", -1)
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", w)
    return (m.group(2), int(m.group(1)) * 8 // nl)


def gname(w):
    t, b = group_of(w)
    return t if t in ("head", "emb") else f"{t}@{b}"


class KM:
    """per-weight option tables of the menu: K / bytes / name, sorted by bytes, family-restricted like consensus192"""

    def __init__(self, M):
        self.M = M; self.W = list(M.W)
        self.opt = {t["w"]: t["opts"] for t in M.T}

    @staticmethod
    def fam(o):
        if o.startswith("q27v3"): return "mad"
        if o == "bf16": return "bf16"
        return o[0] + ("m" if o.endswith("m") else "")

    def K(self, w, o):
        return next(x["K"] for x in self.opt[w] if x["o"] == o)

    def Kvec(self, a):
        return {w: self.Kof[w][a[w]] for w in self.W}

    def fam_opts(self, w, f):
        return sorted(((x["K"], x["o"], x["bytes"]) for x in self.opt[w] if self.fam(x["o"]) == f), key=lambda z: (z[2], z[0]))

    @property
    def Kof(self):
        if not hasattr(self, "_Kof"): self._Kof = {w: {x["o"]: x["K"] for x in self.opt[w]} for w in self.W}
        return self._Kof


def window(M, size, tol=0.02):
    """driver.window: bytes [size - tol, size] (no clamp needed for 2.0-6.0)"""
    return (size - tol) * M.P / 8, size * M.P / 8


def check_file(M, d):
    """the driver's inbox_spec byte / option checks (no queue state) -> reason or None"""
    a = d["assign"]
    for w, o in a.items():
        if w not in M.WI: return f"unknown weight {w}"
        if o not in M.oi[M.WI[w]]: return f"unknown option {o} for {w}"
    if any(w not in a for w in M.W): return "unassigned weight"
    if d["mode"] == "decide":
        lo, hi = window(M, float(d["size"]))
        if not lo <= M.bytes(a) <= hi: return f"bpw {M.bpw(a):.4f} outside window"
    elif abs(M.bpw(a) - float(d["size"])) > 0.02: return "probe bpw off"
    return None
