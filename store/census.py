#!/usr/bin/env python3
"""census.py CONTAINER.rad [--mix MIXFILE] [--recipe RECIPE] [--json OUT] -- what a container holds, weight by weight:
scheme (exl3 / exl3i / plain / affine ...), the trellis rate K and codebook, the activation mode (metadata key
exl3.act.<weight>, f16 when absent), the bytes, and the real bpw (bytes x 8 / 26,895,998,464). design/v3.md §1c.4: the
container must be what the mix says before a KLD means anything.

--mix: radmix's mix file ("<weight> <donor.rad>", "act <weight> <mode>"). Each named weight is checked byte-for-byte
       against its donor's entry (encoding struct, planes' bytes -- the same identity as store/scan.py's vid), and each
       act line against the container's metadata; weights the mix does not name must equal the BASE given by --base.
--recipe: rad-convert's recipe ("PATTERN QUANTISER key=value ..."; "@" = the one stored in the container): each weight's quantiser and, for exl3 / exl3i,
       rate= / bits= / codebook= checked against the stored encoding (first matching pattern wins, as rad-convert).
Exit 1 on any mismatch. Prints a summary table by (scheme, K, codebook, act)."""
import collections, fnmatch, hashlib, json, os, struct, sys

HDR = struct.Struct("<IIQ5Q" + "Q" * 18)
ENT = struct.Struct("<QQQII6qiiiIQQQ2Q")
KV = struct.Struct("<QIIQ")
ENC_SIZE = 24 + 24 + 4 + 4 + 8 * 64
PLANE = struct.Struct("<24sIIqqqq")
assert ENT.size == 136 and PLANE.size == 64 and KV.size == 24
MAGIC, VER, CH = 0x31444152, 2, 1 << 24
UK = {22: 1, 20: 2, 24: 3, 19: 4, 26: 5, 28: 6, 30: 7, 9: 8}           # RAD_U1..RAD_U8 (abi/rad_types.h) -> K
DT = {1: "f32", 2: "f16", 3: "bf16", 8: "i8", 9: "u8", 13: "u32", 37: "u16", **{k: f"u{v}" for k, v in UK.items()}}
PARAMS = 26_895_998_464


def cstr(S, o): return "" if o == 0 else S[o:S.index(b"\0", o)].decode()


class Rad:
    def __init__(self, path):
        self.path = path; self.f = open(path, "rb")
        h = HDR.unpack(self.f.read(HDR.size))
        if h[0] != MAGIC or h[1] != VER: sys.exit(f"{path}: not RAD2")
        self.size = os.path.getsize(path)
        (so, sn, mo, mn, do, dn, po, pn, eo, en) = h[8:18]
        rd = lambda o, n: (self.f.seek(o), self.f.read(n))[1]
        S, D, P, E = rd(so, sn), rd(do, dn * ENT.size), rd(po, pn * 16), rd(eo, en * ENC_SIZE)
        M = rd(mo, mn * KV.size)
        self.recipe = cstr(S, h[6])
        self.meta = {}
        for i in range(mn):
            k, t, _, v = KV.unpack_from(M, i * KV.size)
            self.meta[cstr(S, k)] = cstr(S, v) if t == 2 else v
        self.w = {}
        for i in range(dn):
            e = ENT.unpack_from(D, i * ENT.size)
            eb = E[e[3] * ENC_SIZE:(e[3] + 1) * ENC_SIZE]
            planes = [struct.unpack_from("<QQ", P, (e[15] + p) * 16) for p in range(e[14])]
            self.w[cstr(S, e[0])] = dict(quant=cstr(S, e[1]), opts=cstr(S, e[2]), shape=list(e[5:5 + e[4]]), enc=eb,
                                         planes=planes, bytes=sum(n for _, n in planes), offset=e[16])

    def ident(self, name):   # store/scan.py's vid: name, shape, encoding struct, each plane's bytes
        w = self.w[name]; hs = hashlib.sha256(json.dumps([name, w["shape"]]).encode()); hs.update(w["enc"])
        for po, pn in w["planes"]:
            hs.update(struct.pack("<QQ", po - w["offset"], pn)); self.f.seek(po); left = pn
            while left:
                b = self.f.read(min(CH, left)); hs.update(b); left -= len(b)
        return hs.hexdigest()[:32]


def describe(eb):
    """(scheme, K, codebook, detail) of a stored encoding"""
    scheme = eb[:24].split(b"\0")[0].decode(); tr = eb[24:48].split(b"\0")[0].decode()
    n = struct.unpack_from("<I", eb, 48)[0]
    pl = [PLANE.unpack_from(eb, 56 + 64 * i) for i in range(n)]
    roles = [p[0].split(b"\0")[0].decode() for p in pl]
    if scheme == "exl3i":
        cb, _, k = tr.partition("/h128/k")
        return scheme, k, cb, "+".join(roles)
    if scheme == "exl3":
        ci = roles.index("codes") if "codes" in roles else 0
        return scheme, str(UK.get(pl[ci][1], "?")), tr.split("/")[0], "+".join(roles)
    dt = DT.get(pl[0][1], str(pl[0][1])) if pl else "?"
    return scheme, "", "", f"{tr or '-'} {dt}"


def main():
    a = sys.argv[1:]
    def opt(k):
        if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
    mix, recipe, js, base = opt("--mix"), opt("--recipe"), opt("--json"), opt("--base")
    R = Rad(a[0]); bad = []
    rows = []
    for name, w in R.w.items():
        sch, k, cb, det = describe(w["enc"])
        act = R.meta.get(f"exl3.act.{name}", "f16") if sch in ("exl3", "exl3i") else ""
        rows.append(dict(name=name, scheme=sch, K=k, codebook=cb, act=act, planes=det, bytes=w["bytes"],
                         quantizer=w["quant"], options=w["opts"]))
    by = {r["name"]: r for r in rows}
    if recipe:   # first matching rule wins (rad-convert)
        text = R.recipe if recipe == "@" else open(recipe).read()
        rules = [l.split() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        for r in rows:
            rule = next((x for x in rules if fnmatch.fnmatchcase(r["name"], x[0])), None)
            if not rule: continue
            kv = dict(t.split("=", 1) for t in rule[2:] if "=" in t)
            if r["quantizer"] and r["quantizer"] != rule[1]: bad.append(f"{r['name']}: quantiser {r['quantizer']}, recipe {rule[1]}")
            if rule[1] in ("exl3", "exl3i"):
                if r["scheme"] != rule[1]: bad.append(f"{r['name']}: scheme {r['scheme']}, recipe {rule[1]}")
                want_k = kv.get("rate", kv.get("bits"))
                if want_k and r["K"] != want_k: bad.append(f"{r['name']}: K {r['K']}, recipe {want_k}")
                if "codebook" in kv and r["codebook"] != kv["codebook"]: bad.append(f"{r['name']}: codebook {r['codebook']}, recipe {kv['codebook']}")
    if mix:
        donors = {}; acts = {}; named = set()
        for l in open(mix):
            t = l.split("#", 1)[0].split()
            if len(t) == 3 and t[0] == "act": acts[t[1]] = t[2]
            elif len(t) == 2: named.add(t[0]); donors.setdefault(t[1], []).append(t[0])
        for d, names in donors.items():
            D_ = Rad(d)
            for n in names:
                if n not in R.w: bad.append(f"{n}: named by the mix, not in the container"); continue
                if n not in D_.w: bad.append(f"{n}: not in donor {d}"); continue
                if R.ident(n) != D_.ident(n): bad.append(f"{n}: differs from donor {os.path.basename(d)}")
        for n, m in acts.items():
            if R.meta.get(f"exl3.act.{n}") != m: bad.append(f"{n}: act {R.meta.get(f'exl3.act.{n}')}, mix {m}")
        if base:
            B = Rad(base)
            for n in R.w:
                if n in named: continue
                if n not in B.w or R.ident(n) != B.ident(n): bad.append(f"{n}: not named by the mix and differs from base")
    # summary
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["scheme"], r["K"], r["codebook"], r["act"]); agg[key][0] += 1; agg[key][1] += r["bytes"]
    print(f"{os.path.basename(a[0])}: {len(rows)} weights, {R.size * 8 / PARAMS:.4f} bpw (file), "
          f"{sum(r['bytes'] for r in rows) * 8 / PARAMS:.4f} bpw (weights)")
    print(f"  {'scheme':<8} {'K':>6} {'codebook':<8} {'act':<8} {'n':>4} {'GB':>8}")
    for (s, k, c, ac), (n, b) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"  {s:<8} {k:>6} {c:<8} {ac:<8} {n:>4} {b / 1e9:>8.3f}")
    if js: json.dump(dict(container=a[0], bytes=R.size, weights=rows, mismatches=bad), open(js, "w"), indent=1)
    for b in bad[:40]: print("MISMATCH", b)
    if len(bad) > 40: print(f"... {len(bad) - 40} more")
    checked = " ".join(x for x, v in (("recipe", recipe), ("mix", mix), ("base", base)) if v)
    print("CENSUS", "PASS" if not bad else f"FAIL ({len(bad)})", f"[checked: {checked or 'nothing (summary only)'}]")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
