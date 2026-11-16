# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""flow181.py upd | queue N GENDIR | tally -- w181 (RULES-193 amendment 13) round bookkeeping on the VM.

  upd    : verdicts of my own files (live state: accept / join / reject; identities and verdicts only) -> adds the flag
           of a confirmed base `anom` file to data/alloc193b/confirmed.json (follow-up variants next round)
  queue  : rebuild workers/w181/queue.txt from GENDIR/{anom,rr,own}: drops assignments already in the run / inbox,
           (flag, variant) pairs with 0 confirmed in >= 6 decided, `rev` beyond 8 files in all, xplant other than
           lo / hi / mid; a taken name gets -rN; order anom : rankreg : anom : rankreg : xplant (anom round-robin over
           variants, rankreg by rank with rankanom interleaved). Appends the anom files' meta to data/alloc193b/meta.jsonl.
  tally  : per allocator / variant decided, tentative, confirmed (for the reports)"""
import collections, glob, hashlib, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = "/workspace/rad-local/workers/w181"; D = "/workspace/rad-local/workers/lead-srch/run/drv"
DATA = os.path.join(HERE, "..", "..", "..", "data", "alloc193b")
BAND = {2.0: "b1", 2.5: "b1", 3.0: "b2", 3.5: "b2", 4.0: "b2", 4.5: "b3", 5.0: "b3", 6.0: "b3"}


def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def verdicts(pfx="w181-"):
    """file -> {tag, size, s1: reject|tentative|stale|None, s2: accept|join|reject|None}"""
    st = jl(f"{D}/state.jsonl"); out = {}
    tent = {e["tag"] for e in st if e.get("ev") == "tentative"}
    d1 = {e["tag"]: e["decision"] for e in st if e.get("ev") == "decision" and e.get("stage") != 2}
    d2 = {e["tag"]: e["decision"] for e in st if e.get("ev") == "decision" and e.get("stage") == 2}
    for e in st:
        if e.get("ev") == "inbox" and e["file"].startswith(pfx):
            t = e["tag"]; out[e["file"]] = {"tag": t, "size": float(e["size"]), "s1": "tentative" if t in tent else d1.get(t), "s2": d2.get(t)}
    return out


def meta():
    return {m["name"]: m for m in jl(f"{DATA}/meta.jsonl")}


def var_of(name, M):
    """(allocator, variant, flag) of a ledger / inbox file name (without the w181- prefix and .json)"""
    al = name.split("-")[0]
    if al == "anom" and name in M: return al, M[name]["var"], M[name]["flag"]
    if al == "xplant": return al, name.split("-")[2], None
    return al, "", None


def tally():
    V = verdicts(); M = meta(); T = collections.OrderedDict()
    for f, v in V.items():
        n = f[5:-5]; al, var, fl = var_of(n, M)
        for key in dict.fromkeys(((al, "", ""), (al, var, ""), (al, var, fl or ""))):
            if v["s1"] in (None, "stale"): continue
            r = T.setdefault(key, [0, 0, 0, 0, 0]); r[0] += 1
            if v["s1"] == "tentative":
                r[1] += 1
                if v["s2"] == "accept": r[2] += 1
                elif v["s2"] == "join": r[3] += 1
                elif v["s2"] is None: r[4] += 1
    return T


def upd():
    V = verdicts(); M = meta(); p = f"{DATA}/confirmed.json"; Cf = json.load(open(p)); n = 0
    for f, v in V.items():
        m = M.get(f[5:-5])
        if not m or v["s2"] not in ("accept", "join"): continue
        b = BAND[m["size"]]
        if m["var"] == "s2":                                          # amendment 16: s3 after a confirmed s2
            for c in Cf:
                if c["groups"] == m["groups"] and c["band"] == b and not c.get("s2ok"): c["s2ok"] = True; n += 1
            continue
        if m["var"] not in ("base", "xb", "nbm", "nbp", "nb2"): continue
        if any(c["groups"] == m["groups"] and c["band"] == b for c in Cf): continue
        Cf.append({"groups": m["groups"], "band": b, "dir": m["dir"], "src": f"{f[:-5]} ({v['s2']})"}); n += 1
    if n: json.dump(Cf, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    print(f"confirmed flags: {len(Cf)} ({n} new)")


def h(p):
    d = json.load(open(p)); return hashlib.sha1(json.dumps(d.get("assign", d), sort_keys=True).encode()).hexdigest()


def queue(rnd, gen):
    seen = {h(p) for p in glob.glob(D + "/assign/*.json")} | {h(p) for p in glob.glob(D + "/inbox/*.json")}
    names = {x["name"] for x in jl(f"{DATA}/submitted.jsonl")}; M = meta(); T = tally()
    nrev = sum(1 for n in names if n.endswith("-rev.json") or "-rev-r" in n)
    newm = {m["name"]: m for m in jl(f"{gen}/anom/meta.jsonl")}
    dead = {(k[1], k[2]) for k, r in T.items() if k[0] == "anom" and k[2] and r[0] - r[4] >= 6 and r[2] + r[3] == 0}
    stage = f"{gen}/stage"; os.makedirs(stage, exist_ok=True); by = collections.defaultdict(list); drop = collections.Counter()
    def take(p, key):
        b = os.path.basename(p); hh = h(p)
        if hh in seen: drop["known"] += 1; return None
        if "w181-" + b in names: b = b[:-5] + f"-r{rnd}.json"
        if "w181-" + b in names: drop["name"] += 1; return None
        seen.add(hh); s = f"{stage}/{b}"; json.dump(json.load(open(p)), open(s, "w")); by[key].append(s); return b[:-5]
    A = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{gen}/anom/*.json")):
        m = newm[os.path.basename(p)[:-5]]
        if (m["var"], m["flag"]) in dead: drop["dead"] += 1; continue
        A[m["var"]].append((p, m))
    import random
    for v in A: random.Random(193 + int(rnd)).shuffle(A[v])
    mout = open(f"{DATA}/meta.jsonl", "a")
    order = ["cmb", "s3", "base", "s2", "nb2", "xb", "nbm", "nbp", "rev"]
    # amendment 16: the standing drop rule per token (0 confirmed in >= 8 decided, pending not counted)
    deadtok = {k[0] for k, r in T.items() if not k[1] and not k[2] and r[0] - r[4] >= 8 and r[2] + r[3] == 0}
    for t in itertools.zip_longest(*[A[v] for v in order if A[v]]):
        for x in t:
            if not x: continue
            p, m = x
            if m["var"] == "rev":
                if nrev >= 8: drop["rev cap"] += 1; continue
            n = take(p, "anom")
            if n:
                nrev += m["var"] == "rev"; mout.write(json.dumps(dict(m, name=n)) + "\n")
    mout.close()
    rr = sorted(glob.glob(f"{gen}/rr/rank*.json"), key=lambda p: (int(p[:-5].split("-")[-1]), "rankanom" in p, p))
    for p in rr:
        if os.path.basename(p).split("-")[0] in deadtok: drop["dead token"] += 1; continue
        take(p, "rank")
    for p in sorted(glob.glob(f"{gen}/own/xplant-*.json")):
        if p[:-5].split("-")[-1] in ("lo", "hi", "mid") and "xplant" not in deadtok: take(p, "xplant")
    its = {k: iter(v) for k, v in by.items()}; fin = []
    while True:
        got = False
        for k in ("anom", "rank", "anom", "rank", "xplant"):
            x = next(its.get(k, iter(())), None)
            if x: fin.append(x); got = True
        if not got: break
    tmp = W + "/queue.txt.tmp"; open(tmp, "w").write("".join(p + "\n" for p in fin)); os.replace(tmp, W + "/queue.txt")
    print(f"round {rnd}: queue {len(fin)}: " + ", ".join(f"{k} {len(v)}" for k, v in by.items()) + f"; dropped {dict(drop)}; dead {sorted(dead)}; dead tokens {sorted(deadtok)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "upd": upd()
    elif a[0] == "queue": queue(a[1], a[2])
    elif a[0] == "tally":
        for k, r in tally().items(): print(k, r)
