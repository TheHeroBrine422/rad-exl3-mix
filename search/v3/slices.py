# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""slices.py -- the search's slice registry (w126, #166; docs/design/eval-data.md "two-part, calib-sparing", w123's
protocol: rotate every ~25 decisions, fresh xcorp each rotation, ~50 calib docs drawn from the search budget).

A slice = an x part (fresh xcorp pieces, never reused) + a c part (calib search-pool docs), both built by
eval/slices/mkslice27.sh (or any golden_pick / pickx output) on the host that has the source goldens. For scoring, the
parts are merged into ONE golden in a fixed seeded order with x / c interleaved (goldmix): each document keeps its
fields and gets "part" and "src" (its index in the part); radiance scores the documents in that order, so its
--kld-progress lines arrive interleaved and the early-stop statistic sees both parts from the start.

Registry (JSON, REG): {"slices": {NAME: {"parts": {"x": DIR, "c": DIR}, "mix": DIR, "docs": [{"part", "src", "id",
"tokens", "rows"}], "status": "ready" | "active" | "retired", "decisions": {size: n}, "built", "activated",
"retired"}}, "doc_decisions": {doc id: n}}; doc ids: "c:<calib doc index>" (the golden's "orig"), "x:<source>:<idx>".
  python3 slices.py goldmix OUT SEED x=GOLDEN_X c=GOLDEN_C      the merged, interleaved golden (on the eval host);
                                                                 NAME=DIR@even|@odd|@i,j,..: some documents of DIR
  python3 slices.py add REG NAME MIXDIR [x=DIR c=DIR]           register a built slice (reads MIXDIR/kld.json)
  python3 slices.py show REG
  python3 slices.py ledger REG NAME OUTDIR                      the c part's kld.json for ledger_add.py (role: select)
  python3 slices.py draw REG LEDGER POOLS N SEED [MODE]          N calib docs for the next c part. MODE merged (default,
          user 11-13: fit + search are one pool, eval-data.md): every fit- or search-pool doc, LEAST-USED FIRST by
          the ledger's decision counts (sum of each use's "decisions"; a use without a number counts 1; the
          head-input collections w94 / w119-headx-* count 0: allowed) + this registry's own decisions on it, ties
          random; old modes (search pool only, ledger-fresh docs only): never | no27b | headx. Never a doc of a ready /
          active slice. Prints the doc indices (golden_pick.py's argument list); the c golden: golden_pick.py GOLD
          CORPUS OUT JSONL DOCS
  python3 slices.py poolcheck POOLS GOLDEN [GOLDEN ..]           the pools of each golden's calib docs ("orig"); exit 1
          if any is outside fit + search (a search slice / prior_obs golden must come from the merged pool)
Rotation (driver): a slice is retired once any size has decided `rotate_after` candidates on it (default 25); the
next "ready" slice becomes active (its incumbents are scored first, no early stop)."""
import json, os, random, shutil, sys, time

import numpy as np

KEYS = ("format", "model", "top", "n_vocab")


def interleave(nx, nc):
    N = nx + nc
    return ["x" if (k * nx) // N > ((k - 1) * nx) // N else "c" for k in range(1, N + 1)]


def goldmix(out, parts, seed, select=None):
    """parts: [(name, dir)] (two: x, c). Each part's documents in a seeded random order, merged by interleave().
    select: {name: [doc indices]} -- a part made of some documents of its golden (dry runs: two pseudo-parts of one)"""
    ms = {n: (g, json.load(open(f"{g}/kld.json"))) for n, g in parts}
    for k in KEYS:
        if len({json.dumps(m.get(k)) for _, m in ms.values()}) != 1: raise SystemExit(f"goldens differ in {k}")
    rng = random.Random(seed)
    sel = {n: list((select or {}).get(n) or range(len(m["docs"]))) for n, (g, m) in ms.items()}
    perm = {n: rng.sample(sel[n], len(sel[n])) for n in ms}
    names = [n for n, _ in parts]
    if len(names) == 2: pat = interleave(len(perm[names[0]]), len(perm[names[1]])); pat = [names[0] if p == "x" else names[1] for p in pat]
    else: pat = [n for n in names for _ in perm[n]]
    o = out + ".part"; shutil.rmtree(o, ignore_errors=True); os.makedirs(o)
    m0 = next(iter(ms.values()))[1]; top = m0["top"]
    toff = {}
    for n, (g, m) in ms.items():
        c = np.cumsum([0] + [d["tokens"] for d in m["docs"]]); toff[n] = c
    fh = {f: open(f"{o}/{f}", "wb") for f in ("tokens.i32", "gen.u8", "top.bin", "rows.f32")}
    src = {n: {f: open(f"{g}/{f}", "rb") for f in fh} for n, (g, m) in ms.items()}
    has_gen = {n: os.path.exists(f"{g}/gen.u8") for n, (g, _) in ms.items()}
    rw = {n: os.path.getsize(f"{g}/rows.f32") // max(m["rows"], 1) for n, (g, m) in ms.items()}
    if len(set(rw.values())) != 1: raise SystemExit("rows.f32 widths differ")
    rw = next(iter(rw.values()))
    docs, r0, it = [], 0, {n: iter(perm[n]) for n in names}
    for p in pat:
        i = next(it[p]); g, m = ms[p]; d = dict(m["docs"][i]); rows = d["tokens"] - 1 - d["score_from"]
        def cp(f, off, nb):
            s = src[p][f]; s.seek(off); fh[f].write(s.read(nb))
        cp("tokens.i32", int(toff[p][i]) * 4, d["tokens"] * 4)
        if has_gen[p]: cp("gen.u8", d["row0"], rows)
        else: fh["gen.u8"].write(np.ones(rows, np.uint8).tobytes())
        cp("top.bin", d["row0"] * top * 8, rows * top * 8)
        cp("rows.f32", d["row0"] * rw, rows * rw)
        d.update({"part": p, "src": i, "row0": r0}); docs.append(d); r0 += rows
    for f in fh.values(): f.close()
    for s in src.values():
        for f in s.values(): f.close()
    man = {k: m0[k] for k in m0 if k not in ("docs", "rows")}
    man.update({"rows": r0, "docs": docs, "goldmix": {"parts": {n: g for n, g in parts}, "seed": seed, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}})
    json.dump(man, open(f"{o}/kld.json", "w"), indent=0)
    shutil.rmtree(out, ignore_errors=True); os.replace(o, out)
    return man


MERGED = ("fit", "search")                       # one pool since 2026-11-13 (user; eval-data.md)
HEADX = ("w94-headx-search", "w94-headx-qad", "w119-headx-search", "w119-headx-fit", "w119-headx-verify")


def ledger_use(led, mine=()):
    """per calib doc: its decision count by the ledger (numeric "decisions" summed; a use without a number counts 1;
    the head-input collections count 0; uses by this registry's own slices skipped -- the registry counts those)"""
    use = {}
    for l in open(led):
        r = json.loads(l)
        if r["doc"] == "all" or r["use"] in mine or r["use"] in HEADX: continue
        d = r.get("decisions"); k = d if isinstance(d, (int, float)) and not isinstance(d, bool) else 1
        use[int(r["doc"])] = use.get(int(r["doc"]), 0) + k
    return use


def draw_merged(R, led, pool, n, seed):
    """N fit + search docs, least-used first (ledger decisions + this registry's own decisions), ties random"""
    mine = {f"golden-{k}-c" for k in R.d["slices"]}; lu = ledger_use(led, mine); dd = R.d["doc_decisions"]
    busy = {int(d["id"][2:]) for s_ in R.d["slices"].values() if s_["status"] in ("ready", "active") for d in s_["docs"] if d["id"].startswith("c:")}
    cand = [i for i, p in enumerate(pool) if p in MERGED and i not in busy]
    use = {i: lu.get(i, 0) + dd.get(f"c:{i}", 0) for i in cand}
    rng = random.Random(seed); cand.sort(key=lambda i: (use[i], rng.random()))
    return sorted(cand[:n]), len(cand), use


def doc_id(d):
    if d["part"] == "c": return f"c:{d.get('orig', d['src'])}"
    return f"x:{d.get('source', '')}:{d.get('orig', d['src'])}"


class Registry:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {"slices": {}, "doc_decisions": {}}

    def save(self):
        json.dump(self.d, open(self.path + ".part", "w"), indent=1); os.replace(self.path + ".part", self.path)

    def add(self, name, mixdir, parts=None, man=None, host_dirs=None):
        man = man or json.load(open(f"{mixdir}/kld.json"))
        docs = [{"part": d["part"], "src": d["src"], "id": doc_id(d), "tokens": d["tokens"], "rows": d["tokens"] - 1 - d["score_from"]} for d in man["docs"]]
        self.d["slices"][name] = {"parts": parts or man.get("goldmix", {}).get("parts", {}), "mix": mixdir, "host_dirs": host_dirs or {},
                                  "docs": docs, "status": "ready", "decisions": {}, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.save(); return self.d["slices"][name]

    def active(self):
        a = [n for n, s in self.d["slices"].items() if s["status"] == "active"]
        return a[0] if a else None

    def activate_next(self):
        r = [n for n, s in self.d["slices"].items() if s["status"] == "ready"]
        if not r: return None
        cur = self.active()
        if cur: self.d["slices"][cur]["status"] = "retired"; self.d["slices"][cur]["retired"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        n = sorted(r, key=lambda n: self.d["slices"][n]["built"])[0]
        self.d["slices"][n]["status"] = "active"; self.d["slices"][n]["activated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save(); return n

    def decided(self, name, size, ndocs):
        """one decision on slice `name` for `size`, judged on its first ndocs documents (all, unless stopped early)"""
        s = self.d["slices"][name]; s["decisions"][size] = s["decisions"].get(size, 0) + 1
        for d in s["docs"][:ndocs]: self.d["doc_decisions"][d["id"]] = self.d["doc_decisions"].get(d["id"], 0) + 1
        self.save()

    def due(self, name, rotate_after):
        s = self.d["slices"][name]
        return max(s["decisions"].values(), default=0) >= rotate_after

    def parts_of(self, name):
        return [d["part"] for d in self.d["slices"][name]["docs"]]


def main(argv):
    if argv[0] == "goldmix":
        out, seed = argv[1], int(argv[2]); parts, select = [], {}
        for a in argv[3:]:                                     # NAME=DIR or NAME=DIR@even / @odd / @i,j,k
            n, g = a.split("=", 1)
            if "@" in g:
                g, w = g.split("@", 1); nd = len(json.load(open(f"{g}/kld.json"))["docs"])
                select[n] = list(range(0, nd, 2)) if w == "even" else list(range(1, nd, 2)) if w == "odd" else [int(x) for x in w.split(",")]
            parts.append((n, g))
        man = goldmix(out, parts, seed, select); print(f"{out}: {len(man['docs'])} docs, {man['rows']} rows, parts {[p for p, _ in parts]}")
    elif argv[0] == "add":
        R = Registry(argv[1]); parts = dict(a.split("=", 1) for a in argv[4:]) or None
        s = R.add(argv[2], argv[3], parts); print(f"{argv[2]}: {len(s['docs'])} docs registered")
    elif argv[0] == "show":
        R = Registry(argv[1])
        for n, s in R.d["slices"].items():
            print(f"{n:24s} {s['status']:8s} docs {len(s['docs'])} (x {sum(d['part'] == 'x' for d in s['docs'])}, c {sum(d['part'] == 'c' for d in s['docs'])})"
                  f" decisions {s['decisions']}")
        dd = R.d["doc_decisions"]
        if dd: print(f"doc decisions: {len(dd)} docs, max {max(dd.values())}, calib docs {sum(1 for k in dd if k.startswith('c:'))}")
    elif argv[0] == "draw":
        R = Registry(argv[1]); led, pools, n, seed = argv[2], argv[3], int(argv[4]), int(argv[5]); mode = argv[6] if len(argv) > 6 else "merged"
        pool = json.load(open(pools))["pool"]; mine = {f"golden-{k}-c" for k in R.d["slices"]}
        if mode == "merged":
            pick, ncand, use = draw_merged(R, led, pool, n, seed)
            print(" ".join(map(str, pick)))
            print(f"draw: {ncand} fit + search docs available, {len(pick)} drawn (least-used first); uses on them: "
                  f"min {min(use[i] for i in pick) if pick else 0}, max {max(use[i] for i in pick) if pick else 0}", file=sys.stderr)
            return 0
        used = set()
        for l in open(led):
            r = json.loads(l)
            if r["doc"] == "all" or r["use"] in mine: continue
            if mode in ("no27b", "headx") and r["use"].startswith(("g08-", "g2b-")): continue
            if mode == "headx" and r["use"] in ("w119-headx-search", "w94-headx-search"): continue
            used.add(int(r["doc"]))
        busy = {int(d["id"][2:]) for s_ in R.d["slices"].values() if s_["status"] in ("ready", "active") for d in s_["docs"] if d["id"].startswith("c:")}
        cand = [i for i, p in enumerate(pool) if p == "search" and i not in used and i not in busy]
        rng = random.Random(seed); dd = R.d["doc_decisions"]
        cand.sort(key=lambda i: (dd.get(f"c:{i}", 0), rng.random()))
        pick = sorted(cand[:n])
        print(" ".join(map(str, pick)))
        print(f"draw: {len(cand)} docs in the budget (mode {mode}), {len(pick)} drawn; decisions on them so far: "
              f"max {max([dd.get(f'c:{i}', 0) for i in pick] or [0])}", file=sys.stderr)
    elif argv[0] == "poolcheck":
        pool = json.load(open(argv[1]))["pool"]; bad = 0
        for g in argv[2:]:
            m = json.load(open(g if g.endswith(".json") else f"{g}/kld.json")); c = {}
            for d in m["docs"]:
                if d.get("part", "c") != "c" or d.get("orig") is None or str(d.get("source", "calib")) not in ("calib", ""): continue
                p = pool[int(d["orig"])]; c[p] = c.get(p, 0) + 1
            out = {p: k for p, k in c.items() if p not in MERGED}; bad += bool(out)
            print(f"{g}: calib docs by pool {c}" + (f"  OUTSIDE fit + search: {out}" if out else "  ok"))
        return 1 if bad else 0
    elif argv[0] == "ledger":
        R = Registry(argv[1]); s = R.d["slices"][argv[2]]; os.makedirs(argv[3], exist_ok=True)
        c = s["parts"].get("c")
        shutil.copy(f"{c}/kld.json", f"{argv[3]}/golden-{argv[2]}-c-kld.json"); print(f"{argv[3]}/golden-{argv[2]}-c-kld.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
