# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""#181 training data: stored bf16 goldens of any top-K (top-512 calib / gold3, top-256 gold4), mixed corpora, and the
stream plan (which segment of which document every virtual stream trains on at every step).

Golden rows: (doc i, position j) -> row row0 + j - score_from for score_from <= j <= tokens - 2 (gen rows only when the
golden has gen.u8); a row is the bf16 model's distribution of token j + 1 after the WHOLE prefix 0..j of the document,
so a student window must start at the document's start or carry the state of everything before it.

Window policy (RULES181): every document is used from its start up to CAP tokens (8,192: every xcorp piece whole, the
32k trace chunks' first 8k), cut into segments that carry the student's state (KV / GDN / conv) from the segment
before, detached (truncated backprop). V virtual streams each train on Q positions a step: stream s takes documents
s, s + V, s + 2V, ... of the seeded global order and continues its current document across steps; a step's quota
may hold the end of one document and the start of the next (separate segments). A segment never leaves fewer than
MIN_TAIL tokens of its document for later (it takes them now). The plan is a pure function of (order, V, Q, CAP), so
it does not depend on how many cards (ranks) run the streams."""
import json, os

import numpy as np

REC = {}


def rec(K):
    if K not in REC: REC[K] = np.dtype([("id", "<i4"), ("lp", "<f4")])
    return REC[K]


class Gold:
    def __init__(self, d, name=None, genfile=None):
        """genfile: a gen mask (one byte a row) to use instead of the golden's own gen.u8 (genmask181.py)"""
        self.d, self.name = d, name or os.path.basename(d.rstrip("/"))
        self.m = json.load(open(f"{d}/kld.json")); self.docs = self.m["docs"]
        self.K = int(self.m.get("top", 512)); R = self.m["rows"]
        self.tok = np.memmap(f"{d}/tokens.i32", np.int32, "r")
        gf = genfile or (f"{d}/gen.u8" if os.path.exists(f"{d}/gen.u8") else None)
        self.gen = np.memmap(gf, np.uint8, "r") if gf else None
        if self.gen is not None: assert len(self.gen) == R, (gf, len(self.gen), R)
        self.top = np.memmap(f"{d}/top.bin", rec(self.K), "r", shape=(R, self.K))
        self.rows = np.memmap(f"{d}/rows.f32", np.float32, "r", shape=(R, 3))
        self.off = np.concatenate([[0], np.cumsum([x["tokens"] for x in self.docs])])

    def ntok(self, i): return self.docs[i]["tokens"]

    def ids(self, i, a, b):
        return np.asarray(self.tok[self.off[i] + a:self.off[i] + b], np.int64)

    def scored(self, i, a, b):
        """(positions j in [a, b) with a golden row, their rows)"""
        d = self.docs[i]
        j = np.arange(max(a, d["score_from"]), min(b, d["tokens"] - 1))
        r = d["row0"] + j - d["score_from"]
        if self.gen is not None and len(r):
            g = np.asarray(self.gen[r[0]:r[-1] + 1]).astype(bool)[r - r[0]]; j, r = j[g], r[g]
        return j, r

    def nrows(self, i, cap):
        return len(self.scored(i, 0, min(cap, self.ntok(i)))[0])

    def targets(self, r, dev):
        """(top ids [m, K] long, top log-probs [m, K] f32, tail mass [m] f32) on dev"""
        import torch
        t = np.asarray(self.top[r[0]:r[-1] + 1])[r - r[0]]
        tid = torch.from_numpy(np.ascontiguousarray(t["id"])).long().to(dev)
        tlp = torch.from_numpy(np.ascontiguousarray(t["lp"])).float().to(dev)
        ptail = torch.from_numpy(np.ascontiguousarray(self.rows[r[0]:r[-1] + 1, 2][r - r[0]])).float().to(dev)
        return tid, tlp, ptail


def corpus_tag(name):
    for t in ("xq", "xc8k", "tr32k"):
        if name.startswith(t): return t
    return name


class Corpus:
    """golden chunks + (optionally) the whole-calib golden filtered to one pool; items = (golden index, doc) with >= MIN
    rows in the first CAP tokens, cached in CACHE (lengths only)"""
    def __init__(self, chunks, cap, calib=None, pools=None, pool="qad", min_rows=16, cache=None, genmask=None):
        gm = lambda d: (os.path.join(genmask, os.path.basename(d.rstrip("/")) + ".u8") if genmask else None)
        self.g = [Gold(d, genfile=gm(d)) for d in chunks]
        if genmask: assert all(os.path.exists(gm(d)) for d in chunks), "a chunk without a gen mask"
        if calib:
            gc = Gold(calib, "calib"); self.g.append(gc)
        self.cap = cap
        key = json.dumps({"chunks": [g.d for g in self.g], "cap": cap, "pool": pool, "min": min_rows, "genmask": genmask,
                          "pools": os.path.abspath(pools) if pools else None})
        if cache and os.path.exists(cache) and json.load(open(cache))["key"] == key:
            z = json.load(open(cache)); self.items = [tuple(x) for x in z["items"]]
        else:
            pl = json.load(open(pools))["pool"] if calib else None
            self.items = []
            for gi, g in enumerate(self.g):
                for i in range(len(g.docs)):
                    if g.name == "calib" and pl[i] != pool: continue
                    n = min(cap, g.ntok(i)); nr = g.nrows(i, cap)
                    if nr >= min_rows: self.items.append((gi, i, n, nr))
            if cache:
                json.dump({"key": key, "items": self.items}, open(cache + ".part", "w")); os.replace(cache + ".part", cache)
        self.tags = [corpus_tag(g.name) for g in self.g]

    def summary(self):
        out = {}
        for gi, i, n, nr in self.items:
            t = self.tags[gi]; o = out.setdefault(t, [0, 0, 0]); o[0] += 1; o[1] += n; o[2] += nr
        return {t: {"docs": v[0], "tokens": v[1], "rows": v[2]} for t, v in sorted(out.items())}


def plan(items, order, V, Q, min_tail=128, max_steps=None, queues=None):
    """[(step) -> [(stream) -> [(item index, a, b), ...]]] and per step the total rows; item = (gi, doc, n, nrows).
    Stream s owns order[s::V] (or queues[s] when given: w226's per-stream heads); each step it trains Q positions (tokens) of its documents in sequence,
    a segment ends at the document's end or at the quota, never leaving < min_tail tokens of a document behind (takes them now)."""
    queues = [list(q) for q in queues] if queues is not None else [list(order[s::V]) for s in range(V)]
    pos = [0] * V; cur = [0] * V          # index into the queue, offset in the current document
    steps = []
    while any(pos[s] < len(queues[s]) for s in range(V)):
        st = []
        for s in range(V):
            segs, q = [], Q
            while q > 0 and pos[s] < len(queues[s]):
                it = queues[s][pos[s]]; n = items[it][2]; a = cur[s]
                b = min(n, a + q)
                if n - b < min_tail: b = n
                segs.append((it, a, b)); q -= b - a
                if b >= n: pos[s] += 1; cur[s] = 0
                else: cur[s] = b
                if q < min_tail: break
            st.append(segs)
        steps.append(st)
        if max_steps and len(steps) >= max_steps: break
    return steps
