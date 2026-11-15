# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""replay.py -- time-ordered replay of the #183 search's move predictions (RULES-187 section 2).

Reads a frozen snapshot of the live driver's run (never writes to it): driver.json, registry.json, menu.json,
prior-obs-w1a8.jsonl, drv/state.jsonl, drv/assign/<tag>.json. Rows are built by the driver's own add_obs (the same
code path, run on a bare Driver object), so a fit at refit point N sees exactly the rows the live driver had after
its N-th result.

  replay.py cands SNAP OUT.jsonl          candidate table (one line per candidate with a decision)
  replay.py jobs SNAP                     the job list "N ARM GROUP" (one per line), for xargs -P
  replay.py job SNAP OUTDIR N ARM GROUP   one refit point: fit ARM on rows of results <= N, predict the candidates
                                          whose results are in (N, N+20]; writes OUTDIR/ARM/N-GROUP.json
                                          ARM: a (per-band hard fit, GROUP b1..b4), b (joint), c<sd> (pool187, sd in 0.1 0.3 1.0)
Candidate = submit with a move kind and a decision with a logr (stale decisions have none). Whole-slice = result
without "stopped" (RULES-187 section 1)."""
import json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..")
sys.path.insert(0, V3); sys.path.insert(0, os.path.join(V3, "..", "refit")); sys.path.insert(0, HERE)
import menu as MN            # noqa: E402
import slices as SL          # noqa: E402
import driver as DR          # noqa: E402  (the driver's own add_obs; main() is not run)
import fit as FT             # noqa: E402
import pool187 as PL         # noqa: E402

KINDS = {"solve", "partial", "pair", "block", "thompson", "explore"}
CHUNK = 20
ARMS_AB = ["a", "b"]
POOL_SD = {"c0.1": 0.1, "c0.3": 0.3, "c1.0": 1.0}


class Snap:
    def __init__(self, snap):
        self.snap = snap
        self.conf = json.load(open(f"{snap}/driver.json"))
        self.M = MN.Menu.load(f"{snap}/menu.json")
        self.reg = SL.Registry(f"{snap}/registry.json")
        self.prior = [json.loads(l) for l in open(f"{snap}/prior-obs-w1a8.jsonl") if l.strip()]
        self.groups = self.conf["fit_groups"]
        self.sub, self.res, self.dec, self.order = {}, {}, {}, []
        for l in open(f"{snap}/drv/state.jsonl"):
            e = json.loads(l); ev = e["ev"]
            if ev == "submit": self.sub[e["tag"]] = e
            elif ev == "result":
                if e["tag"] in self.res: raise ValueError(f"duplicate result {e['tag']}")
                self.res[e["tag"]] = e["r"]; self.order.append(e["tag"])
            elif ev == "decision":
                if e["tag"] in self.dec: raise ValueError(f"duplicate decision {e['tag']}")
                self.dec[e["tag"]] = e
        self.idx = {t: i + 1 for i, t in enumerate(self.order)}          # result index N (1-based, every kind)
        self.cands = self._cands()

    def _cands(self):
        out = []
        for t, s in self.sub.items():
            d = self.dec.get(t)
            if s["kind"] not in KINDS or "pred_logr" not in s or d is None or "logr" not in d or t not in self.idx: continue
            r = self.res[t]; stopped = r.get("stopped") is not None
            out.append({"tag": t, "size": s["size"], "band": self.band(s["size"]), "kind": s["kind"],
                        "slice": s["slice"], "ref": s["ref"], "result_idx": self.idx[t], "stopped": stopped,
                        "audit": bool(s.get("audit")), "early": bool(s.get("early")), "decision": d["decision"],
                        "logr": d["logr"], "logr_pooled": d.get("logr_pooled"), "off": d.get("off"),
                        "pred_live": s["pred_logr"], "bpw": s.get("bpw"), "bytes": s.get("bytes"), "pred_b": s.get("b")})
        return sorted(out, key=lambda c: c["result_idx"])

    def band(self, size):
        for g in self.groups:
            if float(size) in {float(z) for z in g["sizes"]}: return g["name"]
        return ["b1", "b2", "b3", "b4"][PL.band_of_size(size)]

    def assign(self, t): return json.load(open(f"{self.snap}/drv/assign/{t}.json"))

    def obs_at(self, N):
        """the driver's observation rows after its N-th result (add_obs on a bare Driver: no state writes)"""
        d = object.__new__(DR.Driver)
        d.R = f"{self.snap}/drv"; d.reg = self.reg; d.sub = self.sub; d.res = {}; d.obs = {}
        for t in self.order[:N]:
            d.res[t] = self.res[t]; d.add_obs(t, self.res[t])
        return list(d.obs.values())


def chunk_of(idx): return CHUNK * ((idx - 1) // CHUNK)          # the refit point whose predictions cover this result


def jobs(S):
    maxN = max(c["result_idx"] for c in S.cands) if S.cands else 0
    gl = [g["name"] for g in S.groups]
    for N in range(CHUNK, maxN, CHUNK):
        chunk = [c for c in S.cands if N < c["result_idx"] <= N + CHUNK]
        if not chunk: continue
        for g in gl:
            if any(c["band"] == g for c in chunk): yield (N, "a", g)
        yield (N, "b", "all")
        for a in POOL_SD: yield (N, a, "all")


def run_job(S, outdir, N, arm, group):
    t0 = time.time()
    cand = [c for c in S.cands if N < c["result_idx"] <= N + CHUNK]
    if arm == "a":
        g = next(x for x in S.groups if x["name"] == group); sz = {float(z) for z in g["sizes"]}
        cand = [c for c in cand if c["band"] == group]
        obs = [o for o in S.obs_at(N) if o.get("size") in sz] + S.prior              # the driver's group rows + prior_obs
    else:
        obs = S.obs_at(N) + S.prior
    rep = {"N": N, "arm": arm, "group": group, "n_obs": len(obs), "n_cand": len(cand)}
    preds = {}
    if cand:
        M, opts = S.M, S.conf.get("fit_opts", {})
        if arm in ARMS_AB:
            Mo = FT.build(M, obs, opts); x = Mo.fit(); tb = Mo.tables(x)
            for c in cand:
                cc, aa = S.assign(c["tag"]), S.assign(c["ref"])
                preds[c["tag"]] = math.log(Mo.predict(x, cc, tb) / Mo.predict(x, aa, tb))
            pr = Mo.predict_rows(x, list(range(len(Mo.D.rows)))); km = np.array([r["kld"] for r in Mo.D.rows])
        else:
            Mo = PL.build_pool(M, obs, opts, POOL_SD[arm]); x = Mo.fit()
            for c in cand:
                k = PL.band_of_size(c["size"])
                preds[c["tag"]] = Mo.predict_pair(x, S.assign(c["tag"]), S.assign(c["ref"]), k)
            pr = Mo.predict_rows(x, list(range(len(Mo.D.rows)))); km = np.array([r["kld"] for r in Mo.D.rows])
        lr = np.log(pr / km)
        rep.update({"rows": len(Mo.D.rows), "labels": Mo.NS, "params": int(Mo.N), "rmse_log": float(np.sqrt(np.mean(lr ** 2))),
                    "cost": float(Mo.res.cost), "nfev": int(Mo.res.nfev)})
    rep["n_nonfinite"] = sum(1 for v in preds.values() if not math.isfinite(v))
    rep["secs"] = round(time.time() - t0, 2); rep["preds"] = preds
    od = os.path.join(outdir, arm); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"{N:04d}-{group}.json")
    json.dump(rep, open(p + ".part", "w")); os.replace(p + ".part", p)
    return rep


def write_cands(S, out):
    with open(out + ".part", "w") as f:
        for c in S.cands: f.write(json.dumps(c) + "\n")
    os.replace(out + ".part", out)


def main(argv):
    cmd = argv[0]
    if cmd == "cands":
        S = Snap(argv[1]); write_cands(S, argv[2]); print(f"{len(S.cands)} candidates, {len(S.order)} results, {len(S.dec)} decisions")
    elif cmd == "jobs":
        S = Snap(argv[1])
        for N, arm, g in jobs(S): print(N, arm, g)
    elif cmd == "job":
        S = Snap(argv[1]); rep = run_job(S, argv[2], int(argv[3]), argv[4], argv[5])
        print(json.dumps({k: v for k, v in rep.items() if k != "preds"}))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
