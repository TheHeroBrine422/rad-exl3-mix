# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""driver.py CONF [--status] [--steps N] -- the V3 search driver (w126, #166; TODO "Search driver", "Continuous early
stop", "Final picks", "Low sizes in parallel", "Heads before w124's un-permute"). Decides nothing on its own: the first
real run needs the user's go (CONF "go": true; without it the driver refuses to submit anything but CONF "dry" evals).

Loop (one pass a few seconds; never waits on one eval):
  1 ingest     new lines of the eval queue's results (rxrun.multi ROOT/results.jsonl + docs/<tag>-<slice>.json; a
               stopped run carries "stopped" and per-document sums for the scored documents only)
  2 decide     a candidate vs its size's CURRENT incumbent on the same slice, paired per document (seqrule.Paired):
               stopped -> reject; whole slice -> accept iff pooled log r < 0 and every part < 0; an accept makes it the
               incumbent. A result on a slice that is no longer active only feeds the fit ("stale"). Audit runs (a
               random audit_frac of the candidates are scored without early stop) are decided by the rule applied after
               the fact -- the same decision as a stopped run -- and log the early-stop bias (prefix vs whole-slice ratio)
  3 fit rows   every result is a row (seqrule labels: "S|full", or "S|pre<k>" with k cut down to a 5% grid), plus
               the generating incumbent's KLD on the same documents as the label's anchor; SE = paired delta-method SE
  4 refit      every refit_every results (and when the first rows exist): fit.py on all rows + CONF prior_obs, run
               through CONF fit_cmd (colo) as a background process; the driver keeps generating from the last model
  5 rotate     when a size has decided rotate_after candidates on the active slice: the registry's next ready slice
               becomes active and every incumbent is scored on it first (whole slice, no early stop)
  6 generate   while fewer than queue_depth evals are outstanding: the size with the fewest decisions in flight +
               done, a move kind by share (moves.kind_shares: V2's kind rewards), moves.generate inside the size's byte
               window [target - size_tol, target]; appended to the queue with the early-stop block (rule + the
               incumbent's per-document sums on the slice, in the golden's order)
  7 report     ROOT/status.txt, ROOT/frontier.json (per release band [T - 0.5 gap below, T + 0.3 gap above]: every
               whole-slice measurement, KLD normalised by the model's slice offset, the non-dominated ones marked)
State: ROOT/state.jsonl (append-only: submit, result, decision, incumbent, rotate, fit), ROOT/assign/<tag>.json.
Restart = the same command (state replayed; outstanding evals stay in the queue)."""
import argparse, collections, json, math, os, random, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import menu as MN            # noqa: E402
import moves as MV           # noqa: E402
import seqrule as SR         # noqa: E402
import slices as SL          # noqa: E402

CST = -6 * 3600


def cst(t=None): return time.strftime("%m-%d %H:%M:%S", time.gmtime((t or time.time()) + CST)) + " CST"


def release_bands(sizes):
    """TODO 'Final picks': [T - 0.5 x gap below, T + 0.3 x gap above] (ends: the one gap they have)"""
    s = sorted(sizes); out = {}
    for i, T in enumerate(s):
        gb = T - s[i - 1] if i > 0 else (s[1] - T if len(s) > 1 else 0.5)
        ga = s[i + 1] - T if i + 1 < len(s) else gb
        out[T] = (round(T - 0.5 * gb, 6), round(T + 0.3 * ga, 6))
    return out


class Driver:
    def __init__(self, conf_path):
        self.cp = conf_path; self.c = json.load(open(conf_path)); c = self.c
        self.R = c["root"]; os.makedirs(f"{self.R}/assign", exist_ok=True); os.makedirs(f"{self.R}/fits", exist_ok=True)
        self.M = MN.Menu.load(c["menu"]); self.reg = SL.Registry(c["registry"])
        self.sizes = [float(s) for s in c["sizes"]]; self.tol = float(c.get("size_tol", 0.02))
        self.rule = c.get("rule", {"kind": "curve", "p": 2.0, "warmup": 0.05})
        self.rng = random.Random(c.get("seed", 0))
        # state
        self.sub = {}            # tag -> submit event
        self.res = {}            # tag -> {"slice", "docs": [B or None], "pos": [n or None], "stopped", ...}
        self.inc = {}            # size -> tag
        self.dec = collections.Counter(); self.kind_stats = collections.defaultdict(lambda: [0, 0])
        self.seen = set(); self.obs = {}; self.n_results = 0; self.fit_at = 0
        self.fit = None          # {"Mo", "x", "pred", "draws", "n_obs", "report"}
        self.fitproc = None; self.res_off = 0; self.bias = []; self.nogen = {}
        self.replay()

    # ------------------------------------------------------------------------------------------------ state
    def rec(self, e):
        e = dict(e, t=cst())
        with open(f"{self.R}/state.jsonl", "a") as f: f.write(json.dumps(e) + "\n")

    def log(self, msg):
        line = f"{cst()} {msg}"; print(line, flush=True)
        with open(f"{self.R}/driver.log", "a") as f: f.write(line + "\n")

    def replay(self):
        p = f"{self.R}/state.jsonl"
        if not os.path.exists(p): return
        for l in open(p):
            try: e = json.loads(l)
            except ValueError: continue
            ev = e["ev"]
            if ev == "submit": self.sub[e["tag"]] = e; self.seen.add(e["hash"])
            elif ev == "result": self.res[e["tag"]] = e["r"]; self.n_results += 1
            elif ev == "incumbent": self.inc[e["size"]] = e["tag"]
            elif ev == "decision":
                self.dec[e["size"]] += 1
                if e.get("kind"): self.kind_stats[e["kind"]][1] += 1; self.kind_stats[e["kind"]][0] += e["decision"] == "accept"
            elif ev == "results_offset": self.res_off = e["off"]
            elif ev == "fit_done": self.fit_at = e["n_results"]
            elif ev == "bias": self.bias.append(e)
        for t, r in self.res.items(): self.add_obs(t, r)
        self.log(f"replayed: {len(self.sub)} submitted, {len(self.res)} results, incumbents {len(self.inc)}")

    def assign(self, tag): return json.load(open(f"{self.R}/assign/{tag}.json"))

    # ------------------------------------------------------------------------------------------------ queue
    def slice_entry(self, name):
        s = self.reg.d["slices"][name]
        return {"name": name, "parts": {"m": s.get("alias", s["mix"])}}

    def submit(self, tag, a, size, kind, slice_name, early=True, ref_tag=None, extra=None):
        h = MV.ahash(a)
        json.dump(a, open(f"{self.R}/assign/{tag}.json", "w"))
        line = {"tag": tag, "mix": self.M.eval_mix(a, self.c["base"]), "slices": [self.slice_entry(slice_name)]}
        if early and ref_tag:
            r = self.res[ref_tag]
            line["early_stop"] = {"rule": self.rule, "ref": [[b, n] for b, n in zip(r["docs"], r["pos"])],
                                  "parts": self.reg.parts_of(slice_name), "ref_tag": ref_tag}
        with open(self.c["evals"], "a") as f: f.write(json.dumps(line) + "\n")
        e = {"ev": "submit", "tag": tag, "size": size, "kind": kind, "slice": slice_name, "hash": h, "early": bool(early and ref_tag),
             "ref": ref_tag, "bytes": self.M.bytes(a), "bpw": self.M.bpw(a), **(extra or {})}
        self.rec(e); self.sub[tag] = e; self.seen.add(h)
        return tag

    def outstanding(self): return [t for t in self.sub if t not in self.res]

    def ingest(self):
        p = self.c["results"]
        if not os.path.exists(p): return 0
        n = 0
        with open(p) as f:
            f.seek(self.res_off)
            while True:
                l = f.readline()
                if not l or not l.endswith("\n"): break
                self.res_off = f.tell()
                try: r = json.loads(l)
                except ValueError: continue
                t = r.get("tag")
                if t not in self.sub or t in self.res: continue
                if r.get("fail"):
                    rr = {"fail": True, "err": r.get("err"), "slice": self.sub[t]["slice"]}
                else:
                    dp = f"{self.c['docs']}/{t}-{r['slice']}.json"
                    dd = json.load(open(dp))["m"] if os.path.exists(dp) else None
                    if dd is None: continue
                    rr = {"slice": r["slice"], "docs": [x[0] if x is not None else None for x in dd],
                          "pos": [x[1] if x is not None else None for x in dd], "stopped": r.get("stopped"), "bytes": r.get("bytes"),
                          "bpw": r.get("bpw"), "kld": r.get("kld")}
                self.res[t] = rr; self.n_results += 1; n += 1
                self.rec({"ev": "result", "tag": t, "r": rr})
                self.on_result(t, rr)
        self.rec({"ev": "results_offset", "off": self.res_off})
        return n

    # ------------------------------------------------------------------------------------------------ decisions
    def paired(self, ref, cand, parts, k=None, rule=None):
        P = SR.Paired(len(parts), rule or {"kind": "full"})
        for i, (a, b, p) in enumerate(zip(ref["docs"], cand["docs"], parts)):
            if k is not None and i >= k: break
            if a is None or b is None: break
            P.add(a, b, p)
            if rule is not None and P.check(): break
        return P

    def on_result(self, t, r):
        s = self.sub[t]; size = s["size"]
        self.add_obs(t, r)
        if r.get("fail"): self.log(f"{t}: FAILED ({r.get('err')})"); return
        if s["kind"] == "incumbent":
            self.log(f"{t}: incumbent of {size} scored on {r['slice']}: KLD {r['kld']:.5f} at {r['bpw']:.3f} bpw"); return
        act = self.reg.active()
        if r["slice"] != act:
            self.rec({"ev": "decision", "tag": t, "size": size, "decision": "stale", "kind": None}); return
        cur = self.inc.get(size); parts = self.reg.parts_of(r["slice"])
        if cur is None or cur not in self.res or self.res[cur].get("slice") != r["slice"]:
            self.rec({"ev": "decision", "tag": t, "size": size, "decision": "stale", "kind": None}); return
        ref = self.res[cur]
        if r.get("stopped"):
            P = self.paired(self.res[s["ref"]] if s.get("ref") in self.res else ref, r, parts)
            d = "reject"; ndocs = P.k
        else:
            if s.get("audit") and s.get("ref") in self.res:                       # the rule after the fact
                Pe = self.paired(self.res[s["ref"]], r, parts, rule=self.rule)
                Pf = self.paired(self.res[s["ref"]], r, parts)
                if Pe.stopped:
                    self.rec({"ev": "bias", "tag": t, "k": Pe.k, "d_prefix": Pe.d(), "d_full": Pf.d(), "se_full": Pf.se()})
                    self.bias.append({"d_prefix": Pe.d(), "d_full": Pf.d()})
                    self.decide(t, size, "reject", Pe, r["slice"], Pe.k); return
            P = self.paired(ref, r, parts); d = P.final(); ndocs = P.k
        self.decide(t, size, d, P, r["slice"], ndocs)

    def decide(self, t, size, d, P, sl, ndocs):
        s = self.sub[t]
        self.reg.decided(sl, str(size), ndocs); self.dec[size] += 1
        self.kind_stats[s["kind"]][1] += 1; self.kind_stats[s["kind"]][0] += d == "accept"
        self.rec({"ev": "decision", "tag": t, "size": size, "decision": d, "kind": s["kind"], "k": P.k, "z": P.z(), "logr": P.d(),
                  "se": P.se(), "parts": P.summary()["parts"]})
        if d == "accept":
            old = self.inc.get(size); self.inc[size] = t
            self.rec({"ev": "incumbent", "size": size, "tag": t, "prev": old})
            self.log(f"ACCEPT {t} ({s['kind']}) for {size}: log r {P.d():+.4f} (z {P.z():+.2f}) vs {old}; {s['bpw']:.4f} bpw")
        else:
            self.log(f"reject {t} ({s['kind']}) for {size}: log r {P.d():+.4f} (z {P.z():+.2f}) after {P.k} docs" + (" [stopped]" if self.res[t].get("stopped") else ""))

    # ------------------------------------------------------------------------------------------------ fit rows
    def add_obs(self, t, r):
        if r.get("fail") or not r.get("docs"): return
        s = self.sub.get(t)
        if s is None: return
        a = self.assign(t); docs, pos = r["docs"], r["pos"]
        N = len(docs); k = sum(1 for x in docs if x is not None)
        if k == 0: return
        g = max(1, math.ceil(N / 20))
        full = k == N
        kq = N if full else (k // g) * g
        if kq < 3: return
        label = f"{r['slice']}|" + ("full" if full else f"pre{kq}")
        kl = sum(docs[:kq]) / max(sum(pos[:kq]), 1)
        ref = s.get("ref") if s.get("ref") in self.res else None
        se = 0.0
        if ref and ref != t:
            P = self.paired(self.res[ref], r, self.reg.parts_of(r["slice"]) if r["slice"] in self.reg.d["slices"] else ["m"] * N, k=kq)
            se = P.se() if P.se() != SR.INF else 0.5
            rr = self.res[ref]
            if all(x is not None for x in rr["docs"][:kq]):
                self.obs[(ref, label)] = {"tag": f"{ref}@{label}", "assign": self.assign(ref), "label": label,
                                          "kld": sum(rr["docs"][:kq]) / max(sum(rr["pos"][:kq]), 1), "se": 0.0}
        self.obs[(t, label)] = {"tag": f"{t}@{label}", "assign": a, "label": label, "kld": kl, "se": se}

    def maybe_refit(self):
        if self.fitproc is not None:
            if self.fitproc[0].poll() is None: return
            proc, out, obs, n_at = self.fitproc; self.fitproc = None
            if proc.returncode != 0 or not os.path.exists(out):
                self.fit_fail_t = time.time(); self.log(f"fit FAILED rc {proc.returncode} (see {out}.log); next try in {self.c.get('fit_retry_s', 120)} s"); return
            self.load_fit(out, obs); self.fit_at = n_at
            self.rec({"ev": "fit_done", "n_results": n_at, "out": out, "report": self.fit["report"]})
            self.log(f"fit done on {len(obs)} rows: {self.fit['report']}")
            return
        due = (self.fit is None and len(self.obs) >= int(self.c.get("min_rows_fit", max(2, len(self.sizes))))) or \
              (self.n_results - self.fit_at >= int(self.c.get("refit_every", 20)))
        if not due or time.time() - getattr(self, "fit_fail_t", 0) < float(self.c.get("fit_retry_s", 120)): return
        obs = list(self.obs.values()) + self.prior_obs()
        n = len([f for f in os.listdir(f"{self.R}/fits") if f.endswith(".in.json")])
        inp, out = f"{self.R}/fits/fit{n:04d}.in.json", f"{self.R}/fits/fit{n:04d}.out.json"
        json.dump({"menu": self.c["menu_remote"] if self.c.get("menu_remote") else self.c["menu"], "obs": obs,
                   "opts": self.c.get("fit_opts", {}), "ndraw": int(self.c.get("ndraw", 16)), "seed": n}, open(inp, "w"))
        cmd = [x.replace("{in}", inp).replace("{out}", out) for x in self.c.get("fit_cmd") or [sys.executable, "-I", f"{HERE}/fit.py", "{in}", "{out}"]]
        self.fitproc = (subprocess.Popen(cmd, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), out, obs, self.n_results)
        self.log(f"fit {n:04d} started on {len(obs)} rows ({self.n_results} results)")

    def prior_obs(self):
        p = self.c.get("prior_obs")
        return [json.loads(l) for l in open(p)] if p and os.path.exists(p) else []

    def load_fit(self, out, obs):
        import fit as FT
        d = json.load(open(out))
        Mo = FT.build(self.M, obs, self.c.get("fit_opts", {}))
        import numpy as np
        x = np.array(d["x"]); draws = [MV.Pred(Mo, np.array(v)) for v in d.get("draws", [])]
        self.fit = {"Mo": Mo, "x": x, "pred": MV.Pred(Mo, x), "draws": draws, "report": d["report"], "model": d["model"]}

    # ------------------------------------------------------------------------------------------------ generation
    def window(self, size):
        """bytes [target - size_tol, target]; the floor clamped to what the menu can reach (a size above the menu's top)"""
        hi = size * self.M.P / 8; lo = (size - self.tol) * self.M.P / 8
        if not hasattr(self, "_maxb"): self._maxb = self.M.fixed + sum(t["opts"][-1]["bytes"] for t in self.M.T)
        if self._maxb < lo: lo = self._maxb - self.tol * self.M.P / 8
        return lo, hi

    def init_assign(self, size):
        ini = (self.c.get("init") or {}).get(str(size)) or (self.c.get("init") or {}).get("default", "nearest")
        if isinstance(ini, dict): return ini
        if ini.endswith(".json"): return json.load(open(ini))
        lv = self.M.levels()
        h = self.c.get("init_head"); e = self.c.get("init_emb")
        under = [n for n in lv if self.M.bpw(self.M.uniform(n, h, e)) <= size]
        return self.M.uniform(under[-1] if under else lv[0], h, e)

    def ensure_incumbents(self, act):
        """every size has an incumbent scored (whole slice) on the active slice"""
        out = set(self.outstanding())
        for size in self.sizes:
            t = self.inc.get(size)
            if t is None:
                a = self.init_assign(size); tag = self.newtag(size, "init")
                self.submit(tag, a, size, "incumbent", act, early=False)
                self.inc[size] = tag; self.rec({"ev": "incumbent", "size": size, "tag": tag, "prev": None})
                continue
            if t in self.res and self.res[t].get("slice") == act: continue
            if any(self.sub[o]["kind"] == "incumbent" and self.sub[o]["size"] == size and self.sub[o]["slice"] == act for o in out): continue
            if t in out: continue
            tag = self.newtag(size, "inc")
            self.submit(tag, self.assign(t), size, "incumbent", act, early=False, extra={"of": t})
            self.inc[size] = tag; self.rec({"ev": "incumbent", "size": size, "tag": tag, "prev": t})

    def newtag(self, size, kind):
        n = len(self.sub)
        return f"{self.c.get('tag_prefix', 'v3s')}-{size:g}-{kind}-{n:05d}"

    def generate(self, act):
        if self.fit is None: return 0
        out = self.outstanding(); depth = int(self.c.get("queue_depth", 6)); made = 0
        inflight = collections.Counter(self.sub[t]["size"] for t in out)
        shares = MV.kind_shares({k: tuple(v) for k, v in self.kind_stats.items()}, self.c.get("kind_share0", MV.SHARE0))
        kcount = collections.Counter(self.sub[t]["kind"] for t in self.sub if self.sub[t]["kind"] in shares)
        tries = 0; failed = set()
        while len(out) + made < depth and tries < 4 * len(self.sizes):
            tries += 1
            ready = [s for s in self.sizes if self.inc.get(s) in self.res and self.res[self.inc[s]].get("slice") == act and s not in failed]
            if not ready: break
            w = self.c.get("size_weights", {})
            size = min(ready, key=lambda s: ((self.dec[s] + inflight[s]) / float(w.get(str(s), 1.0)), self.rng.random()))
            a = self.assign(self.inc[size]); lo, hi = self.window(size)
            kind = MV.pick_kind(shares, kcount, self.rng)
            c = MV.generate(kind, self.fit["pred"], self.fit["draws"], a, lo, hi, self.seen, self.rng)
            if c is None:
                for k2 in sorted(shares, key=lambda k: -shares[k]):
                    if k2 == kind: continue
                    c = MV.generate(k2, self.fit["pred"], self.fit["draws"], a, lo, hi, self.seen, self.rng)
                    if c is not None: kind = k2; break
            if c is None:
                failed.add(size)
                if not self.nogen.get(size): self.log(f"no new candidate for {size} (window {lo * 8 / self.M.P:.4f}-{hi * 8 / self.M.P:.4f} bpw)")
                self.nogen[size] = True; continue
            self.nogen[size] = False
            pr = self.fit["pred"]; dpred = math.log(pr.kl(c) / pr.kl(a))
            audit = self.rng.random() < float(self.c.get("audit_frac", 0.1))
            tag = self.newtag(size, kind)
            self.submit(tag, c, size, kind, act, early=not audit, ref_tag=self.inc[size], extra={"pred_logr": dpred, "audit": audit})
            kcount[kind] += 1; inflight[size] += 1; made += 1
        return made

    # ------------------------------------------------------------------------------------------------ rotation
    def rotate(self):
        act = self.reg.active()
        if act is None:
            act = self.reg.activate_next()
            if act: self.rec({"ev": "rotate", "to": act, "from": None}); self.log(f"slice {act} active")
            return act
        if self.reg.due(act, int(self.c.get("rotate_after", 25))):
            nxt = [n for n, s in self.reg.d["slices"].items() if s["status"] == "ready"]
            if nxt:
                new = self.reg.activate_next(); self.rec({"ev": "rotate", "to": new, "from": act})
                self.log(f"ROTATE {act} -> {new}"); return new
            if not getattr(self, "_warned", False): self.log(f"rotation due on {act} but no ready slice: build one (slices.py)"); self._warned = True
        return act

    # ------------------------------------------------------------------------------------------------ report
    def report(self):
        bands = release_bands(self.sizes); soff = (self.fit or {}).get("model", {}).get("slices", {})
        pts = []
        for t, r in self.res.items():
            if r.get("fail") or not r.get("docs") or any(x is None for x in r["docs"]): continue
            off = soff.get(f"{r['slice']}|full", 0.0)
            pts.append({"tag": t, "bpw": r["bpw"], "kld": r["kld"], "slice": r["slice"], "kld_norm": r["kld"] / math.exp(off), "size": self.sub[t]["size"]})
        fr = {}
        for T, (lo, hi) in bands.items():
            p = sorted([x for x in pts if lo <= (x["bpw"] or 0) <= hi], key=lambda x: (x["bpw"], x["kld_norm"]))
            best = math.inf
            for x in p:
                x["nondominated"] = x["kld_norm"] < best; best = min(best, x["kld_norm"])
            fr[str(T)] = {"band": [lo, hi], "incumbent": self.inc.get(T), "points": p}
        json.dump(fr, open(f"{self.R}/frontier.json.part", "w"), indent=0); os.replace(f"{self.R}/frontier.json.part", f"{self.R}/frontier.json")
        nd = sum(self.dec.values()); acc = sum(v[0] for v in self.kind_stats.values())
        stopped = sum(1 for r in self.res.values() if r.get("stopped"))
        line = (f"{cst()} | slice {self.reg.active()} | submitted {len(self.sub)}, results {len(self.res)} ({stopped} stopped early), "
                f"outstanding {len(self.outstanding())} | decisions {nd}, accepts {acc} | kinds " +
                " ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(self.kind_stats.items())) +
                f" | fit {'none' if not self.fit else self.fit['report'].get('rows')} rows" + (" (running)" if self.fitproc else ""))
        open(f"{self.R}/status.txt", "w").write(line + "\n")
        return line

    # ------------------------------------------------------------------------------------------------ main
    def step(self):
        self.ingest()
        self.maybe_refit()
        act = self.rotate()
        if act is None: return "no slice"
        go = self.c.get("go") or self.c.get("dry")
        if not go: return "no go (CONF go / dry)"
        self.ensure_incumbents(act)
        self.generate(act)
        return self.report()

    def run(self, steps=None, sleep=None):
        sleep = float(self.c.get("poll_s", 5)) if sleep is None else sleep
        i = 0
        while steps is None or i < steps:
            if os.path.exists(f"{self.R}/STOP"): self.log("STOP file: exit"); break
            st = self.step(); i += 1
            if sleep: time.sleep(sleep)
        if self.fitproc and self.fitproc[0].poll() is None and steps is not None: pass
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="driver.py")
    ap.add_argument("conf"); ap.add_argument("--status", action="store_true"); ap.add_argument("--steps", type=int)
    a = ap.parse_args(argv)
    D = Driver(a.conf)
    if a.status: print(D.report()); return 0
    return D.run(a.steps)


if __name__ == "__main__":
    sys.exit(main())
