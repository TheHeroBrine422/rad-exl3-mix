# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""driver.py CONF [--status] [--steps N] -- the V3 search driver (w126, #166; TODO "Search driver", "Continuous early
stop", "Final picks", "Low sizes in parallel", "Heads before w124's un-permute"). Decides nothing on its own: the first
real run needs the user's go (CONF "go": true; without it the driver refuses to submit anything but CONF "dry" evals).

Loop (one pass a few seconds; never waits on one eval):
  1 ingest     new lines of the eval queue's results (rxrun.multi ROOT/results.jsonl + docs/<tag>-<slice>.json; a
               stopped run carries "stopped" and per-document sums for the scored documents only)
  2 decide     a candidate vs its size's CURRENT incumbent on the same slice, paired per document (seqrule.Paired):
               stopped -> reject; whole slice -> accept iff the SIZE-ADJUSTED log r < 0 and every part's < 0, log r =
               the equal-weight mean of the parts' log ratios (w131, #171; seqrule "pmean", the hosts' early stop too);
               an accept makes it the incumbent. Size-adjusted (w127, #167; TODO "Size-adjusted score in the search"):
               log KLD + b (bpw - reference bpw), b = the size's current fit's local slope -d log KLD / d bpw (the
               model's knapsack optimum at size -/+ b_delta; CONF b_default before the first fit; updated each refit;
               recorded per candidate at submit, so the hosts' early stop and the decision use the same b). CONF
               "size_adjust": false = raw KLD (the A/B's control arm). RULES-picks (w145): every per-part sum is the CAPPED
               statistic (b) (a document <= CONF cap 0.2 of its part's whole-slice incumbent KL; weights in the hosts'
               early-stop rule too) and an accept also needs the per-document guard (c) (each part's size-adjusted mean
               log(B_d / A_d) has the sign of its (b); CONF guard "docmean"); cap / guard null = the old rule. A result on a slice that is no longer active only feeds the fit ("stale"). Audit runs (a
               random audit_frac of the candidates are scored without early stop) are decided by the rule applied after
               the fact -- the same decision as a stopped run -- and log the early-stop bias (prefix vs whole-slice ratio)
  3 fit rows   every result is a row per slice part (w131, #171: labels "S|x|full" / "S|c|pre<k>", k = documents scored
               cut down to a 5% grid; one-part slices "S|full" / "S|pre<k>"), plus the generating incumbent's KLD on the
               same documents as the label's anchor; SE = the part's paired delta-method SE
  4 refit      every refit_every results (and when the first rows exist): fit.py on all rows + CONF prior_obs, run
               through CONF fit_cmd (colo) as a background process; the driver keeps generating from the last model
  5 rotate     when the active slice has decided rotate_after (25) candidates, all sizes together (CONF rotate_unit
               "slice", w145 after w141's audit; "size" = the old rule: any one size reaching rotate_after): the
               registry's next ready slice becomes active and every incumbent is scored on it first (whole slice, no
               early stop)
  6 generate   while fewer than queue_depth evals are outstanding: the size with the fewest decisions in flight +
               done, a move kind by share (moves.kind_shares: V2's kind rewards), moves.generate inside the size's byte
               window [target - size_tol, target]; appended to the queue with the early-stop block (rule + the
               incumbent's per-document sums on the slice, in the golden's order)
  monitor      (CONF "monitor": {"name", "golden", "alert_min", "alert_frac"}): every accept (a band's new best) is also
               scored on the held-out monitor golden (ver3-8k: verify pool, whole, no early stop; never a fit row, never a
               decision). Each read vs the size's best so far: (b) capped and (c) per-document mean with paired
               bootstrap CIs (pickstat.py; the "monitor" event's "stats", pooled (a) beside, never alone); "best" =
               the lowest size-adjusted capped score (mon_score). ALERT ONLY (user 11-13: no pre-set N, no automatic
               stop): a size gets a soft alert when its size-adjusted capped best has not improved by more than the paired SE over its last alert_frac (0.2) of its
               decisions, once it has >= alert_min (50); the user / lead ends a size by hand (touch ROOT/STOP-<size>).
               ROOT/monitor.json = the curves + alert state; ROOT/events.jsonl = convplot3 --events lines (alerts)
  7 report     ROOT/status.txt, ROOT/frontier.json (per release band [T - 0.5 gap below, T + 0.3 gap above]: every
               whole-slice measurement, KLD normalised by the model's slice offset, its size-adjusted score
               log KLD_norm + b_T (bpw - T), the non-dominated ones marked, "top": the best CONF topn (5) assignments by
               score). driver.py CONF --freeze-b OUT writes each band's b for the final picks (finalpick.py; before ver4)
Activation mode (w132, #172): CONF "act" (f16 | a8t | a8g<N>, default f16) = the compute mode of every trunk weight in
the eval containers (hostd: radmix act lines; metadata only, the bytes are unchanged). The mode in force is a state
event ("act") and changes only AT A SLICE ROTATION (or the first slice's activation), so every incumbent is rescored on
the new slice in the new mode before any candidate is compared with it; results still in flight on the old slice are
stale (fit rows only). Monitor reads are compared within one mode; at a change every incumbent is rescored on the
monitor golden in the new mode (CONF "act_monitor_rescore", default true). Submits and monitor reads in a non-f16 mode
carry "act" (absent = f16; convplot4 splits the series on it and marks the "act" event).
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
        self.rule = dict(c.get("rule") or {"kind": "curve", "p": 2.0, "warmup": 0.05})     # the pre-registered pick (c17045c)
        if "warmup" in c: self.rule["warmup"] = float(c["warmup"])
        # RULES-picks (w145): decisions on the capped statistic (b) (a document <= CONF cap of its part's incumbent KL;
        # null = uncapped) with the per-document sign guard (c) (CONF guard "docmean"; null = off)
        self.cap = c.get("cap", 0.2); g = c.get("guard", "docmean")
        if g: self.rule["guard"] = g
        self.sizeadj = bool(c.get("size_adjust", True)); self.b = {}; self.alerts = {}; self.res_lines = 0
        self.rng = random.Random(c.get("seed", 0))
        # state
        self.sub = {}            # tag -> submit event
        self.res = {}            # tag -> {"slice", "docs": [B or None], "pos": [n or None], "stopped", ...}
        self.inc = {}            # size -> tag
        self.dec = collections.Counter(); self.kind_stats = collections.defaultdict(lambda: [0, 0])
        self.seen = set(); self.obs = {}; self.n_results = 0
        # fit groups (TODO "Low sizes in parallel": a joint fit for the low sizes vs main-only for 2-7 is the lead's call):
        # [{"name", "sizes", "rows": "all" | "own"}]; default one group over every size and row
        self.groups = c.get("fit_groups") or [{"name": "all", "sizes": self.sizes, "rows": "all"}]
        self.gname = {float(z): g["name"] for g in self.groups for z in g["sizes"]}
        self.fits = {g["name"]: None for g in self.groups}       # {"Mo", "x", "pred", "draws", "report", "model"}
        self.fitprocs = {g["name"]: None for g in self.groups}; self.fit_at = {g["name"]: 0 for g in self.groups}
        self.res_off = 0; self.bias = []; self.nogen = {}; self.fixed0 = self.M.fixed; self.bytes_off = None; self.bytes_src = None
        self.mon = collections.defaultdict(list); self.stopped_sizes = set()
        self.act = "f16"                           # the activation mode in force (w132): changes only at a rotation
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
            elif ev == "results_offset": self.res_off = e["off"]; self.res_lines = e.get("lines", self.res_lines)
            elif ev == "b": self.b[e["size"]] = e["b"]
            elif ev == "alert": self.alerts[e["size"]] = e["on"]
            elif ev == "bytes_offset": self.bytes_off = e["off"]; self.bytes_src = e.get("src", "measured"); self.M.fixed = self.fixed0 + e["off"]
            elif ev == "monitor": self.mon[e["size"]].append(e)
            elif ev == "act": self.act = e["mode"]
            elif ev == "size_stop": self.stopped_sizes.add(e["size"])
            elif ev == "fit_done":
                gn = e.get("group", self.groups[0]["name"]); self.fit_at[gn] = e["n_results"]; last_fit = getattr(self, "_lastfit", {}); last_fit[gn] = e["out"]; self._lastfit = last_fit
            elif ev == "bias": self.bias.append(e)
        for t, r in self.res.items(): self.add_obs(t, r)
        for gn, out in getattr(self, "_lastfit", {}).items():          # the last fit of each group, on the rows it was fitted on
            try:
                tags = json.load(open(out))["obs_tags"]; by = {o["tag"]: o for o in list(self.obs.values()) + self.prior_obs()}
                if gn in self.fits and all(t in by for t in tags): self.fits[gn] = self.load_fit(out, [by[t] for t in tags])
            except (OSError, ValueError, KeyError) as e: self.log(f"replay: fit {gn} not reloaded ({e!r})")
        if self.bytes_off is None and self.c.get("bytes_offset0") is not None:     # w126's measured container offset (8,255,936 B)
            self.bytes_off = int(self.c["bytes_offset0"]); self.bytes_src = "conf"; self.M.fixed = self.fixed0 + self.bytes_off
            self.rec({"ev": "bytes_offset", "off": self.bytes_off, "src": "conf"})
        self.log(f"replayed: {len(self.sub)} submitted, {len(self.res)} results, incumbents {len(self.inc)}, fits {[g for g, f in self.fits.items() if f]}")

    def assign(self, tag): return json.load(open(f"{self.R}/assign/{tag}.json"))

    # ------------------------------------------------------------------------------------------------ queue
    def slice_entry(self, name):
        s = self.reg.d["slices"][name]
        return {"name": name, "parts": {"m": s.get("alias", s["mix"])}}

    def submit(self, tag, a, size, kind, slice_name, early=True, ref_tag=None, extra=None, slice_entry=None):
        h = MV.ahash(a)
        json.dump(a, open(f"{self.R}/assign/{tag}.json", "w"))
        line = {"tag": tag, "mix": self.M.eval_mix(a, self.c["base"]), "slices": [slice_entry or self.slice_entry(slice_name)]}
        if self.act != "f16": line["mix"]["act"] = self.act
        bs = self.b_of(size) if kind not in ("monitor", "incumbent") else 0.0
        bpw = self.M.bpw(a)
        if early and ref_tag:
            r = self.res[ref_tag]; off = bs * (bpw - self.sub[ref_tag]["bpw"])
            line["early_stop"] = {"rule": self.prule(self.rule, off, r["docs"], self.reg.parts_of(slice_name)), "ref": [[b, n] for b, n in zip(r["docs"], r["pos"])],
                                  "parts": self.reg.parts_of(slice_name), "ref_tag": ref_tag}
        with open(self.c["evals"], "a") as f: f.write(json.dumps(line) + "\n")
        e = {"ev": "submit", "tag": tag, "size": size, "kind": kind, "slice": slice_name or (slice_entry or {}).get("name"), "hash": h, "early": bool(early and ref_tag),
             "ref": ref_tag, "bytes": self.M.bytes(a), "bpw": bpw, "b": bs, **({"act": self.act} if self.act != "f16" else {}), **(extra or {})}
        self.rec(e); self.sub[tag] = e
        if kind != "monitor": self.seen.add(h)
        return tag

    def prule(self, rule, off, ref_docs, parts):
        """a seqrule rule with the size offset and the capped statistic's weights (from the incumbent's whole slice)"""
        r = dict(rule, off=off)
        w = SR.cap_weights([x or 0.0 for x in ref_docs], parts, self.cap) if self.cap else None
        if w is not None: r["w"] = w
        return r

    def mscore(self, size, kld, bpw):
        """a band's size-adjusted score: log KLD + b (bpw - T) (raw log KLD when size_adjust is off)"""
        return math.log(kld) + self.b_of(size) * (bpw - size)

    def on_monitor(self, t, s, r):
        """the held-out curve of a size (alert-only: monitor_alert)"""
        size = s["size"]; bpw = self.sub[t]["bpw"]; mode = s.get("act", "f16")
        prev = [m for m in self.mon[size] if m.get("act", "f16") == mode and m["tag"] in self.res]     # one mode's reads only (w132)
        best = min(prev, key=lambda m: self.mon_score(size, m)) if prev else None
        se = imp = st = None
        if best is not None:
            P = self.mon_paired(size, best, t, bpw)
            se, imp = P.se(), -P.da()
            try:                                                            # RULES-picks: (b) and (c) with paired bootstrap CIs
                import pickstat as PS
                st = PS.compare(self.res[best["tag"]]["docs"], r["docs"], off=self.b_of(size) * (bpw - best["bpw"]), cap=self.cap or 1.0,
                                boot=int((self.c.get("monitor") or {}).get("boot", PS.BOOT)))
            except Exception as ex:                                         # noqa: BLE001 -- the monitor never stops the driver
                self.log(f"monitor {size}: bootstrap failed ({ex!r})")
        e = {"ev": "monitor", "size": size, "tag": t, "of": s.get("of"), "kld": r["kld"], "bpw": bpw, "dec_n": self.dec[size],
             "b": self.b_of(size), "vs": best["tag"] if best else None, "improve": imp, "se": se, "stats": st, **({"act": mode} if mode != "f16" else {})}
        self.rec(e); self.mon[size].append(e)
        self.log(f"monitor {size}: {r['kld']:.5f} at {bpw:.4f} bpw ({self.c['monitor']['name']})" +
                 (f", vs best {best['kld']:.5f} (size-adjusted): (b) capped {st['b']['d']:+.4f} [{st['b']['ci'][0]:+.4f}, {st['b']['ci'][1]:+.4f}],"
                  f" (c) per-doc {st['c']['d']:+.4f} [{st['c']['ci'][0]:+.4f}, {st['c']['ci'][1]:+.4f}]; pooled (a) {st['a']['d']:+.4f}" if st else
                  f", vs best {best['kld']:.5f}: size-adjusted (b) {-imp:+.4f} (se {se:.4f})" if best else ""))
        self.monitor_alert(size); self.dump_monitor()

    def mon_paired(self, size, ref, tag, bpw):
        """a monitor read vs an earlier one: the capped statistic (b) (weights from the earlier read), size-adjusted"""
        A = self.res[ref["tag"]]["docs"]; B = self.res[tag]["docs"]
        P = SR.Paired(len(A), self.prule({"kind": "full"}, self.b_of(size) * (bpw - ref["bpw"]), A, ["m"] * len(A)))
        for a, b in zip(A, B): P.add(a, b)
        return P

    def mon_score(self, size, m):
        """a monitor read's size-adjusted score on the capped statistic (RULES-picks: never the raw pooled (a) alone):
        log KLD of the size's first read in the mode (the anchor) + (b) of this read vs the anchor + b (bpw - T)"""
        reads = [x for x in self.mon.get(size, []) if x.get("act", "f16") == m.get("act", "f16") and x["tag"] in self.res]
        anc = reads[0] if reads else m
        if anc["tag"] == m["tag"] or not self.cap: return self.mscore(size, m["kld"], m["bpw"])
        return math.log(anc["kld"]) + self.mon_paired(size, anc, m["tag"], anc["bpw"]).d() + self.b_of(size) * (m["bpw"] - size)

    def monitor_alert(self, size):
        """user 11-13: no pre-set N, no stop. Soft alert when the size's size-adjusted monitor best has not improved by more
        than the paired SE over its last alert_frac of decisions (>= alert_min decisions). -> (on, improve, se) or None"""
        mc = self.c.get("monitor") or {}; n = self.dec[size]
        if n < int(mc.get("alert_min", 50)): return None
        w0 = n - math.ceil(float(mc.get("alert_frac", 0.2)) * n)
        reads = [m for m in self.mon.get(size, []) if m["tag"] in self.res and m.get("act", "f16") == self.act]
        before = [m for m in reads if m.get("dec_n", 0) <= w0]
        if not before: return None
        sc = lambda m: self.mon_score(size, m)
        b0 = min(before, key=sc); b1 = min(reads, key=sc)
        if b1 is b0: imp, se = 0.0, 0.0
        else:
            P = self.mon_paired(size, b0, b1["tag"], b1["bpw"])
            imp, se = -P.da(), P.se()
        on = imp <= se
        if on != self.alerts.get(size, False):
            self.alerts[size] = on
            self.rec({"ev": "alert", "size": size, "on": on, "dec_n": n, "window_from": w0, "improve": imp, "se": se})
            msg = (f"ALERT {size}: monitor best flat over the last {n - w0} of {n} decisions (size-adjusted {imp:+.4f} <= se {se:.4f}); "
                   f"end it by hand: touch {self.R}/STOP-{size:g}") if on else f"alert {size} cleared: monitor best improved {imp:+.4f} > se {se:.4f}"
            self.log(msg)
            with open(f"{self.R}/events.jsonl", "a") as f:
                f.write(json.dumps({"step": self.res_lines, "label": (f"flat {size:g}" if on else f"unflat {size:g}"), "t": cst()}) + "\n")
        return on, imp, se

    def monitor_refs(self):
        """CONF monitor "refs": [{"tag": "ref-V2-3.00", "mix": EVAL MIX}]: scored once each on the monitor golden, so
        convplot3 (--heldout-re MONITOR_NAME) can put the monitor curves on its V2 scale; never a decision or a fit row"""
        mc = self.c.get("monitor") or {}
        for r in mc.get("refs", []):
            if r["tag"] in self.sub: continue
            line = {"tag": r["tag"], "mix": r["mix"], "slices": [{"name": mc["name"], "parts": {"m": mc["golden"]}}]}
            with open(self.c["evals"], "a") as f: f.write(json.dumps(line) + "\n")
            e = {"ev": "submit", "tag": r["tag"], "size": None, "kind": "monitor_ref", "slice": mc["name"], "hash": None, "early": False,
                 "ref": None, "bytes": None, "bpw": None, "b": 0.0}
            self.rec(e); self.sub[r["tag"]] = e

    def dump_monitor(self):
        json.dump({"curves": {str(k): v for k, v in self.mon.items()}, "alerts": {str(k): v for k, v in self.alerts.items()},
                   "b": {str(k): self.b_of(k) for k in self.sizes}}, open(f"{self.R}/monitor.json", "w"), indent=0)

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
                self.res_off = f.tell(); self.res_lines += 1
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
        self.rec({"ev": "results_offset", "off": self.res_off, "lines": self.res_lines})
        return n

    # ------------------------------------------------------------------------------------------------ decisions
    def paired(self, ref, cand, parts, k=None, rule=None, off=0.0):
        P = SR.Paired(len(parts), self.prule(dict(rule or {"kind": "full"}, guard=self.rule.get("guard")), off, ref["docs"], parts))
        for i, (a, b, p) in enumerate(zip(ref["docs"], cand["docs"], parts)):
            if k is not None and i >= k: break
            if a is None or b is None: break
            P.add(a, b, p)
            if rule is not None and P.check(): break
        return P

    def on_result(self, t, r):
        s = self.sub[t]; size = s["size"]
        if s["kind"] == "monitor_ref": self.log(f"{t}: monitor reference scored: KLD {r.get('kld')}"); return
        self.add_obs(t, r)
        if r.get("fail"): self.log(f"{t}: FAILED ({r.get('err')})"); return
        if r.get("bytes") and not self.M.eval_mix(self.assign(t), self.c["base"])["donors"]:
            mb = self.M.bytes(self.assign(t))                               # menu bytes (with the calibrated offset) vs the container
            if self.bytes_off is None or self.bytes_src == "conf":         # the first donor-free container calibrates the offset
                self.bytes_off = r["bytes"] - (mb - self.M.fixed + self.fixed0); self.M.fixed = self.fixed0 + self.bytes_off; self.bytes_src = "measured"
                self.rec({"ev": "bytes_offset", "off": self.bytes_off, "tag": t, "src": "measured"}); self.log(f"bytes offset {self.bytes_off} (container - menu, from {t})")
            else:
                rel = r["bytes"] / mb - 1
                if abs(rel) > float(self.c.get("bytes_tol", 0.0005)): self.log(f"WARN {t}: container {r['bytes']} bytes vs menu {mb} ({100 * rel:+.4f}%)")
        if s["kind"] == "monitor": self.on_monitor(t, s, r); return
        if s["kind"] == "incumbent":
            self.log(f"{t}: incumbent of {size} scored on {r['slice']}: KLD {r['kld']:.5f} at {s['bpw']:.4f} bpw (container {r['bpw']:.4f})"); return
        act = self.reg.active()
        if r["slice"] != act:
            self.rec({"ev": "decision", "tag": t, "size": size, "decision": "stale", "kind": None})
            k = sum(x is not None for x in r["docs"])
            self.log(f"stale {t} ({s['kind']}) for {size}: scored on retired slice {r['slice']} ({'stopped after ' + str(k) + ' docs' if r.get('stopped') else 'whole slice'}): fit rows only"); return
        cur = self.inc.get(size); parts = self.reg.parts_of(r["slice"])
        if cur is None or cur not in self.res or self.res[cur].get("slice") != r["slice"]:
            self.rec({"ev": "decision", "tag": t, "size": size, "decision": "stale", "kind": None}); return
        ref = self.res[cur]
        bs = s.get("b", 0.0) or 0.0                                       # the b recorded at submit (the hosts stopped on it)
        off = lambda rt: bs * (s["bpw"] - self.sub[rt]["bpw"])
        if r.get("stopped"):
            rt = s["ref"] if s.get("ref") in self.res else cur
            P = self.paired(self.res[rt], r, parts, off=off(rt))
            d = "reject"; ndocs = P.k
        else:
            if s.get("audit") and s.get("ref") in self.res:                       # the rule after the fact
                Pe = self.paired(self.res[s["ref"]], r, parts, rule=self.rule, off=off(s["ref"]))
                Pf = self.paired(self.res[s["ref"]], r, parts, off=off(s["ref"]))
                if Pe.stopped:
                    self.rec({"ev": "bias", "tag": t, "k": Pe.k, "d_prefix": Pe.d(), "d_full": Pf.d(), "se_full": Pf.se(), "off": Pe.off})
                    self.bias.append({"d_prefix": Pe.d(), "d_full": Pf.d()})
                    self.decide(t, size, "reject", Pe, r["slice"], Pe.k); return
            P = self.paired(ref, r, parts, off=off(cur)); d = P.final(); ndocs = P.k
        self.decide(t, size, d, P, r["slice"], ndocs)

    def decide(self, t, size, d, P, sl, ndocs):
        s = self.sub[t]
        self.reg.decided(sl, str(size), ndocs); self.dec[size] += 1
        self.kind_stats[s["kind"]][1] += 1; self.kind_stats[s["kind"]][0] += d == "accept"
        self.rec({"ev": "decision", "tag": t, "size": size, "decision": d, "kind": s["kind"], "k": P.k, "z": P.z(), "logr": P.d(), "logr_pooled": P.d_pooled(),
                  "off": P.off, "se": P.se(), "parts": P.summary()["parts"],
                  "stat_b": P.w is not None, **({"why": P.why} if P.why else {})})
        if d == "accept":
            old = self.inc.get(size); self.inc[size] = t
            self.rec({"ev": "incumbent", "size": size, "tag": t, "prev": old})
            self.log(f"ACCEPT {t} ({s['kind']}) for {size}: log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}) vs {old}; {s['bpw']:.4f} bpw")
            mc = self.c.get("monitor")
            if mc:                                                          # the band's new best -> the held-out monitor
                self.submit(self.newtag(size, "mon"), self.assign(t), size, "monitor", None, early=False, extra={"of": t},
                            slice_entry={"name": mc["name"], "parts": {"m": mc["golden"]}})
        else:
            self.log(f"reject {t} ({s['kind']}) for {size}: log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}) after {P.k} docs" + (" [stopped]" if self.res[t].get("stopped") else "") +
                     (" [guard: per-document mean " + ", ".join(f"{p} {P.docmean(p) + P.off:+.4f}" for p in P.part if P.docmean(p) is not None) + "]" if P.why == "guard" else ""))

    # ------------------------------------------------------------------------------------------------ fit rows
    def add_obs(self, t, r):
        if r.get("fail") or not r.get("docs"): return
        s = self.sub.get(t)
        if s is None or s["kind"] in ("monitor", "monitor_ref"): return   # verify data: never a fit row
        a = self.assign(t); docs, pos = r["docs"], r["pos"]
        N = len(docs); k = sum(1 for x in docs if x is not None)
        if k == 0: return
        g = max(1, math.ceil(N / 20))
        full = k == N
        kq = N if full else (k // g) * g
        if kq < 3: return
        parts = self.reg.parts_of(r["slice"]) if r["slice"] in self.reg.d["slices"] else ["m"] * N
        pset = sorted(set(parts)) if len(set(parts)) > 1 else [None]       # one row per part (w131, #171)
        ref = s.get("ref") if s.get("ref") in self.res else None
        rr = self.res[ref] if ref and ref != t else None
        for p in pset:
            ix = [i for i in range(kq) if p is None or parts[i] == p]
            if len(ix) < 3: continue
            label = f"{r['slice']}|" + (f"{p}|" if p else "") + ("full" if full else f"pre{kq}")
            kl = sum(docs[i] for i in ix) / max(sum(pos[i] for i in ix), 1)
            se = 0.0
            if rr is not None:
                P = SR.Paired(len(ix), {"kind": "full"})
                for i in ix: P.add(rr["docs"][i], docs[i])
                se = P.se() if P.se() != SR.INF else 0.5
                if all(rr["docs"][i] is not None for i in ix):
                    self.obs[(ref, label)] = {"tag": f"{ref}@{label}", "assign": self.assign(ref), "label": label, "size": self.sub[ref]["size"],
                                              "kld": sum(rr["docs"][i] for i in ix) / max(sum(rr["pos"][i] for i in ix), 1), "se": 0.0}
            self.obs[(t, label)] = {"tag": f"{t}@{label}", "assign": a, "label": label, "kld": kl, "se": se, "size": s["size"]}

    @property
    def fit(self):
        return self.fits[self.groups[0]["name"]]

    def fitproc_any(self): return any(p is not None for p in self.fitprocs.values())

    def maybe_refit(self):
        for g in self.groups: self.refit_group(g)

    def refit_group(self, g):
        gn = g["name"]; fp = self.fitprocs[gn]
        if fp is not None:
            if fp[0].poll() is None: return
            proc, out, obs, n_at = fp; self.fitprocs[gn] = None
            if proc.returncode != 0 or not os.path.exists(out):
                self.fit_fail_t = time.time(); self.log(f"fit {gn} FAILED rc {proc.returncode} (see {out}.log); next try in {self.c.get('fit_retry_s', 120)} s"); return
            self.fits[gn] = self.load_fit(out, obs); self.fit_at[gn] = n_at
            self.rec({"ev": "fit_done", "group": gn, "n_results": n_at, "out": out, "report": self.fits[gn]["report"]})
            self.log(f"fit {gn} done on {len(obs)} rows: {self.fits[gn]['report']}")
            if self.sizeadj: self.update_b(gn)
            return
        sz = {float(z) for z in g["sizes"]}
        obs = [o for o in self.obs.values() if g.get("rows", "all") == "all" or o.get("size") in sz]
        due = (self.fits[gn] is None and len(obs) >= int(self.c.get("min_rows_fit", max(2, len(sz))))) or \
              (self.fits[gn] is not None and self.n_results - self.fit_at[gn] >= int(self.c.get("refit_every", 20)))
        if not due or time.time() - getattr(self, "fit_fail_t", 0) < float(self.c.get("fit_retry_s", 120)): return
        obs = obs + self.prior_obs()
        n = len([f for f in os.listdir(f"{self.R}/fits") if f.endswith(".in.json")])
        inp, out = f"{self.R}/fits/fit{n:04d}-{gn}.in.json", f"{self.R}/fits/fit{n:04d}-{gn}.out.json"
        json.dump({"menu": self.c["menu_remote"] if self.c.get("menu_remote") else self.c["menu"], "obs": obs,
                   "opts": self.c.get("fit_opts", {}), "ndraw": int(self.c.get("ndraw", 16)), "seed": n}, open(inp, "w"))
        cmd = [x.replace("{in}", inp).replace("{out}", out) for x in self.c.get("fit_cmd") or [sys.executable, "-I", f"{HERE}/fit.py", "{in}", "{out}"]]
        self.fitprocs[gn] = (subprocess.Popen(cmd, stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), out, obs, self.n_results)
        self.log(f"fit {n:04d} ({gn}) started on {len(obs)} rows ({self.n_results} results)")

    def prior_obs(self):
        p = self.c.get("prior_obs")
        return [json.loads(l) for l in open(p)] if p and os.path.exists(p) else []

    # ------------------------------------------------------------------------------------------------ size adjustment
    B_DEFAULT = [(2.5, 1.6), (3.0, 1.6), (4.0, 1.1), (5.5, 1.1), (6.0, 0.7), (7.0, 0.7)]   # V2 board slopes (TODO 11-13)

    def b_default(self, size):
        pts = [(float(x), float(y)) for x, y in (self.c.get("b_default") or self.B_DEFAULT)]
        if size <= pts[0][0]: return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if size <= x1: return y0 + (y1 - y0) * (size - x0) / (x1 - x0)
        return pts[-1][1]

    def b_of(self, size):
        """the size's current b (0 = raw KLD when CONF size_adjust is false)"""
        if not self.sizeadj: return 0.0
        return self.b.get(size, self.b_default(size))

    def slope(self, size, F):
        """-d log KLD / d bpw of the fitted model's knapsack optimum around the size (size -/+ b_delta, central difference;
        one-sided at the menu's ends), from the size's incumbent. None if it cannot be computed."""
        pred = F["pred"]; db = float(self.c.get("b_delta", 0.1))
        t = self.inc.get(size); a0 = self.assign(t) if t in self.sub else self.init_assign(size)
        self.feasible(size); self.window(size); pts = []
        for s2 in (size - db, size, size + db):
            hi = s2 * self.M.P / 8; lo = (s2 - self.tol) * self.M.P / 8
            if self._minb > hi or self._maxb < lo: continue
            m = MV.mckp(pred, a0, hi, lo)
            if m is not None: pts.append((self.M.bpw(m), math.log(pred.kl(m))))
        if len(pts) < 2: return None
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        if x1 - x0 < 1e-4: return None
        b = -(y1 - y0) / (x1 - x0)
        return b if math.isfinite(b) else None

    def update_b(self, gn):
        """after a refit: each of the group's sizes gets b = the new fit's local slope (clamped to CONF b_range)"""
        F = self.fits[gn]; lo_b, hi_b = self.c.get("b_range", [0.05, 6.0])
        for size in self.sizes:
            if self.gname[size] != gn or not self.feasible(size): continue
            try: b = self.slope(size, F)
            except Exception as e:                          # noqa: BLE001 -- keep the old b
                self.log(f"b {size}: slope failed ({e!r}); keeps {self.b_of(size):.3f}"); continue
            if b is None: continue
            b = min(hi_b, max(lo_b, b)); self.b[size] = b
            self.rec({"ev": "b", "size": size, "b": b, "group": gn, "n_results": self.fit_at[gn]})
        self.log("b (size-adjusted score) " + " ".join(f"{z:g}:{self.b_of(z):.3f}" for z in self.sizes if self.gname[z] == gn))

    def freeze_b(self, out, force=False):
        """TODO 'Final pick score': each release band's b, fixed before the ver4 reads (finalpick.py --b)"""
        if os.path.exists(out) and not force: raise SystemExit(f"{out} exists (frozen); --force to overwrite")
        bands = release_bands(self.sizes)
        d = {"frozen": cst(), "conf": os.path.abspath(self.cp), "size_adjust": self.sizeadj,
             "bands": {str(T): {"b": self.b.get(T, self.b_default(T)), "src": "fit" if T in self.b else "default", "band": list(bands[T]),
                                "fit_results": self.fit_at[self.gname[T]]} for T in self.sizes}}
        json.dump(d, open(out + ".part", "w"), indent=1); os.replace(out + ".part", out)
        self.rec({"ev": "b_frozen", "out": out, "bands": d["bands"]})
        return d

    def load_fit(self, out, obs):
        import fit as FT
        d = json.load(open(out))
        Mo = FT.build(self.M, obs, self.c.get("fit_opts", {}))
        import numpy as np
        x = np.array(d["x"]); draws = [MV.Pred(Mo, np.array(v)) for v in d.get("draws", [])]
        return {"Mo": Mo, "x": x, "pred": MV.Pred(Mo, x), "draws": draws, "report": d["report"], "model": d["model"]}

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
        under = [n for n in lv if self.M.bpw(self.M.uniform(n, h, e)) <= size - float(self.c.get("init_margin", 0.005))]   # the byte offset is calibrated on these evals
        return self.M.uniform(under[-1] if under else lv[0], h, e)

    def feasible(self, size):
        """a size the menu can reach at all (its cheapest mix fits under the target)"""
        if not hasattr(self, "_minb"): self._minb = self.M.fixed + sum(t["opts"][0]["bytes"] for t in self.M.T)
        ok = self._minb <= self.window(size)[1]
        if not ok and size not in getattr(self, "_infeasible", set()):
            self._infeasible = getattr(self, "_infeasible", set()) | {size}
            self.log(f"size {size} below the menu's cheapest mix ({self._minb * 8 / self.M.P:.4f} bpw): skipped (add lower levels)")
        return ok

    def ensure_incumbents(self, act):
        """every size has an incumbent scored (whole slice) on the active slice"""
        out = set(self.outstanding())
        for size in self.sizes:
            if not self.feasible(size): continue
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
        if all(f is None for f in self.fits.values()): return 0
        out = self.outstanding(); depth = int(self.c.get("queue_depth", 6)); made = 0
        inflight = collections.Counter(self.sub[t]["size"] for t in out)
        shares = MV.kind_shares({k: tuple(v) for k, v in self.kind_stats.items()}, self.c.get("kind_share0", MV.SHARE0))
        kcount = collections.Counter(self.sub[t]["kind"] for t in self.sub if self.sub[t]["kind"] in shares)
        tries = 0; failed = set()
        while len(out) + made < depth and tries < 4 * len(self.sizes):
            tries += 1
            ready = [s for s in self.sizes if s not in self.stopped_sizes and self.feasible(s) and self.inc.get(s) in self.res and self.res[self.inc[s]].get("slice") == act and s not in failed
                     and self.fits[self.gname[s]] is not None]
            if not ready: break
            w = self.c.get("size_weights", {})
            size = min(ready, key=lambda s: ((self.dec[s] + inflight[s]) / float(w.get(str(s), 1.0)), self.rng.random()))
            a = self.assign(self.inc[size]); lo, hi = self.window(size)
            kind = MV.pick_kind(shares, kcount, self.rng)
            F = self.fits[self.gname[size]]
            c = MV.generate(kind, F["pred"], F["draws"], a, lo, hi, self.seen, self.rng)
            if c is None:
                for k2 in sorted(shares, key=lambda k: -shares[k]):
                    if k2 == kind: continue
                    c = MV.generate(k2, F["pred"], F["draws"], a, lo, hi, self.seen, self.rng)
                    if c is not None: kind = k2; break
            if c is None:
                failed.add(size)
                if not self.nogen.get(size): self.log(f"no new candidate for {size} (window {lo * 8 / self.M.P:.4f}-{hi * 8 / self.M.P:.4f} bpw)")
                self.nogen[size] = True; continue
            self.nogen[size] = False
            pr = F["pred"]; dpred = math.log(pr.kl(c) / pr.kl(a))
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
            if act: self.rec({"ev": "rotate", "to": act, "from": None}); self.log(f"slice {act} active"); self.set_act(act)
            return act
        if self.reg.due(act, int(self.c.get("rotate_after", 25)), self.c.get("rotate_unit", "slice")):
            nxt = [n for n, s in self.reg.d["slices"].items() if s["status"] == "ready"]
            if nxt:
                new = self.reg.activate_next(); self.rec({"ev": "rotate", "to": new, "from": act})
                self.log(f"ROTATE {act} -> {new}"); self.set_act(new); return new
            if not getattr(self, "_warned", False): self.log(f"rotation due on {act} but no ready slice: build one (slices.py)"); self._warned = True
        return act

    def set_act(self, slice_name):
        """at a rotation: CONF "act" takes effect (w132, #172); the incumbents' monitor reads are redone in the new mode"""
        want = self.c.get("act") or "f16"
        if want == self.act: return
        old = self.act; self.act = want
        self.rec({"ev": "act", "mode": want, "act": want, "from": old, "slice": slice_name})
        self.log(f"ACT MODE {old} -> {want} from slice {slice_name}: incumbents rescored on it in {want}")
        with open(f"{self.R}/events.jsonl", "a") as f:                  # convplot3 --events: a marker at the switch
            f.write(json.dumps({"step": self.res_lines, "label": f"act {want}", "t": cst()}) + "\n")
        mc = self.c.get("monitor")
        if mc and self.c.get("act_monitor_rescore", True):
            for size in self.sizes:
                t = self.inc.get(size)
                if t is None or size in self.stopped_sizes: continue
                self.submit(self.newtag(size, "mon"), self.assign(t), size, "monitor", None, early=False, extra={"of": t, "act_rescore": True},
                            slice_entry={"name": mc["name"], "parts": {"m": mc["golden"]}})

    # ------------------------------------------------------------------------------------------------ report
    def kld_norm(self, t, r):
        """a whole-slice KLD on the fit's common scale: per part (w131, #171) KLD_p / exp(offset of "S|p|full"), the x part
        un-compressed by the fit's lamx, then the parts' geometric mean (the accept statistic's equal weights)"""
        md = (self.fits[self.gname[self.sub[t]["size"]]] or {}).get("model", {}); soff = md.get("slices", {}); lam = md.get("lamx", 1.0) or 1.0
        parts = self.reg.parts_of(r["slice"]) if r["slice"] in self.reg.d["slices"] else ["m"] * len(r["docs"])
        if len(set(parts)) < 2: return r["kld"] / math.exp(soff.get(f"{r['slice']}|full", 0.0))
        lv = []
        for p in sorted(set(parts)):
            ix = [i for i, q in enumerate(parts) if q == p]
            kp = sum(r["docs"][i] for i in ix) / max(sum(r["pos"][i] for i in ix), 1)
            ln = math.log(kp) - soff.get(f"{r['slice']}|{p}|full", 0.0)
            lv.append(ln / lam if p == "x" and f"{r['slice']}|{p}|full" in soff else ln)
        return math.exp(sum(lv) / len(lv))

    def report(self):
        bands = release_bands(self.sizes)
        pts = []
        for t, r in self.res.items():
            if r.get("fail") or not r.get("docs") or any(x is None for x in r["docs"]) or self.sub[t]["kind"] in ("monitor", "monitor_ref"): continue
            pts.append({"tag": t, "bpw": self.sub[t]["bpw"], "bpw_container": r["bpw"], "kld": r["kld"], "slice": r["slice"],
                        "kld_norm": self.kld_norm(t, r), "size": self.sub[t]["size"]})
        fr = {}; topn = int(self.c.get("topn", 5))
        for T, (lo, hi) in bands.items():
            p = sorted([dict(x) for x in pts if lo <= (x["bpw"] or 0) <= hi], key=lambda x: (x["bpw"], x["kld_norm"]))
            best = math.inf
            for x in p:
                x["nondominated"] = x["kld_norm"] < best; best = min(best, x["kld_norm"])
                x["score"] = self.mscore(T, x["kld_norm"], x["bpw"])
            by = collections.defaultdict(list)                              # an assignment scored on several slices: mean score
            for x in p: by[self.sub[x["tag"]]["hash"]].append(x)
            top = sorted(({"tags": [x["tag"] for x in v], "bpw": v[0]["bpw"], "score": sum(x["score"] for x in v) / len(v), "n": len(v)}
                          for v in by.values()), key=lambda x: x["score"])[:topn]
            fr[str(T)] = {"band": [lo, hi], "b": self.b_of(T), "incumbent": self.inc.get(T), "stopped": T in self.stopped_sizes,
                          "alert": self.alerts.get(T, False),
                          "monitor": [{"tag": m["tag"], "of": m.get("of"), "kld": m["kld"], "bpw": m.get("bpw"),
                                       "score": self.mon_score(T, m) if m.get("bpw") and m["tag"] in self.res else None,
                                       "stats": m.get("stats")} for m in self.mon.get(T, [])],
                          "top": top, "points": p}
        json.dump(fr, open(f"{self.R}/frontier.json.part", "w"), indent=0); os.replace(f"{self.R}/frontier.json.part", f"{self.R}/frontier.json")
        nd = sum(self.dec.values()); acc = sum(v[0] for v in self.kind_stats.values())
        stopped = sum(1 for r in self.res.values() if r.get("stopped"))
        line = (f"{cst()} | slice {self.reg.active()} ({self.act}) | submitted {len(self.sub)}, results {len(self.res)} ({stopped} stopped early), "
                f"outstanding {len(self.outstanding())} | decisions {nd}, accepts {acc} | kinds " +
                " ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(self.kind_stats.items())) +
                " | b " + " ".join(f"{z:g}:{self.b_of(z):.2f}" for z in self.sizes) +
                (" | ALERT " + " ".join(f"{z:g}" for z, v in sorted(self.alerts.items()) if v) if any(self.alerts.values()) else "") +
                " | fits " + " ".join(f"{g}:{'none' if not f else f['report'].get('rows')}" + ("*" if self.fitprocs[g] else "") for g, f in self.fits.items()))
        open(f"{self.R}/status.txt", "w").write(line + "\n")
        return line

    # ------------------------------------------------------------------------------------------------ main
    def step(self):
        self.ingest()
        for size in self.sizes:                                             # sizes ended by hand (alert-only monitor)
            if size not in self.stopped_sizes and os.path.exists(f"{self.R}/STOP-{size:g}"):
                self.stopped_sizes.add(size); self.rec({"ev": "size_stop", "size": size, "by": "hand"}); self.log(f"SIZE STOP {size} (by hand)")
            if self.c.get("monitor"): self.monitor_alert(size)
        self.maybe_refit()
        act = self.rotate()
        if act is None: return "no slice"
        go = self.c.get("go") or self.c.get("dry")
        if not go: return "no go (CONF go / dry)"
        self.ensure_incumbents(act)
        self.monitor_refs()
        self.generate(act)
        return self.report()

    def run(self, steps=None, sleep=None):
        sleep = float(self.c.get("poll_s", 5)) if sleep is None else sleep
        i = 0
        while steps is None or i < steps:
            if os.path.exists(f"{self.R}/STOP"): self.log("STOP file: exit"); break
            st = self.step(); i += 1
            if sleep: time.sleep(sleep)
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="driver.py")
    ap.add_argument("conf"); ap.add_argument("--status", action="store_true"); ap.add_argument("--steps", type=int)
    ap.add_argument("--freeze-b", metavar="OUT", help="write each release band's current b (the final picks' size adjustment) and exit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dump-obs", help="write the fit input (every row) to this file and exit (fit.py check / offline fits)")
    a = ap.parse_args(argv)
    D = Driver(a.conf)
    if a.status: print(D.report()); return 0
    if a.freeze_b: print(json.dumps(D.freeze_b(a.freeze_b, a.force), indent=1)); return 0
    if a.dump_obs:
        json.dump({"menu": D.c.get("menu_remote") or D.c["menu"], "obs": list(D.obs.values()) + D.prior_obs(), "opts": D.c.get("fit_opts", {})},
                  open(a.dump_obs, "w")); print(f"{a.dump_obs}: {len(D.obs)} rows"); return 0
    return D.run(a.steps)


if __name__ == "__main__":
    sys.exit(main())
