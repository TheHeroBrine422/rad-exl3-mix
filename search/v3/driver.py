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
  monitor      (CONF "monitor": {"name", "golden", "alert_min", "alert_frac", "when"}): every accept ("when": "accept", default;
               "rotate" = each size's incumbent once per slice rotation, #183) (a band's new best) is also
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
Two-stage accept (#189, RULES-confirm.md; CONF "confirm": {"z", "pairs": [[search, confirm], ..], "sizes", "golive_from_line",
"tries"}; absent = the driver above, event for event): a stage-1 accept or pool join of a listed size is TENTATIVE ("tentative"
event; it counts as the search slice's decision, nothing else changes). The candidate and its reference (the size's current
incumbent; for a join that would only replace the pool's worst member: the current worst member) are scored whole on the
confirmation slice paired with the stage-1 slice (kind "confirm", no early stop; one eval per (assignment hash, slice), cached:
a rotation's incumbent retag is the same model; "confirm_submit" events). Stage 2 = the stage-1 statistic (driver.paired on
the confirmation slice's per-document sums, strata, size offset with the candidate's recorded b, CONF guard) with z = da / se
(Paired.z); confirmed iff z < CONF z (-1). Confirmed -> applied against the CURRENT state (rule 3: the incumbent / worst at the
end of the confirmation; K > 1 waits until the pool is scored on the active slice); else rejected. The verdict is a "confirm"
event and a "decision" event with "stage": 2 ("accept" | "join" (joined the pool, not incumbent) | "reject" with "why":
"confirm"; "logr" = stage 2's log r, "logr1" = stage 1's), so convplot4 chains the stage-2 log r with no change. Stage-2
decisions never count towards rotate_after (registry "confirm_decisions"). Confirmation slices carry "role": "confirm" in the
registry (never rotated to; the driver refuses to start without it). Go-live (rule 7, once, CONF golive_from_line): each
listed size's incumbent vs its incumbent at that state line, on the first pair's confirmation slice, same test; not
confirmed -> the size reverts to the switch-time model and its pool to that model alone ("golive" / "golive_size" events);
a size's tentatives wait until its go-live check is resolved.
State: ROOT/state.jsonl (append-only: submit, result, decision, incumbent, rotate, fit), ROOT/assign/<tag>.json.
Restart = the same command (state replayed; outstanding evals stay in the queue)."""
import argparse, collections, json, math, os, random, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import menu as MN            # noqa: E402
import moves as MV           # noqa: E402
import seqrule as SR         # noqa: E402
import slices as SL          # noqa: E402
import poolrule as PR        # noqa: E402

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
        # decode term (#183, user 11-13 "KL + lambda_d decode"; w159's #176-02 table): CONF "decode": {"menu": a menu with
        # dec_us / dec_us_8k per option, "lambda_rel": log-KLD per log decode step (0.5 = a 1 % faster step is worth 0.5 % KL),
        # "w8k": weight of the 8k column (0.5), "alpha": the e2e slope, "F_ms": the fixed part of a step}. Every decision /
        # early-stop offset gets lambda_rel (dec_B - dec_A) / step_A beside b (bpw_B - bpw_A); move generation stays KL-model
        # driven. Absent or lambda 0 = KL + size only.
        dc = c.get("decode") or {}; self.lam_d = float(dc.get("lambda_rel", 0.0)); self.decus = {}; self.dec_F = float(dc.get("F_ms", 4.08))
        if dc.get("menu"):
            import gzip
            mm = json.load(gzip.open(dc["menu"]) if dc["menu"].endswith(".gz") else open(dc["menu"])); w8 = float(dc.get("w8k", 0.5))
            al = float(dc.get("alpha", 0.976))
            for t in mm["tensors"]:
                for o in t["opts"]:
                    if o.get("dec_us") is not None:
                        self.decus[(t["w"], o["o"])] = al * ((1 - w8) * o["dec_us"] + w8 * o.get("dec_us_8k", o["dec_us"])) / 1000.0
        # fit groups (TODO "Low sizes in parallel": a joint fit for the low sizes vs main-only for 2-7 is the lead's call):
        # [{"name", "sizes", "rows": "all" | "own"}]; default one group over every size and row
        self.groups = c.get("fit_groups") or [{"name": "all", "sizes": self.sizes, "rows": "all"}]
        self.gname = {float(z): g["name"] for g in self.groups for z in g["sizes"]}
        self.fits = {g["name"]: None for g in self.groups}       # {"Mo", "x", "pred", "draws", "report", "model"}
        self.fitprocs = {g["name"]: None for g in self.groups}; self.fit_at = {g["name"]: 0 for g in self.groups}
        self.res_off = 0; self.bias = []; self.nogen = {}; self.fixed0 = self.M.fixed; self.bytes_off = None; self.bytes_src = None
        self.mon = collections.defaultdict(list); self.stopped_sizes = set()
        # elite pool (#188, RULES-pool): K members per size; K = 1 is today's driver (no pool events, no extra random draws)
        self.K = int(c.get("pool", 1)); self.pool = {}; self.cross_share0 = float(c.get("cross_share0", 0.15))
        self.reseed_c = c.get("reseed"); self.reseed_seen = False; self.reseed_on = None; self.reseed_plan = None
        self.reseed_done = set(); self.reselected = {}
        # two-stage accept (#189, RULES-confirm): CONF "confirm" absent = None = today's driver (no new events, no draws)
        self.cf = c.get("confirm") or None; self.tent = {}; self.cfx = {}; self.golive = None; self.golive_res = {}; self.golive_end = False
        self.cf_warned = set()
        if self.cf:
            self.cf_z = float(self.cf.get("z", -1.0)); self.cf_pairs = {a: b for a, b in self.cf["pairs"]}
            self.cf_sizes = {float(z) for z in self.cf.get("sizes", self.sizes)}; self.cf_tries = int(self.cf.get("tries", 2))
            self.check_confirm_registry()
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
            if ev == "submit":
                self.sub[e["tag"]] = e; self.seen.add(e["hash"])
                if e.get("kind") == "confirm": self.cfx.setdefault((e["hash"], e["slice"]), []).append(e["tag"])
            elif ev == "result": self.res[e["tag"]] = e["r"]; self.n_results += 1
            elif ev == "incumbent": self.inc[e["size"]] = e["tag"]
            elif ev == "decision" and e.get("stage") == 2:                  # (#189) counted at its tentative; the accept here
                if e.get("kind"): self.kind_stats[e["kind"]][0] += e["decision"] in ("accept", "join")
            elif ev == "decision":
                self.dec[e["size"]] += 1
                if e.get("kind"): self.kind_stats[e["kind"]][1] += 1; self.kind_stats[e["kind"]][0] += e["decision"] == "accept"
            elif ev == "tentative":
                self.tent[e["tag"]] = e; self.dec[e["size"]] += 1
                if e.get("kind"): self.kind_stats[e["kind"]][1] += 1
            elif ev == "confirm": self.tent.pop(e["tag"], None)
            elif ev == "golive":
                if e["phase"] == "start": self.golive = e
                else: self.golive_end = True
            elif ev == "golive_size": self.golive_res[e["size"]] = e
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
            elif ev == "pool": self.pool[e["size"]] = list(e["members"])
            elif ev == "reseed_start": self.reseed_seen = True; self.reseed_on = e["slice"]; self.reseed_plan = {float(k): list(v) for k, v in e["plan"].items()}
            elif ev == "reseed_abandon": self.reseed_on = None; self.reseed_plan = None
            elif ev == "reseed": self.reseed_done.add(e["size"]); self.reselected[e["size"]] = e["slice"]
            elif ev == "reselect": self.reselected[e["size"]] = e["slice"]
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
        bpw = self.M.bpw(a); dms = self.dec_ms(a)
        if early and ref_tag:
            r = self.res[ref_tag]; off = bs * (bpw - self.sub[ref_tag]["bpw"])
            dr = self.dec_of(ref_tag)
            if self.lam_d and dms is not None and dr is not None and kind not in ("monitor", "incumbent"):
                off += self.lam_d * (dms - dr) / (self.dec_F + dr)
            st = self.reg.strata_of(slice_name)
            line["early_stop"] = {"rule": self.prule(self.rule, off, r["docs"], self.reg.parts_of(slice_name), st), "ref": [[b, n] for b, n in zip(r["docs"], r["pos"])],
                                  "parts": self.reg.parts_of(slice_name), "ref_tag": ref_tag, **({"strata": st} if st else {})}
        with open(self.c["evals"], "a") as f: f.write(json.dumps(line) + "\n")
        e = {"ev": "submit", "tag": tag, "size": size, "kind": kind, "slice": slice_name or (slice_entry or {}).get("name"), "hash": h, "early": bool(early and ref_tag),
             "ref": ref_tag, "bytes": self.M.bytes(a), "bpw": bpw, "b": bs, **({"dec": dms, "lam_d": self.lam_d} if dms is not None else {}), **({"act": self.act} if self.act != "f16" else {}), **(extra or {})}
        self.rec(e); self.sub[tag] = e
        if kind != "monitor": self.seen.add(h)
        if kind == "confirm": self.cfx.setdefault((h, e["slice"]), []).append(tag)
        return tag

    def dec_ms(self, a):
        """the predicted decode step (ms, without the fixed part) of assignment a; None without a decode table"""
        if not self.decus: return None
        return sum(self.decus.get((w, o), 0.0) for w, o in a.items())

    def dec_of(self, t):
        """a submitted tag's predicted decode (ms): its submit event's, else from its assignment file (tags submitted
        before the decode table was configured)"""
        e = self.sub.get(t) or {}
        if e.get("dec") is None and self.decus and os.path.exists(f"{self.R}/assign/{t}.json"):
            e["dec"] = self.dec_ms(json.load(open(f"{self.R}/assign/{t}.json")))
        return e.get("dec")

    def doff(self, t, rt):
        """the decode part of a size-adjusted offset: lambda_rel (dec_t - dec_rt) / step_rt, 0 without the table / a recorded dec"""
        x, y = self.dec_of(t), self.dec_of(rt)
        return self.lam_d * (x - y) / (self.dec_F + y) if self.lam_d and x is not None and y is not None else 0.0

    def prule(self, rule, off, ref_docs, parts, strata=None):
        """a seqrule rule with the size offset and the capped statistic's weights (from the incumbent's whole slice); strata
        (RULES-strata §4, #186): the cap is within (part, stratum) and the rule says "strata": true"""
        r = dict(rule, off=off)
        w = SR.cap_weights([x or 0.0 for x in ref_docs], parts, self.cap, strata=strata) if self.cap else None
        if w is not None: r["w"] = w
        if strata is not None: r["strata"] = True
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

    # ------------------------------------------------------------------------------------------------ elite pool (#188, RULES-pool)
    def members(self, size):
        """the size's pool members (tags); K = 1: the incumbent alone (today's driver)"""
        t = self.inc.get(size)
        if self.K <= 1 or size not in self.pool: return [t] if t is not None else []
        return list(self.pool[size])

    def scored_on(self, size, act):
        """every member has a whole-slice score on the active slice (K > 1: and none of them failed)"""
        mem = self.members(size)
        return bool(mem) and all(m in self.res and self.res[m].get("slice") == act and (self.K <= 1 or not self.res[m].get("fail")) for m in mem)

    def reseed_wait(self, size, act):
        """a size whose re-seed set is still being scored on the active slice generates nothing yet"""
        return self.reseed_on == act and self.reseed_plan is not None and size in self.reseed_plan and size not in self.reseed_done

    def worst(self, size):
        """RULES-pool §2: the pool's worst member: the largest size-adjusted paired statistic against the incumbent, on the active slice"""
        act = self.reg.active(); parts = self.reg.parts_of(act); st = self.reg.strata_of(act)
        inc = self.inc[size]; bs = self.b_of(size); da = {}
        for m in self.members(size):
            if m == inc: da[m] = 0.0; continue
            off = bs * (self.sub[m]["bpw"] - self.sub[inc]["bpw"]) + (self.doff(m, inc) if bs else 0.0)
            da[m] = self.paired(self.res[inc], self.res[m], parts, off=off, strata=st).da()
        return PR.worst_of(inc, da)

    def reference(self, size):
        """RULES-pool §2: a new candidate's early-stop reference: the worst member once the pool is full, else the incumbent"""
        if self.K <= 1 or len(self.members(size)) < self.K: return self.inc[size]
        return self.worst(size)

    def parent(self, size):
        """RULES-pool §4: the incumbent (K = 1: always, no draw); else the incumbent w.p. 0.6, else another member"""
        return self.inc[size] if self.K <= 1 else PR.choose_parent(self.inc[size], self.members(size), self.rng)

    def gen_move(self, kind, size, F, a, lo, hi, other):
        """one candidate of a move kind; cross (K > 1) takes the second parent (moves.generate_cross); the others as before.
        size: for the non-cross branch only (RULES-187 reranker hook goes here; see SWITCH-186-188.md)"""
        if kind == "cross": return MV.generate_cross(F["pred"], a, other, lo, hi, self.seen, self.rng) if other is not None else None
        return MV.generate(kind, F["pred"], F["draws"], a, lo, hi, self.seen, self.rng)

    def set_incumbent(self, size, t):
        old = self.inc.get(size); self.inc[size] = t
        self.rec({"ev": "incumbent", "size": size, "tag": t, "prev": old}); return old

    def pool_judge(self, t, s, r, parts, st, off):
        """RULES-pool §3: a whole-slice candidate against the reference (the worst member once the pool is full, else the
        incumbent) and against the incumbent. -> (P for the decision, "accept" | "reject", the join info or None)"""
        size = s["size"]; inc = self.inc[size]
        full = len(self.members(size)) >= self.K
        W = self.worst(size) if full else inc
        Pi = self.paired(self.res[inc], r, parts, off=off(inc), strata=st); acc_i = Pi.final() == "accept"
        if W == inc: Pw, acc_w = Pi, acc_i
        else: Pw = self.paired(self.res[W], r, parts, off=off(W), strata=st); acc_w = Pw.final() == "accept"
        mode, new_inc = PR.join_action(full, acc_i, acc_w)
        if mode is None: return Pi, "reject", None
        info = {"join": mode, "inc": new_inc, "vs": W, **({"replaced": W} if mode == "replace" else {})}
        return (Pi if acc_i else Pw), "accept", info

    def pool_join(self, t, size, info):
        """a join: added (pool not full) or replacing the worst member; recorded as a 'pool' event"""
        mem = self.members(size)
        new = mem + [t] if info["join"] == "add" else [t if m == info["replaced"] else m for m in mem]
        self.pool[size] = new
        self.rec({"ev": "pool", "size": size, "members": new, "why": "join", "tag": t, "join": info["join"], "replaced": info.get("replaced")})

    def monitor_on_change(self, size, t):
        """the held-out monitor reads the incumbent when it changes (CONF monitor when: accept), as decide() does"""
        mc = self.c.get("monitor")
        if mc and mc.get("when", "accept") == "accept":
            self.submit(self.newtag(size, "mon"), self.assign(t), size, "monitor", None, early=False, extra={"of": t},
                        slice_entry={"name": mc["name"], "parts": {"m": mc["golden"]}})

    def ensure_pool(self, act):
        """RULES-pool §5 (K > 1): the incumbent as ensure_incumbents; every other member is rescored in full on the active slice.
        A change of the pool's list (a rescored member, a dropped failure, the incumbent's new tag) is a 'pool' event"""
        before = {z: self.members(z) for z in self.sizes}; inc_before = dict(self.inc)
        self.ensure_incumbents(act)
        out = set(self.outstanding())
        for size in self.sizes:
            if self.inc.get(size) is None or not self.feasible(size): continue
            mem = [self.inc[size] if m == inc_before.get(size) else m for m in self.members(size)]
            new = []
            for m in mem:
                r = self.res.get(m)
                if m == self.inc[size] or m in out or (r is not None and r.get("slice") == act and not r.get("fail")): new.append(m); continue
                if r is not None and r.get("slice") == act:                              # a failed member leaves the pool
                    self.log(f"{m}: pool member of {size} failed on {act}: dropped"); continue
                tag = self.newtag(size, "inc")
                self.submit(tag, self.assign(m), size, "incumbent", act, early=False, extra={"of": m})
                out.add(tag); new.append(tag)
            if new != before[size]:
                self.pool[size] = new; self.rec({"ev": "pool", "size": size, "members": new, "why": "rescore"})

    def reseed_step(self, act):
        """RULES-pool §6 (once, K > 1): when CONF reseed.slice becomes active, the listed sizes' frontier top is scored in full on
        it; when a size's set (with its incumbent) is complete, the ranking picks its incumbent and pool"""
        rc = self.reseed_c
        if not rc or self.K <= 1: return
        if self.reseed_on is not None and self.reseed_on != act:
            self.rec({"ev": "reseed_abandon", "slice": self.reseed_on})
            self.log(f"re-seed on {self.reseed_on} abandoned: the slice rotated before its set was complete")
            self.reseed_on = None; self.reseed_plan = None
        if self.reseed_on is None and not self.reseed_seen and act == rc["slice"]: self.reseed_start(act, rc)
        if self.reseed_on != act or not self.reseed_plan: return
        for size in sorted(self.reseed_plan):
            tags = self.reseed_plan[size]
            if size in self.reseed_done or any(x not in self.res for x in tags): continue
            inc = self.inc.get(size)
            if inc is None or not (inc in self.res and self.res[inc].get("slice") == act and not self.res[inc].get("fail")): continue
            self.reseed_apply(size, act, tags)

    def reseed_start(self, act, rc):
        src = json.load(open(rc.get("source", f"{self.R}/frontier.json")))
        plan = {}; ntop = int(rc.get("top", 3))
        for size in sorted(float(z) for z in rc["sizes"]):
            t0 = self.inc.get(size)
            if t0 is None or not self.feasible(size): continue
            seen_h = {MV.ahash(self.assign(t0))}; picks = []
            for x in (src.get(str(size)) or {}).get("top", []):
                tg = x["tags"][0]; h = MV.ahash(self.assign(tg))
                if h in seen_h: continue
                seen_h.add(h); picks.append(tg)
                if len(picks) >= ntop: break
            plan[size] = []
            for tg in picks:
                tag = self.newtag(size, "reseed")
                self.submit(tag, self.assign(tg), size, "reseed", act, early=False, extra={"of": tg})
                plan[size].append(tag)
        self.reseed_seen = True; self.reseed_on = act; self.reseed_plan = {k: v for k, v in plan.items() if v}
        self.rec({"ev": "reseed_start", "slice": act, "source": rc.get("source"), "plan": {str(k): v for k, v in self.reseed_plan.items()}})
        self.log(f"RE-SEED on {act}: {sum(len(v) for v in self.reseed_plan.values())} candidates for {len(self.reseed_plan)} sizes (frontier top {ntop}, incumbent rescored)")

    def reseed_apply(self, size, act, tags):
        inc = self.inc[size]; parts = self.reg.parts_of(act); st = self.reg.strata_of(act); scored = []
        for x in tags:
            if self.res[x].get("fail") or self.res[x].get("slice") != act: continue
            bs = self.sub[x].get("b", 0.0) or 0.0
            off = bs * (self.sub[x]["bpw"] - self.sub[inc]["bpw"]) + (self.doff(x, inc) if bs else 0.0)
            P = self.paired(self.res[inc], self.res[x], parts, off=off, strata=st)
            scored.append((x, P.da(), P.final() == "accept"))
        new, pool = PR.reseed_pick(inc, scored, self.K)
        self.reseed_done.add(size); self.reselected[size] = act
        self.rec({"ev": "reseed", "size": size, "slice": act, "inc": new, "pool": pool, "prev": inc, "ranking": [[x, round(da, 6), a] for x, da, a in scored]})
        self.pool[size] = pool; self.rec({"ev": "pool", "size": size, "members": pool, "why": "reseed"})
        if new != inc: self.set_incumbent(size, new); self.monitor_on_change(size, new)
        self.log(f"RE-SEED {size:g} on {act}: incumbent {'-> ' + new if new != inc else 'kept'}; pool {pool} ({len(scored)} scored, {sum(1 for _, _, a in scored if a)} accepted)")

    def reselect(self, act):
        """RULES-pool §5 (K > 1): once per slice and size, when every member is scored on the active slice: the accepted member with
        the lowest statistic against the incumbent becomes the incumbent (no new evaluation)"""
        if self.K <= 1: return
        parts = self.reg.parts_of(act); st = self.reg.strata_of(act)
        for size in self.sizes:
            if self.reselected.get(size) == act or self.inc.get(size) is None or not self.scored_on(size, act) or self.reseed_wait(size, act): continue
            inc = self.inc[size]; bs = self.b_of(size); scored = []
            for m in self.members(size):
                if m == inc: continue
                off = bs * (self.sub[m]["bpw"] - self.sub[inc]["bpw"]) + (self.doff(m, inc) if bs else 0.0)
                P = self.paired(self.res[inc], self.res[m], parts, off=off, strata=st)
                scored.append((m, P.da(), P.final() == "accept"))
            new = PR.reselect_pick(inc, scored)
            self.reselected[size] = act
            self.rec({"ev": "reselect", "size": size, "slice": act, "inc": new or inc, "ranking": [[m, round(da, 6), a] for m, da, a in scored]})
            if new is not None:
                self.set_incumbent(size, new); self.monitor_on_change(size, new)
                self.log(f"RESELECT {size:g} on {act}: incumbent {inc} -> {new}")

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
    def paired(self, ref, cand, parts, k=None, rule=None, off=0.0, strata=None):
        P = SR.Paired(len(parts), self.prule(dict(rule or {"kind": "full"}, guard=self.rule.get("guard")), off, ref["docs"], parts, strata))
        for i, (a, b, p) in enumerate(zip(ref["docs"], cand["docs"], parts)):
            if k is not None and i >= k: break
            if a is None or b is None: break
            P.add(a, b, p, strata[i] if strata else None)
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
        if s["kind"] == "reseed":                                          # RULES-pool §6: decided when the size's set is complete
            self.log(f"{t}: re-seed candidate of {size} scored on {r['slice']}: KLD {r['kld']:.5f}"); return
        if s["kind"] == "confirm":                                         # (#189) decided in confirm_step (a fit row above)
            self.log(f"{t}: confirmation eval of {s.get('of')} ({size}) scored on {r['slice']}: KLD {r['kld']:.5f}"); return
        act = self.reg.active()
        if r["slice"] != act:
            self.rec({"ev": "decision", "tag": t, "size": size, "decision": "stale", "kind": None})
            k = sum(x is not None for x in r["docs"])
            self.log(f"stale {t} ({s['kind']}) for {size}: scored on retired slice {r['slice']} ({'stopped after ' + str(k) + ' docs' if r.get('stopped') else 'whole slice'}): fit rows only"); return
        cur = self.inc.get(size); parts = self.reg.parts_of(r["slice"]); st = self.reg.strata_of(r["slice"])
        if cur is None or cur not in self.res or self.res[cur].get("slice") != r["slice"] or (self.K > 1 and not self.scored_on(size, r["slice"])):
            self.rec({"ev": "decision", "tag": t, "size": size, "decision": "stale", "kind": None}); return
        ref = self.res[cur]
        bs = s.get("b", 0.0) or 0.0                                       # the b recorded at submit (the hosts stopped on it)
        off = lambda rt: bs * (s["bpw"] - self.sub[rt]["bpw"]) + (self.doff(t, rt) if bs else 0.0)   # decode term with the size term
        if r.get("stopped"):
            rt = s["ref"] if s.get("ref") in self.res else cur
            P = self.paired(self.res[rt], r, parts, off=off(rt), strata=st)
            d = "reject"; ndocs = P.k
        else:
            if s.get("audit") and s.get("ref") in self.res:                       # the rule after the fact
                Pe = self.paired(self.res[s["ref"]], r, parts, rule=self.rule, off=off(s["ref"]), strata=st)
                Pf = self.paired(self.res[s["ref"]], r, parts, off=off(s["ref"]), strata=st)
                if Pe.stopped:
                    self.rec({"ev": "bias", "tag": t, "k": Pe.k, "d_prefix": Pe.d(), "d_full": Pf.d(), "se_full": Pf.se(), "off": Pe.off})
                    self.bias.append({"d_prefix": Pe.d(), "d_full": Pf.d()})
                    self.decide(t, size, "reject", Pe, r["slice"], Pe.k); return
            if self.K > 1:                                                  # RULES-pool §3: the join rule (the reference is the worst member)
                P, d, info = self.pool_judge(t, s, r, parts, st, off); self.decide(t, size, d, P, r["slice"], P.k, pool=info); return
            P = self.paired(ref, r, parts, off=off(cur), strata=st); d = P.final(); ndocs = P.k
        self.decide(t, size, d, P, r["slice"], ndocs)

    def decide(self, t, size, d, P, sl, ndocs, pool=None):
        s = self.sub[t]
        if d == "accept" and self.cf and size in self.cf_sizes: return self.tentative(t, size, P, sl, ndocs, pool)   # (#189)
        self.reg.decided(sl, str(size), ndocs); self.dec[size] += 1
        self.kind_stats[s["kind"]][1] += 1; self.kind_stats[s["kind"]][0] += d == "accept"
        self.rec({"ev": "decision", "tag": t, "size": size, "decision": d, "kind": s["kind"], "k": P.k, "z": P.z(), "logr": P.d(), "logr_pooled": P.d_pooled(),
                  "off": P.off, "se": P.se(), "parts": P.summary()["parts"],
                  "stat_b": P.w is not None, **({"why": P.why} if P.why else {})})
        if d == "accept" and pool is not None: self.pool_join(t, size, pool)
        if d == "accept" and (pool is None or pool["inc"]):
            old = self.inc.get(size); self.inc[size] = t
            self.rec({"ev": "incumbent", "size": size, "tag": t, "prev": old})
            self.log(f"ACCEPT {t} ({s['kind']}) for {size}: log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}) vs {old}; {s['bpw']:.4f} bpw")
            mc = self.c.get("monitor")
            if mc and mc.get("when", "accept") == "accept":                 # the band's new best -> the held-out monitor
                self.submit(self.newtag(size, "mon"), self.assign(t), size, "monitor", None, early=False, extra={"of": t},
                            slice_entry={"name": mc["name"], "parts": {"m": mc["golden"]}})
        elif d == "accept":
            self.log(f"JOIN {t} ({s['kind']}) for {size}: {pool['join']}{' ' + pool['replaced'] if pool.get('replaced') else ''} (pool {len(self.members(size))}/{self.K}): log r {P.d():+.4f}, size-adjusted {P.da():+.4f}, vs {pool['vs']}")
        else:
            self.log(f"reject {t} ({s['kind']}) for {size}: log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}) after {P.k} docs" + (" [stopped]" if self.res[t].get("stopped") else "") +
                     (" [guard: per-document mean " + ", ".join(f"{p} {P.docmean(p) + P.off:+.4f}" for p in P.part if P.docmean(p) is not None) + "]" if P.why == "guard" else ""))

    # ------------------------------------------------------------------------------------------------ two-stage accept (#189, RULES-confirm)
    def check_confirm_registry(self):
        """CONF confirm: every pair's slices exist, each confirmation slice has "role": "confirm" and was never a search slice"""
        S = self.reg.d["slices"]; bad = []
        for a, b in self.cf["pairs"]:
            if a not in S or b not in S: bad.append(f"pair ({a}, {b}): slice not in the registry"); continue
            if S[a].get("role") == "confirm": bad.append(f"{a}: a search slice with role confirm")
            if S[b].get("role") != "confirm": bad.append(f"{b}: no \"role\": \"confirm\" (slices.py role REG {b} confirm, with the driver stopped)")
            if S[b]["status"] == "active" or sum(S[b].get("decisions", {}).values()):
                bad.append(f"{b}: was a search slice (status {S[b]['status']}, decisions {S[b].get('decisions')})")
        if bad: raise SystemExit("CONF confirm: registry not ready:\n  " + "\n  ".join(bad))

    def cslice(self, sl):
        """the confirmation slice paired with search slice sl; None (logged once): its tentatives wait for a pair in CONF"""
        cs = self.cf_pairs.get(sl)
        if cs is None and sl not in self.cf_warned:
            self.cf_warned.add(sl); self.log(f"WARN no confirmation slice paired with {sl}: its tentatives wait (add a pair to CONF confirm.pairs)")
        return cs

    def tentative(self, t, size, P, sl, ndocs, pool):
        """RULES-confirm §1: a stage-1 accept / join is recorded and counted as the search slice's decision; nothing else changes"""
        s = self.sub[t]
        self.reg.decided(sl, str(size), ndocs); self.dec[size] += 1; self.kind_stats[s["kind"]][1] += 1
        target = "inc" if pool is None or pool["inc"] else "worst"
        e = {"ev": "tentative", "tag": t, "size": size, "kind": s["kind"], "slice": sl, "cslice": self.cf_pairs.get(sl), "target": target,
             "vs": self.inc.get(size) if target == "inc" else pool["vs"], "k": P.k, "z": P.z(), "logr": P.d(), "logr_pooled": P.d_pooled(),
             "off": P.off, "se": P.se(), "parts": P.summary()["parts"], **({"pool": pool} if pool is not None else {})}
        self.rec(e); self.tent[t] = e
        self.log(f"TENTATIVE {t} ({s['kind']}) for {size}: stage 1 log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}) vs {e['vs']}"
                 + (f" (pool {pool['join']}{', would be incumbent' if pool['inc'] else ''})" if pool else "") + f"; confirmation on {e['cslice']}")
        self.cslice(sl)

    def cf_eval(self, mt, cs, size, why, owner):
        """the whole-slice confirmation eval of model `mt` (a tag) on cs, cached per (assignment hash, slice) -> ("ok", eval tag) |
        ("wait", eval tag) | ("fail", None) once CONF tries (2) evals failed; submits one (kind "confirm") when none is usable"""
        h = self.sub[mt]["hash"]; tags = self.cfx.get((h, cs), [])
        for x in reversed(tags):
            if x not in self.res: return "wait", x
            r = self.res[x]
            if not r.get("fail") and r.get("docs") and all(v is not None for v in r["docs"]): return "ok", x
        if len(tags) >= self.cf_tries: return "fail", None
        x = self.newtag(size, "confirm")
        self.submit(x, self.assign(mt), size, "confirm", cs, early=False, extra={"of": mt, "for": owner})
        self.rec({"ev": "confirm_submit", "tag": owner, "eval": x, "of": mt, "slice": cs, "why": why, "try": len(tags) + 1})
        return "wait", x

    def cf_stat(self, ref_t, ref_x, t, x, cs, bs):
        """stage 2 = the stage-1 statistic (paired, strata, CONF guard) on the confirmation slice: eval x (model t) vs eval ref_x
        (model ref_t); size offset bs (bpw_t - bpw_ref) plus the decode term, as on_result"""
        off = bs * (self.sub[t]["bpw"] - self.sub[ref_t]["bpw"]) + (self.doff(t, ref_t) if bs else 0.0)
        return self.paired(self.res[ref_x], self.res[x], self.reg.parts_of(cs), off=off, strata=self.reg.strata_of(cs))

    def confirm_step(self, act):
        """every pending tentative, in tentative order: submit what its confirmation needs; decide it once everything is scored"""
        for t in list(self.tent):
            if t in self.tent: self.confirm_try(t, act)

    def confirm_try(self, t, act):
        e = self.tent[t]; size = e["size"]; cs = e.get("cslice") or self.cslice(e["slice"])
        if cs is None: return
        sc, xc = self.cf_eval(t, cs, size, "candidate", t)
        if self.golive_wait(size): return                                  # rule 7 first
        inc = self.inc.get(size)
        if inc is None: return
        if self.K > 1 and not (self.scored_on(size, act) and not self.reseed_wait(size, act)): return   # the worst needs the pool on act
        target = e["target"]; ref = inc
        if self.K > 1 and target == "worst":                                # the CURRENT worst member (rule 3); the incumbent -> target inc
            ref = self.worst(size)
            if ref == inc: target = "inc"
        why = "reference" if self.sub[ref]["hash"] == self.sub[e["vs"]]["hash"] else "reference changed"
        sr, xr = self.cf_eval(ref, cs, size, why, t)
        if "fail" in (sc, sr): return self.confirm_verdict(t, e, "reject", None, cs, xc, ref, xr, target, why="eval failed")
        if "wait" in (sc, sr): return
        P = self.cf_stat(ref, xr, t, xc, cs, self.sub[t].get("b", 0.0) or 0.0)
        ok = P.z() < self.cf_z
        self.confirm_verdict(t, e, ("accept" if target == "inc" else "join") if ok else "reject", P, cs, xc, ref, xr, target)

    def confirm_verdict(self, t, e, verdict, P, cs, xc, ref, xr, target, why=None):
        """RULES-confirm §2: the stage-2 verdict ("confirm" + stage-2 "decision" events), applied against the current state"""
        size = e["size"]; s = self.sub[t]
        st = ({"k": P.k, "z": P.z(), "logr": P.d(), "logr_pooled": P.d_pooled(), "off": P.off, "se": P.se(), "parts": P.summary()["parts"],
               "stat_b": P.w is not None} if P is not None else {"logr": None})
        info = None
        if verdict != "reject" and self.K > 1:
            full = len(self.members(size)) >= self.K
            info = {"join": "replace" if full else "add", "inc": verdict == "accept", "vs": ref, **({"replaced": self.worst(size)} if full else {})}
        self.reg.confirmed(cs, str(size), P.k if P is not None else 0)
        changed = self.sub[ref]["hash"] != self.sub[e["vs"]]["hash"]
        self.rec({"ev": "confirm", "tag": t, "size": size, "verdict": verdict, **({"why": why} if why else {}), "slice": cs, "eval": xc, "ref": ref,
                  "ref_eval": xr, "target": target, "ref_changed": changed, "z_thr": self.cf_z, **st,
                  **({"da": P.da(), "final": P.final()} if P is not None else {}), "logr1": e["logr"], "z1": e["z"], "slice1": e["slice"],
                  **({"pool": info} if info else {})})
        self.rec({"ev": "decision", "stage": 2, "tag": t, "size": size, "decision": verdict, "kind": s["kind"], **st, "logr1": e["logr"], "z1": e["z"],
                  "slice": cs, "slice1": e["slice"], "vs": ref, **({"why": "confirm" + (f" ({why})" if why else "")} if verdict == "reject" else {})})
        self.tent.pop(t, None); self.kind_stats[s["kind"]][0] += verdict in ("accept", "join")
        nums = (f"stage 2 on {cs}: log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}); " if P is not None else f"stage 2 on {cs}: {why}; ") + \
               f"stage 1 on {e['slice']}: log r {e['logr']:+.4f} (z {e['z']:+.2f}); vs {ref}{' (changed)' if changed else ''}"
        if info is not None: self.pool_join(t, size, info)
        if verdict == "accept":
            old = self.set_incumbent(size, t); self.monitor_on_change(size, t)
            self.log(f"ACCEPT (confirmed) {t} ({s['kind']}) for {size}: {nums}; was {old}; {s['bpw']:.4f} bpw")
        elif verdict == "join":
            self.log(f"JOIN (confirmed) {t} ({s['kind']}) for {size}: {info['join']}{' ' + info['replaced'] if info.get('replaced') else ''} (pool {len(self.members(size))}/{self.K}): {nums}")
        else:
            self.log(f"reject (confirm) {t} ({s['kind']}) for {size}: {nums}")

    def golive_wait(self, size):
        """rule 7: a size's tentatives wait until its go-live check is resolved"""
        if self.golive_end or self.golive is None: return False
        return any(p["size"] == size for p in self.golive["plan"]) and size not in self.golive_res

    def golive_step(self):
        """RULES-confirm §7 (once; CONF golive_from_line): each listed size's incumbent vs its incumbent at that state line, on the first
        pair's confirmation slice, under rule 2. Not confirmed -> back to the switch-time model, the pool to that model alone"""
        n = self.cf.get("golive_from_line")
        if n is None or self.golive_end: return
        cs = self.cf["pairs"][0][1]
        if self.golive is None:
            sw = {}; i = 0
            with open(f"{self.R}/state.jsonl") as f:
                for i, l in enumerate(f, 1):
                    if i > n: break
                    try: x = json.loads(l)
                    except ValueError: continue
                    if x.get("ev") == "incumbent": sw[x["size"]] = x["tag"]
            if i < n: raise SystemExit(f"CONF confirm.golive_from_line {n}: the state has only {i} lines")
            plan, same = [], []
            for size in sorted(self.cf_sizes):
                cur, old = self.inc.get(size), sw.get(size)
                if cur is None or old is None or size in self.stopped_sizes or not self.feasible(size): continue
                if self.sub[cur]["hash"] == self.sub[old]["hash"]: same.append({"size": size, "cur": cur, "sw": old}); continue
                plan.append({"size": size, "cur": cur, "sw": old, "b": self.b_of(size)})
            self.golive = {"ev": "golive", "phase": "start", "from_line": n, "slice": cs, "plan": plan, "same": same}
            self.rec(self.golive)
            names = " ".join(f"{x['size']:g}" for x in plan)
            self.log(f"GO-LIVE check on {cs} (incumbents at state line {n}): {len(plan)} sizes to confirm ({names}), {len(same)} unchanged")
            for x in same:
                g = {"ev": "golive_size", "size": x["size"], "verdict": "same", "cur": x["cur"], "sw": x["sw"]}
                self.rec(g); self.golive_res[x["size"]] = g
        for p in self.golive["plan"]:
            size = p["size"]
            if size in self.golive_res: continue
            sc, xc = self.cf_eval(p["cur"], cs, size, "golive", "golive")
            ss, xs = self.cf_eval(p["sw"], cs, size, "golive", "golive")
            if "fail" in (sc, ss):
                g = {"ev": "golive_size", "size": size, "verdict": "error", "cur": p["cur"], "sw": p["sw"], "why": "eval failed: kept"}
                self.rec(g); self.golive_res[size] = g; self.log(f"GO-LIVE {size:g}: confirmation eval failed {self.cf_tries}x: incumbent kept (check by hand)"); continue
            if "wait" in (sc, ss): continue
            P = self.cf_stat(p["sw"], xs, p["cur"], xc, cs, p["b"])
            verdict = "kept" if P.z() < self.cf_z else "reverted"
            self.reg.confirmed(cs, str(size), P.k)
            g = {"ev": "golive_size", "size": size, "verdict": verdict, "cur": p["cur"], "sw": p["sw"], "eval": xc, "ref_eval": xs, "slice": cs,
                 "k": P.k, "z": P.z(), "logr": P.d(), "da": P.da(), "off": P.off, "se": P.se(), "parts": P.summary()["parts"], "inc_now": self.inc.get(size)}
            self.rec(g); self.golive_res[size] = g
            self.log(f"GO-LIVE {size:g}: {p['cur']} vs switch-time {p['sw']} on {cs}: log r {P.d():+.4f}, size-adjusted {P.da():+.4f} (z {P.z():+.2f}) -> {verdict}")
            if verdict == "reverted":
                self.set_incumbent(size, p["sw"])
                if self.K > 1: self.pool[size] = [p["sw"]]; self.rec({"ev": "pool", "size": size, "members": [p["sw"]], "why": "golive"})
                self.monitor_on_change(size, p["sw"])
        if all(p["size"] in self.golive_res for p in self.golive["plan"]):
            self.golive_end = True
            self.rec({"ev": "golive", "phase": "done", "verdicts": {f"{z:g}": g["verdict"] for z, g in sorted(self.golive_res.items())}})
            self.log("GO-LIVE done: " + " ".join(f"{z:g}:{g['verdict']}" for z, g in sorted(self.golive_res.items())))

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
        st = self.reg.strata_of(r["slice"]) if r["slice"] in self.reg.d["slices"] else None
        if st:                                                             # RULES-strata §6: one row per (part, stratum)
            groups = sorted(set(zip(parts, st)))
        else:
            pset = sorted(set(parts)) if len(set(parts)) > 1 else [None]   # one row per part (w131, #171)
            groups = [(p, None) for p in pset]
        ref = s.get("ref") if s.get("ref") in self.res else None
        rr = self.res[ref] if ref and ref != t else None
        for p, sg in groups:
            ix = [i for i in range(kq) if (p is None or parts[i] == p) and (sg is None or st[i] == sg)]
            if len(ix) < 3: continue
            label = f"{r['slice']}|" + (f"{p}|" if p else "") + (f"{sg}|" if sg else "") + ("full" if full else f"pre{kq}")
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
        share0 = self.c.get("kind_share0", MV.SHARE0)
        if self.K > 1 and "cross" not in share0: share0 = dict(share0, cross=self.cross_share0)     # RULES-pool §4
        shares = MV.kind_shares({k: tuple(v) for k, v in self.kind_stats.items()}, share0)
        kcount = collections.Counter(self.sub[t]["kind"] for t in self.sub if self.sub[t]["kind"] in shares)
        tries = 0; failed = set()
        while len(out) + made < depth and tries < 4 * len(self.sizes):
            tries += 1
            ready = [s for s in self.sizes if s not in self.stopped_sizes and self.feasible(s) and self.scored_on(s, act) and not self.reseed_wait(s, act) and s not in failed
                     and self.fits[self.gname[s]] is not None]
            if not ready: break
            w = self.c.get("size_weights", {})
            size = min(ready, key=lambda s: ((self.dec[s] + inflight[s]) / float(w.get(str(s), 1.0)), self.rng.random()))
            par = self.parent(size); a = self.assign(par); lo, hi = self.window(size)
            oth = PR.choose_other(par, self.members(size), self.rng) if self.K > 1 else None; other = self.assign(oth) if oth else None
            kind = MV.pick_kind(shares, kcount, self.rng)
            F = self.fits[self.gname[size]]
            c = self.gen_move(kind, size, F, a, lo, hi, other)
            if c is None:
                for k2 in sorted(shares, key=lambda k: -shares[k]):
                    if k2 == kind: continue
                    c = self.gen_move(k2, size, F, a, lo, hi, other)
                    if c is not None: kind = k2; break
            if c is None:
                failed.add(size)
                if not self.nogen.get(size): self.log(f"no new candidate for {size} (window {lo * 8 / self.M.P:.4f}-{hi * 8 / self.M.P:.4f} bpw)")
                self.nogen[size] = True; continue
            self.nogen[size] = False
            rf = self.reference(size); pr = F["pred"]; dpred = math.log(pr.kl(c) / pr.kl(self.assign(rf)))
            af = (self.c.get("audit_frac_by_size") or {}).get(f"{size:g}", self.c.get("audit_frac", 0.1))   # (#183) per size
            audit = self.rng.random() < float(af)
            tag = self.newtag(size, kind)
            self.submit(tag, c, size, kind, act, early=not audit, ref_tag=rf, extra={"pred_logr": dpred, "audit": audit, **({"parent": par, "other": oth} if self.K > 1 else {})})
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
            nxt = self.reg.ready()                                        # (#189) never a confirmation slice
            if nxt:
                new = self.reg.activate_next(); self.rec({"ev": "rotate", "to": new, "from": act})
                self.log(f"ROTATE {act} -> {new}"); self.set_act(new); self.monitor_at_rotation(); return new
            if not getattr(self, "_warned", False): self.log(f"rotation due on {act} but no ready slice: build one (slices.py)"); self._warned = True
        return act

    def monitor_at_rotation(self):
        """CONF monitor "when": "rotate" (#183, user 11-14: the fixed tracking slice is the per-accept curve; the verify
        monitor is a slower long-context check): at each rotation every size's incumbent without a monitor read yet is
        scored on the monitor once"""
        mc = self.c.get("monitor")
        if not mc or mc.get("when", "accept") != "rotate": return
        read = {m.get("of") for v in self.mon.values() for m in v} | {e.get("of") for e in self.sub.values() if e.get("kind") == "monitor"}
        for size in self.sizes:
            t = self.inc.get(size)
            if t is None or t in read or size in self.stopped_sizes: continue
            self.submit(self.newtag(size, "mon"), self.assign(t), size, "monitor", None, early=False, extra={"of": t},
                        slice_entry={"name": mc["name"], "parts": {"m": mc["golden"]}})

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
        st = self.reg.strata_of(r["slice"]) if r["slice"] in self.reg.d["slices"] else None
        if st:                                              # RULES-strata §6: the strata of a part weighted by w_s (seqrule §4)
            lv = []
            for p in sorted(set(parts)):
                tm = []
                for sg in sorted({st[i] for i, q in enumerate(parts) if q == p}):
                    ix = [i for i, q in enumerate(parts) if q == p and st[i] == sg]
                    kp = sum(r["docs"][i] for i in ix) / max(sum(r["pos"][i] for i in ix), 1)
                    tm.append((SR.STRATUM_W[sg], math.log(kp) - soff.get(f"{r['slice']}|{p}|{sg}|full", 0.0)))
                ln = sum(w * x for w, x in tm) / sum(w for w, _ in tm)
                has = any(f"{r['slice']}|{p}|{sg}|full" in soff for sg in {st[i] for i, q in enumerate(parts) if q == p})
                lv.append(ln / lam if p == "x" and has else ln)
            return math.exp(sum(lv) / len(lv))
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
        if self.K > 1: line += " | pool " + " ".join(f"{z:g}:{len(self.members(z))}" for z in self.sizes if self.inc.get(z) is not None)
        if self.cf:
            pend = collections.Counter(e["size"] for e in self.tent.values())
            line += f" | confirm pending {len(self.tent)}" + (" (" + " ".join(f"{z:g}:{n}" for z, n in sorted(pend.items())) + ")" if pend else "")
            if self.golive is not None and not self.golive_end: line += f" | go-live {len(self.golive_res)}/{len(self.golive['plan']) + len(self.golive.get('same', []))}"
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
        if self.K > 1: self.ensure_pool(act); self.reseed_step(act); self.reselect(act)
        else: self.ensure_incumbents(act)
        if self.cf: self.golive_step(); self.confirm_step(act)
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
