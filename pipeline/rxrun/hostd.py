# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxrun.hostd HOSTCONF [--unit DIR] -- one host's side of the multi-host runner (w120, 11-12; rxrun.multi).

Daemon mode (what rxrun.multi starts on each host through ssh): line JSON on stdin / stdout.
  out  {"hello": NAME, "pid", "settings_stamp", "tree_stamp", "kinds", "slots", "limits", "levels_done": [..],
        "units": {ID: "running" | "finished"}}            once, at start
  in   {"id": ID, "kind": "level", "levels": [NAME, ..]}  build these ladder levels here (one rxrun.run, its devices)
       {"id": ID, "kind": "eval", "tag": T, "mix": {...}, "slices": [...]}   pack + mix + score here (below)
       {"ack": ID}                                       the coordinator has recorded ID's result (dir kept, marked)
       {"quit": true}
  out  {"id": ID, "event": "started"} / {"id": ID, "rc": 0 | 1, "result": {...}}   (results of units finished while no
        coordinator was connected are sent again after every hello until acked)
Every unit runs as its own detached process (`--unit DIR`, setsid): a dropped ssh (the box2 forward, a VM restart)
does not stop it; the next hostd sees DIR/result.json. A unit dir: ROOT/units/<ID>/{req.json, pid, unit.log,
result.json, acked}.

level unit: rxrun.run on the host's ladder conf (HOSTCONF "run_conf") restricted to the unit's levels, swj_batch = the
  unit's level count (one swj pass for all of them). Restartable as rxrun.run is. Result: per level DONE, jobs, wall.
eval unit (pack + score): {"tag", "mix": {"base": ALIAS | PATH, "levels": {weight: LEVEL, ...} | "trunk": LEVEL |
  "parity": {"even": LEVEL, "odd": LEVEL}, "donors": {weight: PATH, ...} (head / embedding / V2 donors)}, "slices":
  [{"name": S, "parts": {"x": GOLDEN, "c": GOLDEN}}]}. Steps: every level's pack (HOSTCONF eval.pack_cmd on the level's
  final.recipe, conv.lock + ram.lock exclusive; packs kept in eval.packs and reused) -> the mix (eval.mix_cmd, radmix
  over the base container) -> each part scored (eval.score_cmd, the eval card's lock + ram.lock shared). Result: the
  search's results.jsonl fields (tag, bytes, bpw, positions, kld, p99, top1, slice) per slice pooled over its parts, the
  parts, and per-document KL sums (eval/docstat/readout.py's inputs, small) -- rows files stay on the host.
Command templates: lists of strings with {recipe} {out} {reuse} {base} {mixfile} {mix} {golden} {card} {model}.
Limits (HOSTCONF "limits"): kinds, k_min / k_max (levels outside are not taken), max_unit_est_h; the coordinator
checks them before it hands out a unit, hostd checks again."""
import argparse, fcntl, glob, hashlib, json, os, signal, subprocess, sys, threading, time

PARAMS = 26895998464
TREE_DIRS = ["pipeline/rxq", "pipeline/rxrun", "pipeline/config", "store", "mix", "v3/e3/exl3frac", "v3/e3/exl3frac2", "v3/e3/exl3cb"]
SRC_EXT = (".py", ".cu", ".cuh", ".cpp", ".h", ".hip", ".toml")


def _hipify_output(name):
    """= rxq.engine._hipify_output (not imported: rxq.engine needs torch)"""
    return name.endswith(".hip") or "_hip." in name or name.startswith("hip_")


def tree_stamp(tree, dirs=None, exl3_src=None):
    """sha256 (16 hex) of the staged tree's sources (relative path + bytes) and the fork's kernel sources: two hosts
    with the same stamp run the same code (the rxq code stamp inside every tensor file proves it again later)"""
    h = hashlib.sha256(); files = []
    for d in dirs or TREE_DIRS:
        base = os.path.join(tree, d)
        for f in sorted(glob.glob(f"{base}/**/*", recursive=True)):
            if os.path.isfile(f) and f.endswith(SRC_EXT) and "__pycache__" not in f and not _hipify_output(os.path.basename(f)):
                files.append((os.path.relpath(f, tree), f))
    if exl3_src:
        for f in sorted(glob.glob(f"{exl3_src}/**/*", recursive=True)):
            if os.path.isfile(f) and f.endswith(SRC_EXT) and not _hipify_output(os.path.basename(f)):
                files.append(("exl3/" + os.path.relpath(f, exl3_src), f))
    for rel, f in files:
        h.update(rel.encode()); h.update(b"\0"); h.update(open(f, "rb").read()); h.update(b"\0")
    return h.hexdigest()[:16], len(files)


def load(p):
    if p.endswith(".toml"):
        import tomllib
        return tomllib.load(open(p, "rb"))
    return json.load(open(p))


class Host:
    def __init__(self, hc_path):
        self.hc_path = os.path.abspath(hc_path); self.hc = load(hc_path)
        self.name = self.hc["name"]; self.root = self.hc["root"]; self.U = f"{self.root}/units"
        os.makedirs(self.U, exist_ok=True)
        self.rc = load(self.hc["run_conf"]) if self.hc.get("run_conf") else None

    # ---------------------------------------------------------------------------------------------- identity
    def stamps(self):
        from rxq import config as C
        st = C.settings_stamp(C.load(self.rc["settings"])) if self.rc else None
        env = (self.rc or {}).get("env") or {}
        ts, n = tree_stamp(self.hc["tree"], self.hc.get("tree_dirs"), env.get("RXQ_EXL3_SRC") or os.environ.get("RXQ_EXL3_SRC"))
        return st, ts, n

    def levels_done(self):
        if not self.rc: return []
        r = self.rc["root"]
        return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{r}/*/DONE"))

    def units(self):
        out = {}
        for d in sorted(glob.glob(f"{self.U}/*/req.json")):
            u = os.path.dirname(d); i = os.path.basename(u)
            if os.path.exists(f"{u}/acked"): continue
            if os.path.exists(f"{u}/result.json"): out[i] = "finished"
            elif alive(u): out[i] = "running"
            else: out[i] = "dead"
        return out

    # ------------------------------------------------------------------------------------------------ daemon
    def serve(self):
        proto = os.fdopen(os.dup(1), "w", buffering=1); os.dup2(2, 1)
        wl = threading.Lock()

        def send(o):
            with wl: proto.write(json.dumps(o) + "\n"); proto.flush()
        st, ts, nf = self.stamps()
        send({"hello": self.name, "pid": os.getpid(), "settings_stamp": st, "tree_stamp": ts, "tree_files": nf,
              "kinds": self.hc.get("limits", {}).get("kinds", ["level", "eval"]), "slots": self.hc.get("slots", {"level": 1, "eval": 1}),
              "limits": self.hc.get("limits", {}), "levels_done": self.levels_done(), "units": self.units()})
        stop = threading.Event(); sent = {}

        def watch():                                   # results of finished units (also ones started before this hostd)
            while not stop.is_set():
                for i, s in self.units().items():
                    if s == "running" or sent.get(i) == s: continue
                    if s == "finished":
                        try: r = json.load(open(f"{self.U}/{i}/result.json"))
                        except ValueError: continue
                        send({"id": i, "rc": r.get("rc", 1), "result": r})
                    else: send({"id": i, "rc": 1, "result": {"err": "unit process gone without a result (killed?)"}})
                    sent[i] = s
                stop.wait(2.0)
        th = threading.Thread(target=watch, daemon=True); th.start()
        for line in sys.stdin:
            if not line.strip(): continue
            req = json.loads(line)
            if req.get("quit"): break
            if "ack" in req:
                u = f"{self.U}/{req['ack']}"
                if os.path.isdir(u): open(f"{u}/acked", "w").close()
                continue
            try:
                self.start(req); send({"id": req["id"], "event": "started"})
            except Exception as e:                    # noqa: BLE001
                send({"id": req.get("id"), "rc": 1, "result": {"err": f"refused: {e}"}})
        stop.set()
        return 0

    def check_limits(self, req):
        lim = self.hc.get("limits", {})
        if req["kind"] not in lim.get("kinds", ["level", "eval"]): raise ValueError(f"kind {req['kind']} not taken here")
        if req["kind"] == "level":
            from rxq import config as C, ladder
            cfg = C.load(self.rc["settings"]); by = {v["name"]: v for v in ladder.levels(cfg)}
            for n in req["levels"]:
                if n not in by: raise ValueError(f"level {n} is not in this host's settings ladder (stamp differs?)")
                K = by[n]["K"]
                if K < lim.get("k_min", -1) or K > lim.get("k_max", 99): raise ValueError(f"level {n}: K{K} outside k_min / k_max")

    def start(self, req):
        i = req["id"]; u = f"{self.U}/{i}"
        if os.path.exists(f"{u}/req.json"):
            old = json.load(open(f"{u}/req.json"))
            if req.get("attempt", 0) > old.get("attempt", 0) and not alive(u):           # a retry: a fresh unit dir
                arch = f"{u}.a{old.get('attempt', 0)}-{int(time.time())}"; os.replace(u, arch); open(f"{arch}/acked", "w").close()
            elif alive(u) or os.path.exists(f"{u}/result.json"): return                # already here (a resent request)
        if not os.path.exists(f"{u}/req.json"): self.check_limits(req)
        os.makedirs(u, exist_ok=True)
        if not os.path.exists(f"{u}/req.json"):
            json.dump(req, open(f"{u}/req.json.part", "w")); os.replace(f"{u}/req.json.part", f"{u}/req.json")
        lf = open(f"{u}/unit.log", "a")
        p = subprocess.Popen([sys.executable, "-m", "rxrun.hostd", self.hc_path, "--unit", u], stdin=subprocess.DEVNULL,
                             stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ))
        lf.close(); open(f"{u}/pid", "w").write(str(p.pid))
        threading.Thread(target=p.wait, daemon=True).start()          # reap


def alive(u):
    try: pid = int(open(f"{u}/pid").read())
    except (OSError, ValueError): return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    try:                                              # a zombie or a reused pid is not our unit
        return "rxrun.hostd" in open(f"/proc/{pid}/cmdline").read().replace("\0", " ") and \
            open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except OSError: return False


# --------------------------------------------------------------------------------------------------- units
class Lock:
    def __init__(self, path, excl):
        self.path, self.excl, self.fd = path, excl, None

    def __enter__(self):
        if self.path:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX if self.excl else fcntl.LOCK_SH)
        return self

    def __exit__(self, *e):
        if self.fd is not None: fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd); self.fd = None


def sh(cmd, log, env=None, nice=10):
    with open(log, "a") as lf:
        lf.write(f"$ {' '.join(cmd)}\n"); lf.flush()
        return subprocess.run(["nice", "-n", str(nice)] + cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                              env=env or dict(os.environ)).returncode


def fill(tmpl, **kw):
    return [str(x).format(**kw) for x in tmpl]


def run_level(H, u, req):
    conf = dict(H.rc); conf["levels"] = list(req["levels"]); conf["swj_batch"] = max(1, len(req["levels"]))
    json.dump(conf, open(f"{u}/conf.json", "w"), indent=1)
    t = time.time()
    env = dict(os.environ); env.update({k: str(v) for k, v in (conf.get("env") or {}).items()})
    rc = sh([conf.get("python", sys.executable), "-m", "rxrun.run", f"{u}/conf.json"], f"{u}/rxrun.out", env=env, nice=0)
    res = {"rc": 0 if rc == 0 else 1, "kind": "level", "levels": {}, "wall": round(time.time() - t, 1), "host": H.name}
    for n in req["levels"]:
        d = f"{conf['root']}/{n}"; jl = f"{d}/logs/jobs.jsonl"
        recs = [json.loads(x) for x in open(jl)] if os.path.exists(jl) else []
        res["levels"][n] = {"done": os.path.exists(f"{d}/DONE"), "jobs": len(recs), "job_wall": round(sum(r.get("wall", 0) for r in recs), 1)}
    if not all(v["done"] for v in res["levels"].values()):
        res["rc"] = 1; res["err"] = f"rxrun.run rc {rc}, levels not DONE: {[n for n, v in res['levels'].items() if not v['done']]}"
    return res


def doc_sums(golden, rows):
    import numpy as np
    m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(rows, np.float32).reshape(-1, 4)
    out, allk = [], []
    for d in m["docs"]:
        s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[s]
        k = K[s, 0][g].astype(np.float64); out.append([round(float(k.sum()), 6), int(g.sum())]); allk.append(k)
    return out, np.concatenate(allk) if allk else np.zeros(0)


def run_eval(H, u, req):
    E = H.hc["eval"]; R = H.rc["root"]; t0 = time.time()
    P = E.get("packs", f"{H.root}/packs"); MX = E.get("mixes", f"{H.root}/mixes"); SC = E.get("scores", f"{H.root}/scores")
    for d in (P, MX, SC): os.makedirs(d, exist_ok=True)
    mx = req["mix"]; tag = req["tag"]; log = f"{u}/eval.log"
    alias = E.get("aliases", {})
    rp = lambda x: alias.get(x, x)
    # the trunk assignment: weight -> level
    from rxq import mix as M
    trunk = M.trunk_names()
    if "levels" in mx: lv = dict(mx["levels"])
    elif "trunk" in mx: lv = {w: mx["trunk"] for w in trunk}
    else: lv = {w: mx["parity"]["even" if int(w.split(".")[1]) % 2 == 0 else "odd"] for w in trunk}
    need = sorted(set(lv.values()))
    miss = [n for n in need if not os.path.exists(f"{R}/{n}/DONE")]
    if miss: return {"rc": 1, "err": f"levels not built on this host: {miss}"}
    for n in need:                                                # packs (kept, reused by later evals)
        pk = f"{P}/{n}.rad"
        if os.path.exists(pk): continue
        with Lock(E.get("conv_lock"), True), Lock(E.get("ram_lock"), True):
            if not os.path.exists(pk):
                if os.path.exists(pk + ".part"): os.remove(pk + ".part")
                rc = sh(fill(E["pack_cmd"], recipe=f"{R}/{n}/final.recipe", out=pk + ".part", reuse="", model=E.get("model", "")), log)
                if rc or not os.path.exists(pk + ".part"): return {"rc": 1, "err": f"pack {n} rc {rc} ({log})"}
                os.replace(pk + ".part", pk)
    mf = f"{MX}/{tag}.mix"; mrad = f"{MX}/{tag}.rad"
    lines = [f"{w} {P}/{lv[w]}.rad" for w in trunk] + [f"{w} {rp(p)}" for w, p in (mx.get("donors") or {}).items()]
    open(mf, "w").write("\n".join(lines) + "\n")
    if not os.path.exists(mrad):
        with Lock(E.get("conv_lock"), True), Lock(E.get("ram_lock"), True):
            rc = sh(fill(E["mix_cmd"], out=mrad + ".part", base=rp(mx["base"]), mixfile=mf), log)
            if rc or not os.path.exists(mrad + ".part"): return {"rc": 1, "err": f"mix {tag} rc {rc} ({log})"}
            os.replace(mrad + ".part", mrad)
    import numpy as np
    nbytes = os.path.getsize(mrad); res = {"rc": 0, "kind": "eval", "tag": tag, "host": H.name, "bytes": nbytes,
                                          "bpw": nbytes * 8 / PARAMS, "slices": {}}
    for sl in req["slices"]:
        parts = {}; ks = []
        for pn, g in sl["parts"].items():
            g = rp(g); o = f"{SC}/{tag}-{sl['name']}-{pn}"
            if not os.path.exists(o + ".json.rows"):
                with Lock(E.get("card_lock"), True), Lock(E.get("ram_lock"), False):
                    rc = sh(fill(E["score_cmd"], mix=mrad, golden=g, out=o, card=E.get("card", 0)), log, nice=0)
                if rc or not os.path.exists(o + ".json.rows"): return {"rc": 1, "err": f"score {tag} {sl['name']}-{pn} rc {rc} ({log})"}
            j = json.load(open(o + ".json"))["all"]; docs, k = doc_sums(g, o + ".json.rows")
            parts[pn] = {"positions": int(len(k)), "kld": float(k.mean()) if len(k) else None, "p99": float(np.percentile(k, 99)) if len(k) else None,
                         "top1": j.get("top1_agreement"), "positions_all": j.get("positions"), "docs": docs}
            ks.append(k)
        k = np.concatenate(ks); n_all = sum(p["positions_all"] or 0 for p in parts.values())
        top1 = sum((p["top1"] or 0) * (p["positions_all"] or 0) for p in parts.values()) / n_all if n_all else None
        res["slices"][sl["name"]] = {"tag": tag, "slice": sl["name"], "bytes": nbytes, "positions": int(len(k)), "kld": float(k.mean()),
                                     "p99": float(np.percentile(k, 99)), "top1": top1, "parts": parts}
    res["wall"] = round(time.time() - t0, 1)
    return res


def unit_main(hc, u):
    H = Host(hc); req = json.load(open(f"{u}/req.json"))
    try:
        res = run_level(H, u, req) if req["kind"] == "level" else run_eval(H, u, req) if req["kind"] == "eval" else {"rc": 1, "err": f"kind {req['kind']}"}
    except Exception as e:                            # noqa: BLE001
        import traceback; traceback.print_exc(); res = {"rc": 1, "err": repr(e)[:400]}
    res.setdefault("host", H.name); res["id"] = req["id"]; res["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(f"{u}/result.json.part", "w")); os.replace(f"{u}/result.json.part", f"{u}/result.json")
    return res["rc"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxrun.hostd")
    ap.add_argument("hostconf"); ap.add_argument("--unit"); ap.add_argument("--stamp", action="store_true")
    a = ap.parse_args(argv)
    if a.unit: return unit_main(a.hostconf, a.unit)
    H = Host(a.hostconf)
    if a.stamp: print(json.dumps(dict(zip(("settings_stamp", "tree_stamp", "tree_files"), H.stamps())))); return 0
    signal.signal(signal.SIGHUP, signal.SIG_IGN)       # a dropped ssh ends stdin; the units go on anyway
    return H.serve()


if __name__ == "__main__":
    sys.exit(main())
