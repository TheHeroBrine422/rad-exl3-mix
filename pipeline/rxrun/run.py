# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxrun.run CONF [--status] [--plan] [--estimate [--rate R] [--swj-h H]] -- the local ladder driver (w105).

Builds every level of the settings file's ladder (rxq.ladder.levels: pass 1 -> 2 -> 3, the order inside a pass as
listed; conditional levels only if CONF names their pass) on our own cards, restartable at any point:
  per level ROOT/<level>/:
    specs/      the jobs' rxq specs (rxq.ladder.level_jobs, the settings file's split; = mkjobs92 / rqah's stage())
    out/<job>/  the job dirs (rxrun.pool: N at a time, persistent workers; finished jobs skipped)
    dup/<job>/  duplicate-seed jobs (spec.json + outputs): dup_frac of the level's tensors (min 1), one tensor a job,
                seed_offset + dup_seed_delta, the families in rotation across levels, layers random (seeded by the
                settings stamp and the level, so a rerun picks the same); never assembled; rxrun.seedsd reads them
    level/      rxq assemble of out/* (+ level/c9 from the writers' in-job C9)            [CPU lane]
    final/      rxq.post apply level SWJ,level/c9 (or a copy when no post step applies) [CPU lane]
    final.recipe  rxq.mix recipe final (the pack's rad-convert recipe; packs are per mix, not here)
    DONE        written last: a level with DONE is skipped
  ROOT/swj-<scales stamp[:8]>/<level>/ the swj level pass (rxq.swjfit --fused --no-cache, swj_batch levels in one staged pass) on a device
                    with kind "swj" (the card lock + ram.lock shared for the pass)
Every step writes .part then renames (assemble, post, recipe), or has its own .done (swjfit), so a killed run leaves
nothing a rerun would trust. The CPU steps run in a background lane while the card goes on with the next level.
Hold: touch ROOT/HOLD -> no new job starts; the running ones finish, the CPU lane drains, the run exits (rm to resume).
Pause one card: touch ROOT/HOLD-<device name> (e.g. HOLD-c3) -> that card finishes its jobs, frees its locks and idles;
the others go on; the swj pass picks an unpaused swj card (waits if all are paused); rm to rejoin (rxrun.pool).
Status: ROOT/status.txt (one line, rewritten after every job and step; also --status): level, jobs, jobs/h and the
throughput in estimated R9700-seconds per wall second over the measured jobs, ETA (CST) of the whole ladder.

CONF (JSON or TOML): root, settings (the settings file), python, env {..} (PYTHONPATH, RXQ_RXM, RXQ_EXL3_SRC, ...),
inputs {ckpt, hessians, hin, hout}, model + swj_meta (swjfit's MODEL / META), template (config.json for assemble;
default: model's config.json + quantization_config exl3), passes ["1","2","3"], levels [K-codebook, ...] (optional
subset), layers ("A-B", dry runs: partial assembly), steps [jobs, assemble, swj, post, recipe], swj_batch (1),
swj_env {..}, dup_frac (0.015), dup_seed_delta (1), chunk (jobs a card-lock session; 0 = the whole level), mode
(persist | proc), ram_lock, allow_standin (tests), devices [{name, device, host: "local", lock, ram_lock, workers,
vram_gib, max_job_vram, nice, env, kinds: ["quant", "swj"]}]. Only host "local" runs today; a remote host (the 9070 XT
boxes) is a device whose worker command goes through ssh (README.md "Hosts")."""
import argparse, concurrent.futures as cf, copy, hashlib, json, os, random, shutil, subprocess, sys, threading, time

from rxq import config as C, ladder, qwen35
from . import pool

CST = -6 * 3600


def cst(t=None, fmt="%m-%d %H:%M"):
    return time.strftime(fmt, time.gmtime((t or time.time()) + CST)) + " CST"


def load_conf(p):
    if p.endswith(".toml"):
        import tomllib
        return tomllib.load(open(p, "rb"))
    return json.load(open(p))


def swj_planned(cfg, lv, steps=None):
    """the level has a swj pass under cfg (and steps, when given)"""
    from rxq import post
    return (steps is None or "swj" in steps) and any(x[0] == "swj" for x in post.plan_level(cfg, lv["K"]))


def level_current(D, cfg, lv, steps=None):
    """D/DONE exists and was made at cfg's scales stamp where the level has a swj pass (stamp split, user 11-12:
    a DONE without "scales" counts only for levels without swj)"""
    if not os.path.exists(f"{D}/DONE"): return False
    try: got = json.load(open(f"{D}/DONE")).get("scales")
    except ValueError: return False
    return got == (C.scales_stamp(cfg) if swj_planned(cfg, lv, steps) else None)


class Run:
    def __init__(self, conf):
        self.c = conf; self.root = conf["root"]; os.makedirs(self.root, exist_ok=True)
        self.cfg = C.load(conf["settings"]); self.st = C.settings_stamp(self.cfg); self.ss = C.scales_stamp(self.cfg)
        self.py = conf.get("python", sys.executable)
        self.env = dict(os.environ); self.env.update({k: str(v) for k, v in (conf.get("env") or {}).items()})
        self.steps = conf.get("steps", ["jobs", "assemble", "swj", "post", "recipe"])
        passes = conf.get("passes", ["1", "2", "3"])
        lvs = [v for v in ladder.levels(self.cfg) if v["pass"] in passes]
        if conf.get("levels"): lvs = [v for v in lvs if f"{ladder.fmt_k(v['K'])}-{v['codebook']}" in conf["levels"] or v["name"] in conf["levels"]]
        self.levels = lvs
        self.devs = conf.get("devices") or []
        for d in self.devs:
            if d.get("host", "local") != "local": raise SystemExit(f"device {d['name']}: host {d['host']}: remote hosts are not wired yet")
        self.hold = f"{self.root}/HOLD"
        self.log_lock = threading.Lock(); self.cur = ""; self._est = {}

    # ------------------------------------------------------------------------------------------------ helpers
    def log(self, msg):
        line = f"{cst(fmt='%m-%d %H:%M:%S')} {msg}"
        with self.log_lock:
            print(line, flush=True)
            with open(f"{self.root}/run.log", "a") as f: f.write(line + "\n")

    def held(self): return os.path.exists(self.hold)

    def D(self, lv): return f"{self.root}/{lv['name']}"

    def sh(self, args, log, env=None, nice=10):
        e = dict(self.env); e.update(env or {})
        with open(log, "a") as lf:
            r = subprocess.run(["nice", "-n", str(nice)] + args, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=e)
        return r.returncode

    # --------------------------------------------------------------------------------------------- jobs, dups
    def specs(self, lv):
        d = f"{self.D(lv)}/specs"
        if os.path.exists(f"{d}/jobs.jsonl"): return d
        inp = self.c["inputs"]
        jobs = ladder.level_jobs(self.cfg, lv, "local", "w105", None, layers=self.c.get("layers"))
        shutil.rmtree(d + ".part", ignore_errors=True); os.makedirs(d + ".part")
        with open(f"{d}.part/jobs.jsonl", "w") as fh:
            for jb in jobs:
                p = jb["params"]
                spec = {"type": "rxq.quant", "version": 1, "level": p["level"], "K": ladder._kn(p["K"]), "codebook": p["codebook"],
                        "seed_offset": p["settings"]["seed_offset"], "keys": list(p["keys"]), "settings_stamp": p["settings"]["rxq"],
                        "config": os.path.abspath(self.c["settings"]),
                        "inputs": {"ckpt": inp["ckpt"], "hessians": inp["hessians"], "hin": inp.get("hin"), "wpre": None,
                                   "hout": inp["hout"] if p["hout"] else None}}
                n = f"L{p['layer']:02d}-g{p['group']}"
                json.dump(spec, open(f"{d}.part/{n}.json", "w"), indent=1)
                fh.write(json.dumps({"name": n, "est_s": p["req"]["est_s"], "vram_gb": p["req"]["vram_gb"], "keys": p["keys"]}) + "\n")
        os.replace(d + ".part", d)
        return d

    def job_list(self, lv):
        d = self.specs(lv)
        rows = [json.loads(x) for x in open(f"{d}/jobs.jsonl")]
        return [pool.make_job(r["name"], f"{d}/{r['name']}.json", f"{self.D(lv)}/out/{r['name']}", cfg=self.cfg) for r in rows]

    def dups(self, lv):
        """the level's duplicate-seed jobs (deterministic: the same picks on every rerun)"""
        frac = float(self.c.get("dup_frac", 0.015))
        if frac <= 0: return []
        d = self.specs(lv); rows = [json.loads(x) for x in open(f"{d}/jobs.jsonl")]
        keys = [(r["name"], k) for r in rows for k in r["keys"]]
        rng = random.Random(int(hashlib.sha256(f"{self.st}:{lv['name']}:dup".encode()).hexdigest()[:16], 16))
        fams = sorted({qwen35.family(k) for _, k in keys})
        off = [v["name"] for v in ladder.levels(self.cfg)].index(lv["name"]) if lv in ladder.levels(self.cfg) else 0
        n = max(1, round(frac * len(keys))); picks = []
        for i in range(n):
            fam = fams[(off + i) % len(fams)]
            cand = [x for x in keys if qwen35.family(x[1]) == fam and x not in picks]
            if cand: picks.append(rng.choice(cand))
        out = []
        for gname, key in picks:
            sp = json.load(open(f"{d}/{gname}.json"))
            sp = copy.deepcopy(sp); sp["keys"] = [key]; sp["seed_offset"] = sp.get("seed_offset", 0) + int(self.c.get("dup_seed_delta", 1))
            name = f"D-{gname.split('-')[0]}-{qwen35.family(key)}"
            jd = f"{self.D(lv)}/dup/{name}"; os.makedirs(jd, exist_ok=True)
            if not os.path.exists(f"{jd}/spec.json"):
                json.dump(sp, open(f"{jd}/spec.json.part", "w"), indent=1); os.replace(f"{jd}/spec.json.part", f"{jd}/spec.json")
            out.append(pool.make_job(name, f"{jd}/spec.json", jd, cfg=self.cfg))
        return out

    def run_jobs(self, lv):
        jobs = self.job_list(lv); dup = self.dups(lv)
        q = pool.JobQueue(jobs + dup)                       # the real jobs first, the duplicates after them
        total = len(jobs) + len(dup); left = len(q)
        if not left: return True
        self.log(f"{lv['name']}: {left} of {total} jobs to run ({len(dup)} duplicate-seed)")
        qd = [d for d in self.devs if "quant" in d.get("kinds", ["quant"])]
        if not qd: raise SystemExit("no device of kind quant")
        logdir = f"{self.D(lv)}/logs"; errs = []

        def on_done(rec):
            self.status(lv, rec)

        def one(dev):
            try:
                pool.run_jobs(q, dev, logdir, mode=self.c.get("mode", "persist"), hold=self.hold, chunk=self.c.get("chunk") or None,
                              py=self.py, on_done=on_done, tag=lv["name"], guard_gib=float(self.c.get("guard_gib", 3.0)))
            except Exception as e:                       # noqa: BLE001
                errs.append(f"{dev['name']}: {e}")
        qd = [dict(d, env={**(self.c.get("env") or {}), **(d.get("env") or {})}) for d in qd]   # the job processes' env
        ths = [threading.Thread(target=one, args=(d,)) for d in qd]
        for t in ths: t.start()
        for t in ths: t.join()
        if errs: raise pool.PoolError("; ".join(errs))
        return all(pool.job_done(j["_spec"], j["out"]) for j in jobs + dup)

    # ---------------------------------------------------------------------------------------------- the steps
    def template(self):
        t = self.c.get("template")
        if t: return t
        if not self.c.get("model"): return None
        p = f"{self.root}/template.json"
        if not os.path.exists(p):
            c = json.load(open(f"{self.c['model']}/config.json")); c["quantization_config"] = {"quant_method": "exl3"}
            json.dump(c, open(p + ".part", "w"), indent=1); os.replace(p + ".part", p)
        return p

    def assemble(self, lv):
        D = self.D(lv); dst = f"{D}/level"
        if os.path.exists(f"{dst}/stamp.json"): return
        shutil.rmtree(dst + ".part", ignore_errors=True)
        outs = sorted(f"{D}/out/{n}" for n in os.listdir(f"{D}/out"))
        args = [self.py, "-m", "rxq", "assemble", dst + ".part"] + outs
        if self.template(): args += ["--template", self.template()]
        if self.c.get("layers"): args.append("--keys-from")
        if self.c.get("allow_standin"): args.append("--allow-standin")
        t = time.time()
        with pool.Locks(None, self.c.get("ram_lock")):
            rc = self.sh(args, f"{D}/assemble.log")
        if rc or not os.path.exists(f"{dst}.part/stamp.json"): raise RuntimeError(f"assemble {lv['name']} rc {rc} ({D}/assemble.log)")
        os.replace(dst + ".part", dst)
        self.log(f"{lv['name']}: assembled in {time.time() - t:.0f} s")

    def swj_needed(self, lv):
        return swj_planned(self.cfg, lv, self.steps)

    def swjdir(self): return f"{self.root}/swj-{self.ss[:8]}"

    def is_done(self, lv): return level_current(self.D(lv), self.cfg, lv, self.steps)

    def swj(self, lvs):
        S = self.swjdir(); todo = [v for v in lvs if not os.path.exists(f"{S}/{v['name']}/.done")]
        if not todo: return
        sw = [d for d in self.devs if "swj" in d.get("kinds", [])]
        if not sw: raise SystemExit("no device of kind swj")
        while all(pool.paused(self.hold, d) for d in sw):         # every swj card paused (HOLD-<dev>): wait
            if self.held(): raise RuntimeError("HOLD while the swj cards are paused")
            time.sleep(5)
        dev = next(d for d in sw if not pool.paused(self.hold, d))
        tag = "+".join(ladder.fmt_k(v["K"]) for v in todo)
        args = [self.py, "-m", "rxq.swjfit", self.c["model"], self.c["swj_meta"], S, "--scratch", f"{self.root}/swjscr/{tag}",
                "--fused", "--no-cache"] + (["--numerics", self.cfg["post"]["swj"]["numerics"]]
                                            if self.cfg["post"]["swj"].get("numerics", "f32") != "f32" else [])
        for v in todo: args += ["--src", f"{v['name']}={self.D(v)}/level:level"]
        env = {"VRAM_CAP_GB": "30", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1", "TOKENIZERS_PARALLELISM": "false",
               **{k: str(v) for k, v in (dev.get("env") or {}).items()}, **{k: str(v) for k, v in (self.c.get("swj_env") or {}).items()}}
        self.cur = f"swj {tag}"; self.status(todo[0]); t = time.time()
        self.log(f"swj {tag}: start on {dev['name']}")
        with pool.Locks(dev.get("lock"), dev.get("ram_lock")):
            rc = self.sh(args, f"{self.root}/swj-{tag}.log", env=env, nice=dev.get("nice", 5))
        bad = [v["name"] for v in todo if not os.path.exists(f"{S}/{v['name']}/.done")]
        if rc or bad: raise RuntimeError(f"swj {tag} rc {rc}, not done {bad} ({self.root}/swj-{tag}.log; a rerun resumes)")
        dt = time.time() - t
        with open(f"{self.root}/swj-times.jsonl", "a") as f: f.write(json.dumps({"levels": [v["name"] for v in todo], "wall": round(dt, 1)}) + "\n")
        shutil.rmtree(f"{self.root}/swjscr/{tag}", ignore_errors=True)
        self.log(f"swj {tag}: done in {dt / 60:.0f} min")

    def post(self, lv):
        D = self.D(lv); F = f"{D}/final"
        if os.path.exists(f"{F}/.done"): return
        for p in (F + ".part", F + ".part.scales"): shutil.rmtree(p, ignore_errors=True)
        scales = []
        if self.swj_needed(lv): scales.append(f"{self.swjdir()}/{lv['name']}")
        if os.path.exists(f"{D}/level/c9/.done"): scales.append(f"{D}/level/c9")
        t = time.time()
        if scales:
            with pool.Locks(None, self.c.get("ram_lock")):
                rc = self.sh([self.py, "-m", "rxq.post", "apply", f"{D}/level", ",".join(scales), F + ".part"], f"{D}/post.log")
            if rc or not os.path.exists(f"{F}.part/.done"): raise RuntimeError(f"post {lv['name']} rc {rc} ({D}/post.log)")
        else:
            shutil.copytree(f"{D}/level", F + ".part", copy_function=os.link); open(f"{F}.part/.done", "w").close()
        os.replace(F + ".part", F); shutil.rmtree(F + ".part.scales", ignore_errors=True)
        self.log(f"{lv['name']}: post ({', '.join(os.path.basename(s) for s in scales) or 'copy'}) in {time.time() - t:.0f} s")

    def recipe(self, lv):
        D = self.D(lv); R = f"{D}/final.recipe"
        if os.path.exists(R): return
        with open(R + ".part", "w") as f:
            rc = subprocess.run([self.py, "-m", "rxq.mix", "recipe", f"{D}/final"], stdout=f, stderr=subprocess.PIPE, text=True, env=self.env)
        if rc.returncode: raise RuntimeError(f"recipe {lv['name']}: {rc.stderr[-300:]}")
        os.replace(R + ".part", R)

    def finish(self, lv):
        D = self.D(lv)
        json.dump({"level": lv["name"], "steps": self.steps, "settings": self.st, "scales": self.ss if self.swj_needed(lv) else None,
                   "t": cst()}, open(f"{D}/DONE.part", "w"))
        os.replace(f"{D}/DONE.part", f"{D}/DONE")
        self.log(f"{lv['name']}: DONE")
        self.status(lv)

    # ------------------------------------------------------------------------------------------------ status
    def est_left(self):
        """estimated R9700-seconds of the jobs not done, over every level not DONE; swj passes left"""
        s, nswj = 0.0, 0
        for lv in self.levels:
            if self.is_done(lv): continue
            if lv["name"] not in self._est:
                jobs = ladder.level_jobs(self.cfg, lv, "m", "r", None, layers=self.c.get("layers"))
                self._est[lv["name"]] = [(f"L{j['params']['layer']:02d}-g{j['params']['group']}", j["params"]["req"]["est_s"]) for j in jobs]
            for n, e in self._est[lv["name"]]:
                if not os.path.exists(f"{self.D(lv)}/out/{n}/job.json"): s += e
            if self.swj_needed(lv) and not os.path.exists(f"{self.swjdir()}/{lv['name']}/.done"): nswj += 1
        return s, nswj

    def measured(self, window=400):
        """(jobs/h, est-seconds per busy second) over the last `window` finished jobs of this root (busy = the union of
        the jobs' wall intervals: lock waits and holds do not count)"""
        recs = []
        for lv in self.levels:
            p = f"{self.D(lv)}/logs/jobs.jsonl"
            if os.path.exists(p): recs += [json.loads(x) for x in open(p) if '"rc": 0' in x]
        recs = sorted(recs, key=lambda r: r["t1"])[-window:]
        if not recs: return None, None
        iv = sorted((r["t0"], r["t1"]) for r in recs); busy, cur = 0.0, None
        for a, b in iv:
            if cur is None or a > cur[1]:
                if cur: busy += cur[1] - cur[0]
                cur = [a, b]
            else: cur[1] = max(cur[1], b)
        busy += cur[1] - cur[0]
        return len(recs) / busy * 3600, sum(r["est"] for r in recs) / busy

    def swj_h(self):
        p = f"{self.root}/swj-times.jsonl"
        if not os.path.exists(p): return float(self.c.get("swj_h", 0.75))
        rows = [json.loads(x) for x in open(p)]
        return sum(r["wall"] for r in rows) / sum(len(r["levels"]) for r in rows) / 3600

    def status(self, lv=None, rec=None):
        left, nswj = self.est_left(); jph, rate = self.measured()
        ndone = sum(self.is_done(v) for v in self.levels)
        if lv is not None:
            jobs = self._est.get(lv["name"]) or []
            jd = sum(os.path.exists(f"{self.D(lv)}/out/{n}/job.json") for n, _ in jobs)
            where = f"{lv['name']} ({lv['pass']}) jobs {jd}/{len(jobs)}" + (f", {self.cur}" if self.cur else "")
        else: where = "-"
        if rate:
            h = left / rate / 3600 + nswj * self.swj_h()
            eta = f"ETA {cst(time.time() + h * 3600)} ({h:.1f} h: quant {left / rate / 3600:.1f} + swj {nswj} x {self.swj_h():.2f})"
            meas = f"{jph:.0f} jobs/h, {rate:.2f} est-s/s"
        else:
            h = left / 3600 + nswj * self.swj_h()
            eta = f"ETA ~{h:.1f} h at the model's est (no jobs measured yet)"; meas = "- jobs/h"
        line = (f"{cst()} | levels {ndone}/{len(self.levels)} | {where} | {meas} | {eta}" + (" | HOLD" if self.held() else "")
                + "".join(f" | paused {d['name']}" for d in self.devs if pool.paused(self.hold, d)))
        with self.log_lock:
            open(f"{self.root}/status.txt.part", "w").write(line + "\n"); os.replace(f"{self.root}/status.txt.part", f"{self.root}/status.txt")
        return line

    # ------------------------------------------------------------------------------------------------- main
    def run(self):
        self.log(f"rxrun.run start: settings {self.st}, {len(self.levels)} levels, steps {self.steps}, devices "
                 f"{[(d['name'], d.get('workers', 1)) for d in self.devs]}")
        lane = cf.ThreadPoolExecutor(1); futs = []; swj_wait = []; batch = int(self.c.get("swj_batch", 1))

        def cpu(fn, *a):
            futs.append(lane.submit(fn, *a))

        def check():
            for f in [f for f in futs if f.done()]:
                futs.remove(f); f.result()                # raises the lane's error here

        def flush_swj(force=False):
            while swj_wait and (force or len(swj_wait) >= batch):
                b = swj_wait[:batch]; del swj_wait[:batch]
                for f in list(futs): f.result()            # their assemblies first
                self.swj(b)
                for v in b: cpu(self.post_tail, v)
        try:
            for lv in self.levels:
                check()
                if self.is_done(lv): self.log(f"{lv['name']}: skip (DONE)"); continue
                if os.path.exists(f"{self.D(lv)}/DONE"):          # DONE at another scales stamp: redo swj + post only
                    self.log(f"{lv['name']}: DONE at another scales stamp: redo swj / post (jobs and assembly kept)")
                    for x in ("DONE", "final.recipe"):
                        if os.path.exists(f"{self.D(lv)}/{x}"): os.remove(f"{self.D(lv)}/{x}")
                    shutil.rmtree(f"{self.D(lv)}/final", ignore_errors=True)
                if self.held(): self.log("HOLD: stop before " + lv["name"]); break
                self.cur = ""; self.status(lv)
                if "jobs" in self.steps and not self.run_jobs(lv):
                    self.log(f"{lv['name']}: jobs not all done (held)"); break
                if "assemble" not in self.steps: cpu(self.finish, lv); continue
                cpu(self.assemble, lv)
                if self.swj_needed(lv): swj_wait.append(lv); flush_swj()
                else: cpu(self.post_tail, lv)
            if swj_wait and not self.held(): flush_swj(force=True)
            for f in futs: f.result()
        finally:
            lane.shutdown(wait=True)
        self.log(f"rxrun.run end: {self.status()}")

    def post_tail(self, lv):
        if "post" in self.steps: self.post(lv)
        if "recipe" in self.steps and "post" in self.steps: self.recipe(lv)
        self.finish(lv)

    def estimate(self, rate=None, swj_h=None):
        """the projected card time of the whole ladder: est R9700-seconds / rate (measured est-s per wall second) +
        swj passes"""
        tot, n, nswj = 0.0, 0, 0
        for lv in self.levels:
            for j in ladder.level_jobs(self.cfg, lv, "m", "r", None, layers=self.c.get("layers")):
                tot += j["params"]["req"]["est_s"]; n += 1
            nswj += self.swj_needed(lv)
        jph, mr = self.measured(); rate = rate or mr or 1.0; sh = swj_h if swj_h is not None else self.swj_h()
        dup = float(self.c.get("dup_frac", 0.015))
        q = tot / rate / 3600 * (1 + dup)
        return {"levels": len(self.levels), "jobs": n, "est_card_h": round(tot / 3600, 1), "rate": round(rate, 3),
                "quant_h": round(q, 1), "swj_levels": nswj, "swj_h": round(nswj * sh, 1),
                "card_days": round((q + nswj * sh) / 24, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxrun.run")
    ap.add_argument("conf"); ap.add_argument("--status", action="store_true"); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--estimate", action="store_true"); ap.add_argument("--rate", type=float); ap.add_argument("--swj-h", type=float)
    a = ap.parse_args(argv)
    r = Run(load_conf(a.conf))
    if a.status: print(r.status()); return
    if a.estimate: print(json.dumps(r.estimate(a.rate, a.swj_h))); return
    if a.plan:
        for lv in r.levels:
            st = "DONE" if r.is_done(lv) else "stale" if os.path.exists(f"{r.D(lv)}/DONE") else "todo"
            print(f"{lv['pass']:2s} {lv['name']:28s} {st}  swj {r.swj_needed(lv)}")
        return
    r.run()


if __name__ == "__main__":
    main()
