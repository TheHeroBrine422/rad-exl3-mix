#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""run173.py CONF.json -- (w158, #173-03) build, census and score A8 calibration mixes on ds8 (+ box2 cards over the 10 GbE
link). One process on ds8 (nohup); restartable: every step is skipped when its output exists (.part then rename).

CONF: {"root": DIR, "menu": MENU.json, "specs": [SPECS.jsonl, ..], "base": BASE.rad, "slices": {"NAME": {"ds8": GOLDEN,
       "box2": GOLDEN}}, "cards": CARDS.txt, "max_ready": 3, "min_free_gb": 150, "box2_min_free_gb": 105}
SPECS line: {"tag", "assign": {weight: option}, "ab": "t4" | "t5", "slices": [NAME, ..], "census": [extra census args]}
Per spec (ROOT/mix, ROOT/census, ROOT/scores):
  .mix   every trunk weight -> its level pack (menu src: packs-mad), output.weight -> head donor, token_embd.weight ->
         embedding donor, drop blk.N.ssm_a / ssm_b + add blk.N.ssm_ab (a/b donor K4 / K5), drop every vision-tower weight
         (v.*: not in the menu), act <w> a8g128 on the 336 trunk linears and the 48 ssm_ab
  radmix under conv.lock (exclusive) + ram.lock (shared); census (patched store/census.py: mad = encoding v4) --fast with
         --mix / --base and --expect-act a8g128 --expect-emb trellis --max-version 4 + the spec's own (codebook / head);
         a census failure drops the spec (logged), the container is deleted
  score  radiance (rad10-e157b) on the first free card of CARDS ("ds8:N" / "box2:N" lines; re-read every pass, so cards can be
         added while it runs), non-blocking card-lock tries only (never queues on a held card) + ram.lock shared; box2: the
         container is copied over the link, scored there, the json / rows copied back and the copy deleted
  the container is deleted once every slice of its spec is scored (the .mix and census stay)."""
import json, os, queue, re, shutil, subprocess, sys, threading, time

Q = "/data/q27"; W = f"{Q}/w158"; E = f"{Q}/rad10-e157b"; PY = f"{Q}/venv/bin/python"
BOX = ["ssh", "-i", "/root/.ssh/dbox", "-o", "StrictHostKeyChecking=no", "-o", "Compression=no", "-c", "aes128-gcm@openssh.com", "root@10.29.6.6"]
LOCKS = {"ds8": f"{Q}/locks", "box2": "/data/q27/locks"}
LOGL = threading.Lock()


def now(): return time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() - 6 * 3600)) + " CST"


def log(C, msg):
    with LOGL, open(f"{C['root']}/run.log", "a") as f: f.write(f"{now()} {msg}\n")


def free_gb(path): s = os.statvfs(path); return s.f_bavail * s.f_frsize / 2 ** 30


def sh(cmd, out=None, env=None):
    with open(out or os.devnull, "a") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)


def trunk_names():
    out = []
    for l in range(64):
        out += [f"blk.{l}.{t}.weight" for t in (("attn_qg", "attn_k", "attn_v", "attn_output") if l % 4 == 3 else ("ssm_qkv", "ssm_z", "ssm_out"))]
        out += [f"blk.{l}.ffn_gate_up.weight", f"blk.{l}.ffn_down.weight"]
    return out


class Run:
    def __init__(self, conf):
        self.C = C = json.load(open(conf)); self.conf = conf
        for d in ("mix", "census", "scores", "tmp"): os.makedirs(f"{C['root']}/{d}", exist_ok=True)
        M = json.load(open(C["menu"])); self.src = {}; packs = dict(zip(M["built"]["trunk_levels"], M["built"]["packs"]))
        for t in M["tensors"]:
            for o in t["opts"]:
                p = (o.get("src") or {}).get("path") or (packs.get(o["o"]) if t["w"] not in ("output.weight", "token_embd.weight") else None)
                self.src[(t["w"], o["o"])] = p
        self.ab = C["ab"]; self.trunk = trunk_names()
        self.names = subprocess.run([PY, "-I", f"{W}/code/radnames.py", C["base"]], capture_output=True, text=True).stdout.split()
        self.vis = [n for n in self.names if n.startswith("v.")]
        self.specs = []
        for p in C["specs"]:
            for l in open(p):
                if l.strip(): self.specs.append(json.loads(l))
        if C.get("order") == "size": self.specs.sort(key=lambda s: s.get("meta", {}).get("bpw_menu", 0))   # small containers first (box2's disk)
        self.ready = queue.Queue(); self.busy = set(); self.lk = threading.Lock(); self.done_mix = False

    def p(self, kind, tag, ext=""): return f"{self.C['root']}/{kind}/{tag}{ext}"

    def scored(self, s): return all(os.path.exists(self.p("scores", f"{s['tag']}-{sl}", ".json")) for sl in s["slices"])

    def mixfile(self, s):
        a = s["assign"]; L = []
        for w in self.trunk + ["output.weight", "token_embd.weight"]:
            src = self.src.get((w, a[w]))
            if not src: raise KeyError(f"{s['tag']}: no source for {w} = {a[w]}")
            L.append(f"{w} {src}")
        ab = self.ab[s["ab"]]
        for l in range(64):
            if l % 4 != 3: L += [f"drop blk.{l}.ssm_a.weight", f"drop blk.{l}.ssm_b.weight", f"add blk.{l}.ssm_ab.weight {ab}"]
        L += [f"drop {n}" for n in self.vis]
        if s.get("act", True):
            L += [f"act {w} a8g128" for w in self.trunk] + [f"act blk.{l}.ssm_ab.weight a8g128" for l in range(64) if l % 4 != 3]
        return "\n".join(L) + "\n"

    # ------------------------------------------------------------------ mixer
    def mixer(self):
        C = self.C
        for s in self.specs:
            t = s["tag"]
            if self.scored(s) or os.path.exists(self.p("census", t, ".fail")): continue
            while self.ready.qsize() >= C.get("max_ready", 3) or free_gb(C["root"]) < C.get("min_free_gb", 150) + 30:
                if os.path.exists(f"{C['root']}/STOP"): return
                time.sleep(20)
            if os.path.exists(f"{C['root']}/STOP"): return
            rad = self.p("mix", t, ".rad"); mf = self.p("mix", t, ".mix"); ok = self.p("census", t, ".ok")
            if not (os.path.exists(rad) and os.path.exists(ok)):
                open(mf, "w").write(self.mixfile(s))
                if os.path.exists(rad + ".part"): os.remove(rad + ".part")
                t0 = time.time()
                rc = sh(["flock", f"{Q}/locks/conv.lock", "flock", "-s", f"{Q}/locks/ram.lock", "nice", "-n", "10",
                         f"{W}/tools/radmix", rad + ".part", C["base"], mf], self.p("mix", t, ".mixlog"))
                if rc: log(C, f"radmix {t} FAILED rc {rc}"); continue
                cargs = (["--mix", mf, "--base", C["base"], "--fast", "--expect-emb", "trellis", "--max-version", "4"]
                         + (["--expect-act", "a8g128"] if s.get("act", True) else []) + s.get("census", []))
                rc = sh([PY, "-I", f"{W}/code/census.py", rad + ".part"] + cargs, self.p("census", t, ".txt"))
                if rc:
                    os.remove(rad + ".part"); open(self.p("census", t, ".fail"), "w").write("census mismatch\n")
                    log(C, f"census {t} FAILED: {open(self.p('census', t, '.txt')).read().strip().splitlines()[-1]}"); continue
                os.replace(rad + ".part", rad); open(ok, "w").write("CENSUS PASS\n")
                log(C, f"mix {t} {os.path.getsize(rad)} B {(os.path.getsize(rad) * 8 / 26895998464):.4f} bpw ({time.time() - t0:.0f} s)")
            for sl in s["slices"]:
                if not os.path.exists(self.p("scores", f"{t}-{sl}", ".json")): self.ready.put((s, sl))
        self.done_mix = True

    # ------------------------------------------------------------------ scorers
    def cards(self):
        try: return [l.split()[0] for l in open(self.C["cards"]) if l.strip() and not l.startswith("#")]
        except OSError: return []

    def finish(self, s):
        if self.scored(s):
            r = self.p("mix", s["tag"], ".rad")
            if os.path.exists(r): os.remove(r); log(self.C, f"{s['tag']}: all slices scored, container deleted")

    def score_ds8(self, card, s, sl):
        C = self.C; t = s["tag"]; o = self.p("scores", f"{t}-{sl}"); g = C["slices"][sl]["ds8"]
        cmd = (f"exec 8>{Q}/locks/card{card}.lock; flock -n 8 || exit 75; flock -s {Q}/locks/ram.lock env ROCR_VISIBLE_DEVICES={card} "
               f"RADIANCE_HOME={E}/home {E}/install/bin/radiance --model {self.p('mix', t, '.rad')} --tp 1 --kv-cache-dtype bf16 "
               f"--max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref {g} --kld-out {o}.json.part > {o}.log 2>&1 "
               f"&& mv {o}.json.part.rows {o}.json.rows && mv {o}.json.part {o}.json")
        return subprocess.call(["bash", "-c", cmd], stdin=subprocess.DEVNULL)

    def score_box2(self, card, s, sl):
        C = self.C; t = s["tag"]; o = self.p("scores", f"{t}-{sl}"); g = C["slices"][sl]["box2"]; R = "/data/q27/w158/remote"
        rad = self.p("mix", t, ".rad"); size = os.path.getsize(rad) / 2 ** 30
        chk = subprocess.run(BOX + [f"mkdir -p {R}; df -BG --output=avail /data | tail -n 1; exec 8>{LOCKS['box2']}/card{card}.lock; flock -n 8 && echo FREE"],
                             capture_output=True, text=True, timeout=60)
        if chk.returncode or "FREE" not in chk.stdout: return 75
        av = int(re.sub(r"\D", "", chk.stdout.split()[0]))
        try: mf = json.load(open(self.conf)).get("box2_min_free_gb", 105)          # re-read: the floor can be changed while running
        except Exception: mf = C.get("box2_min_free_gb", 105)
        if av < mf + size: log(C, f"box2 disk {av} GB: not copying {t}"); return 75
        rc = sh(["rsync", "-a", "--whole-file", "-e", " ".join(BOX[:-1]), rad, f"{BOX[-1]}:{R}/{t}.rad"], self.p("tmp", t, ".rsync"))
        if rc: subprocess.call(BOX + [f"rm -f {R}/{t}.rad"]); return 1
        ro = f"{R}/{t}-{sl}"
        cmd = (f"exec 8>{LOCKS['box2']}/card{card}.lock; flock -n 8 || {{ rm -f {R}/{t}.rad; exit 75; }}; flock -s {LOCKS['box2']}/ram.lock env "
               f"ROCR_VISIBLE_DEVICES={card} RADIANCE_HOME={E}/home {E}/install/bin/radiance --model {R}/{t}.rad --tp 1 --kv-cache-dtype bf16 "
               f"--max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref {g} --kld-out {ro}.json.part > {ro}.log 2>&1 "
               f"&& mv {ro}.json.part.rows {ro}.json.rows && mv {ro}.json.part {ro}.json; rc=$?; rm -f {R}/{t}.rad; exit $rc")
        rc = subprocess.call(BOX + [cmd], stdin=subprocess.DEVNULL)
        if rc: subprocess.call(BOX + [f"rm -f {R}/{t}.rad"]); sh(["rsync", "-e", " ".join(BOX[:-1]), f"{BOX[-1]}:{ro}.log", o + ".log"]); return rc
        for ext in (".log", ".json.rows", ".json"):
            if sh(["rsync", "-e", " ".join(BOX[:-1]), f"{BOX[-1]}:{ro}{ext}", (o + ext + ".part") if ext == ".json" else o + ext]): return 1
        os.replace(o + ".json.part", o + ".json"); subprocess.call(BOX + [f"rm -f {ro}.json.rows"])
        return 0

    def scorer(self, card_id):
        host, card = card_id.split(":")
        fails = {}
        while True:
            if card_id not in self.cards() or os.path.exists(f"{self.C['root']}/STOP"): time.sleep(30); continue
            try: s, sl = self.ready.get(timeout=30)
            except queue.Empty:
                if self.done_mix and self.ready.empty() and not self.busy: return
                continue
            key = (s["tag"], sl)
            with self.lk:
                if key in self.busy or os.path.exists(self.p("scores", f"{s['tag']}-{sl}", ".json")): continue
                self.busy.add(key)
            t0 = time.time()
            try: rc = (self.score_ds8 if host == "ds8" else self.score_box2)(card, s, sl)
            except Exception as e: rc = 1; log(self.C, f"{card_id} {s['tag']} {sl}: exception {e!r}")
            with self.lk: self.busy.discard(key)
            if rc == 75: self.ready.put((s, sl)); time.sleep(45); continue          # card held: back in the queue
            if rc:
                fails[key] = fails.get(key, 0) + 1
                log(self.C, f"{card_id} score {s['tag']} {sl} FAILED rc {rc} (try {fails[key]})")
                if fails[key] < 2: self.ready.put((s, sl))
                time.sleep(30); continue
            j = json.load(open(self.p("scores", f"{s['tag']}-{sl}", ".json")))["all"]
            log(self.C, f"{card_id} score {s['tag']} {sl} kld {j.get('mean_kld', j.get('kld'))} pos {j.get('positions')} ({time.time() - t0:.0f} s)")
            self.finish(s)

    def main(self):
        log(self.C, f"start pid {os.getpid()} ({len(self.specs)} specs)")
        th = [threading.Thread(target=self.mixer, daemon=True)]
        seen = set()
        th[0].start()
        while True:
            for c in self.cards():
                if c not in seen:
                    seen.add(c); x = threading.Thread(target=self.scorer, args=(c,), daemon=True); x.start(); th.append(x)
            if self.done_mix and self.ready.empty() and not self.busy and not any(x.is_alive() for x in th[1:]): break
            if os.path.exists(f"{self.C['root']}/STOP") and not self.busy: break
            time.sleep(20)
        log(self.C, "done")


if __name__ == "__main__":
    Run(sys.argv[1]).main()
