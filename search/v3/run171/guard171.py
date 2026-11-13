#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""guard171.py COORD.json PREFIX [--once] -- the search's disk guard + coordinator keeper (#171; VM, light). Every POLL s:
  1 ds8 df and the mix containers MIXES/<PREFIX>*.rad; every container whose tag has a line in the coordinator's
    results.jsonl (scored or failed) is deleted (a mix is used by one eval only; the .mix spec files stay);
  2 /data free < HOLD_GB (100): touch ROOT/HOLD (the coordinator takes no new units and exits once idle; the driver keeps
    queueing); >= RESUME_GB (130) and the HOLD is ours: remove it;
  3 no HOLD and no coordinator running (its pid file): start `python3 -m rxrun.multi COORD --follow` (cwd = the repo's
    pipeline/), log ROOT/multi.out. So a HOLD that drained the coordinator, or a crash, is resumed.
Log: ROOT/guard171.log. Stop: kill the guard (the coordinator keeps running; its pid is in ROOT/guard171.coord.pid)."""
import json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); PIPE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "pipeline"))
SSH = os.environ.get("DS8_SSH", "/workspace/rad-local/bin/ssh-ds8.sh")
MIXES = os.environ.get("MIXES", "/data/q27/mixes")
HOLD_GB, RESUME_GB, POLL = int(os.environ.get("HOLD_GB", 100)), int(os.environ.get("RESUME_GB", 130)), int(os.environ.get("POLL", 60))


def cst(): return time.strftime("%m-%d %H:%M:%S", time.gmtime(time.time() - 6 * 3600)) + " CST"


class Guard:
    def __init__(self, coord, prefix):
        self.cp = os.path.abspath(coord); self.c = json.load(open(coord)); self.R = self.c["root"]; self.prefix = prefix
        self.res = f"{self.R}/results.jsonl"; self.off = 0; self.done = set(); self.pidf = f"{self.R}/guard171.coord.pid"

    def log(self, m):
        with open(f"{self.R}/guard171.log", "a") as f: f.write(f"{cst()} {m}\n")

    def scan_results(self):
        if not os.path.exists(self.res): return
        with open(self.res) as f:
            f.seek(self.off)
            while True:
                l = f.readline()
                if not l or not l.endswith("\n"): break
                self.off = f.tell()
                try: t = json.loads(l).get("tag", "")
                except ValueError: continue
                if t.startswith(self.prefix): self.done.add(t)

    def coord_alive(self):
        try: pid = int(open(self.pidf).read())
        except (OSError, ValueError): return False
        try: os.kill(pid, 0)
        except OSError: return False
        try: return "rxrun.multi" in open(f"/proc/{pid}/cmdline").read()
        except OSError: return False

    def start_coord(self):
        p = subprocess.Popen([sys.executable, "-m", "rxrun.multi", self.cp, "--follow"], cwd=PIPE, stdin=subprocess.DEVNULL,
                             stdout=open(f"{self.R}/multi.out", "a"), stderr=subprocess.STDOUT, start_new_session=True)
        open(self.pidf, "w").write(str(p.pid)); self.log(f"coordinator started (pid {p.pid})")

    def step(self):
        self.scan_results()
        out = subprocess.run([SSH, f"df -BG --output=avail /data | tail -1; ls {MIXES}/ | grep '^{self.prefix}.*\\.rad$' || true"],
                             capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL).stdout.split()
        if not out: self.log("ds8 unreachable"); return
        df = int(out[0].rstrip("G")); rads = out[1:]
        rm = [r for r in rads if r[:-4] in self.done]
        if rm:
            subprocess.run([SSH, "cd " + MIXES + " && rm -f " + " ".join(rm)], timeout=60, stdin=subprocess.DEVNULL)
            self.log(f"deleted {len(rm)} scored mixes: {' '.join(r[:-4] for r in rm)}")
        hold = f"{self.R}/HOLD"
        if df < HOLD_GB and not os.path.exists(hold):
            open(hold, "w").write(f"guard171: ds8 df {df} GB\n"); self.log(f"HOLD on: ds8 /data {df} GB free (< {HOLD_GB})")
        elif df >= RESUME_GB and os.path.exists(hold) and open(hold).read().startswith("guard171"):
            os.remove(hold); self.log(f"HOLD off: ds8 /data {df} GB free")
        if not os.path.exists(hold) and not self.coord_alive(): self.start_coord()
        with open(f"{self.R}/guard171.status", "w") as f: f.write(f"{cst()} df {df} GB, {len(rads) - len(rm)} mixes kept, hold {os.path.exists(hold)}, coordinator {self.coord_alive()}\n")


def main(argv):
    G = Guard(argv[0], argv[1]); G.log(f"guard171 start (hold < {HOLD_GB} GB, resume >= {RESUME_GB} GB, poll {POLL} s)")
    signal.signal(signal.SIGTERM, lambda *a: (G.log("guard171 stop"), sys.exit(0)))
    while True:
        try: G.step()
        except Exception as e: G.log(f"step failed: {e!r}")                  # noqa: BLE001 -- keep guarding
        if "--once" in argv: return 0
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
