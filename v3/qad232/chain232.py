#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""chain232.py (box2; run232.sh starts it) -- w232 part A: record the bf16 goldens (top-512, bf16 KV) of the xg-fin chunks (sel232.py's
W/docs/xg-fin.c<k>.jsonl, deepest first) as TP2 lanes, then merge them into QROOT/ext/gold/xg-fin with the gen mask (genmask181) and check them.
  lane A = cards 0,1: its two card locks are taken (blocking, polled) and held for the whole run; chunks are claimed by mkdir.
  lane B = cards 2,3: opportunistic -- only when BOTH locks can be taken without waiting and MemAvailable >= 25 GB, one chunk at a time, the locks
           released between chunks (another job may hold cards 2,3 for hours; nothing is preempted).
  A chunk that fails on TP2 (a deep document that does not fit 2 x 32 GB) is retried at the end on all four cards (TP4, the four locks polled).
Restartable: a recorded chunk is W/rec/c<k> with .done (written after the radiance run, as .part then renamed); a restart clears claims and
the .part directories. Log W/chain.log (CT)."""
import fcntl, json, os, shutil, subprocess, sys, threading, time

Q = "/data/q27"; E = os.environ.get
QROOT = E("W232_QROOT", Q + "/queue"); W = E("W232_ROOT", Q + "/w232"); C = E("W232_CODE", W + "/code/v3/qad232"); PY = E("W232_PY", Q + "/venv/bin/python")
LIBS = "/opt/rocm-10.1/lib:/data/q27/ffmpeg-libs"; MODEL = E("W232_MODEL", Q + "/models/qwen38-27b-bf16")
REC_RB, REC_RH = E("W232_RB", Q + "/w206/install/bin/radiance"), Q + "/w206/install/share/radiance"      # the queue's recording build (qenv.sh box2)
LOCKS = E("W232_LOCKS", Q + "/locks"); DOCS = W + "/docs"; FINAL = QROOT + "/ext/gold/xg-fin"
LOCK = threading.Lock(); ABORT = threading.Event(); MERGE = []
MIN_GB = int(E("W232_MIN_GB", 300))


def log(s):
    with LOCK, open(f"{W}/chain.log", "a") as f: f.write(f"{time.strftime('%m-%d %-I:%M %p CST', time.localtime())} {s}\n")


def exists(p): return os.path.exists(p)
def claim(key):
    try: os.mkdir(f"{W}/claims/{key}"); return True
    except FileExistsError: return False


def free_gb(path): return int(subprocess.check_output(["df", "-BG", "--output=avail", path]).split()[-1].decode().strip("G"))
def mem_avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) // 1048576


class CardLocks:
    """flock on /data/q27/locks/card<N>.lock for the given cards (order ascending), plus the shared ram lock (lock order: card, then ram)"""
    def __init__(self, cards): self.cards = cards; self.fds = []

    def try_take(self):
        fds = []
        for c in self.cards + ["ram"]:
            fd = os.open(f"{LOCKS}/{'ram' if c == 'ram' else 'card' + str(c)}.lock", os.O_RDWR | os.O_CREAT)
            try: fcntl.flock(fd, (fcntl.LOCK_SH if c == "ram" else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                for f in fds: os.close(f)
                return False
            fds.append(fd)
        self.fds = fds; return True

    def take(self, what):
        t0 = time.time(); said = False
        while not self.try_take():
            if ABORT.is_set(): raise RuntimeError("abort")
            if not said and time.time() - t0 > 60: log(f"{what}: waiting for the card locks {self.cards}"); said = True
            time.sleep(30)

    def release(self):
        for f in self.fds: os.close(f)
        self.fds = []


def chunks(): return json.load(open(f"{DOCS}/xg-fin.chunks.json"))


def rec_chunk(cards, tp, ch, tag):
    g = f"{W}/rec/c{ch['k']}"
    if exists(f"{g}/.done"): return True
    os.makedirs(f"{W}/rec", exist_ok=True); shutil.rmtree(g + ".part", ignore_errors=True)
    while free_gb(W) < MIN_GB:
        log(f"disk < {MIN_GB} GB free: waiting"); time.sleep(300)
        if ABORT.is_set(): raise RuntimeError("abort")
    mml = (ch["max_tokens"] + 64 + 255) // 256 * 256; seqs = 1 if ch["max_tokens"] > 20000 else (2 if tp == 2 else 4)
    y = f"{W}/rec/c{ch['k']}.{tag.split()[0]}-tp{tp}.yaml"
    open(y, "w").write(f"""memory:
  host-headroom-mib: 8192
models:
  xg_fin_c{ch['k']}:
    model: {MODEL}
    tp: {tp}
    kv-cache-dtype: bf16
    max-model-len: {mml}
    max-num-seqs: {seqs}
    kld-record: {g}.part
    kld-corpus: {DOCS}/{ch['file']}
    kld-top: 512
""")
    s = time.time(); log(f"{tag}: recording chunk {ch['k']} ({ch['docs']} docs, {ch['tokens']} tokens, max {ch['max_tokens']}, TP{tp}, max-model-len {mml}, seqs {seqs})")
    with open(f"{W}/logs/rec-c{ch['k']}-{tag.split()[0]}-tp{tp}.log", "a") as f:
        rc = subprocess.run([REC_RB, "--config", y], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=dict(os.environ, LD_LIBRARY_PATH=LIBS, RADIANCE_HOME=REC_RH, ROCR_VISIBLE_DEVICES=cards)).returncode
    ok = rc == 0 and exists(f"{g}.part/kld.json")
    if ok:
        m = json.load(open(f"{g}.part/kld.json"))
        ok = len(m["docs"]) == ch["docs"] and m["rows"] == ch["rows"] and os.path.getsize(f"{g}.part/top.bin") == m["rows"] * 512 * 8 \
            and os.path.getsize(f"{g}.part/tokens.i32") == 4 * ch["tokens"]
        if not ok: log(f"{tag}: chunk {ch['k']} output has the wrong size / counts (docs {len(m['docs'])} rows {m['rows']}): not accepted")
    log(f"{tag}: record chunk {ch['k']} rc {rc} ok {ok} in {(time.time() - s) / 60:.1f} min ({ch['tokens'] / max(time.time() - s, 1):.0f} tok/s)")
    if ok: open(f"{g}.part/.done", "w").close(); os.replace(g + ".part", g)
    return ok


def todo(ch): return not exists(f"{W}/rec/c{ch['k']}/.done") and not exists(f"{W}/rec/c{ch['k']}.tp4")


def lane(name, cards, opportunistic):
    cl = CardLocks(cards); tag = f"lane {name} TP2"; cs = ",".join(map(str, cards))
    try:
        if not opportunistic: cl.take(tag); log(f"{tag}: card locks {cards} held")
        while not ABORT.is_set():
            left = [ch for ch in chunks() if todo(ch)]
            if not left: break
            if opportunistic:
                if mem_avail_gb() < 25 or not cl.try_take(): time.sleep(60); continue
            got = None
            for ch in left:
                if claim(f"rec-c{ch['k']}"): got = ch; break
            if got is None:
                if opportunistic: cl.release()
                time.sleep(30)
                if not any(todo(ch) and not exists(f"{W}/claims/rec-c{ch['k']}") for ch in chunks()): break
                continue
            if not rec_chunk(cs, 2, got, tag): open(f"{W}/rec/c{got['k']}.tp4", "w").write("failed on TP2\n"); log(f"{tag}: chunk {got['k']} failed on TP2: queued for TP4")
            if opportunistic: cl.release()
    except Exception as e:
        log(f"{tag} FAILED: {type(e).__name__}: {e}")
        if not opportunistic: ABORT.set()                  # lane B is a bonus: its failure must not stop lane A
    finally: cl.release()


def tp4_phase():
    for ch in chunks():
        if exists(f"{W}/rec/c{ch['k']}.tp4") and not exists(f"{W}/rec/c{ch['k']}/.done"):
            cl = CardLocks([0, 1, 2, 3]); cl.take("TP4 fallback")
            try:
                if rec_chunk("0,1,2,3", 4, ch, "TP4 all four cards"): os.remove(f"{W}/rec/c{ch['k']}.tp4")
                else: raise RuntimeError(f"chunk {ch['k']} failed on TP4 too")
            finally: cl.release()


def sh(cmd, logfile):
    with open(logfile, "a") as f: return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode


def finish():
    if exists(f"{FINAL}/.done"): return
    os.makedirs(f"{QROOT}/ext/xg-fin", exist_ok=True); os.makedirs(f"{QROOT}/ext/genmask", exist_ok=True)
    corpus_out = f"{QROOT}/ext/xg-fin/xg-fin.jsonl"; mask = f"{QROOT}/ext/genmask/xg-fin.u8"
    rc = sh([PY, "-I", f"{C}/merge232.py", DOCS, f"{W}/rec", FINAL, corpus_out, mask, MODEL], f"{W}/logs/merge.out")
    log(f"merge232 rc {rc}: {open(f'{W}/logs/merge.out').read().strip().splitlines()[-1][:300]}")
    if rc: raise RuntimeError("merge failed")
    rc = sh([PY, "-I", f"{C}/check232.py", FINAL, mask, corpus_out, MODEL], f"{W}/logs/check.out")
    log(f"check232 rc {rc}: {open(f'{W}/logs/check.out').read().strip().splitlines()[-1][:600]}")
    if rc: raise RuntimeError("check failed")
    for ch in chunks(): shutil.rmtree(f"{W}/rec/c{ch['k']}", ignore_errors=True)      # the merged golden holds the data


def main():
    os.makedirs(f"{W}/logs", exist_ok=True); shutil.rmtree(f"{W}/claims", ignore_errors=True); os.makedirs(f"{W}/claims")
    for f in os.listdir(f"{W}/rec") if exists(f"{W}/rec") else []:
        if f.endswith(".part"): shutil.rmtree(f"{W}/rec/{f}", ignore_errors=True)
    log(f"chain232 start (pid {os.getpid()}); chunks {[(c['k'], c['docs'], c['tokens'], c['max_tokens']) for c in chunks()]}")
    if not exists(f"{FINAL}/.done"):
        ths = [threading.Thread(target=lane, args=("A", [0, 1], False)), threading.Thread(target=lane, args=("B", [2, 3], True))]
        for t in ths: t.start(); time.sleep(5)
        for t in ths: t.join()
        if ABORT.is_set(): log("chain232 aborted"); sys.exit(1)
        tp4_phase()
        finish()
    open(f"{W}/chain.done", "w").close(); log("chain232 done")


if __name__ == "__main__":
    try: main()
    except Exception as e: log(f"chain232 FAILED: {type(e).__name__}: {e}"); sys.exit(1)
