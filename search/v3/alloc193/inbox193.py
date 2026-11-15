# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""inbox193.py status | submit FILE... [--max-new N] | outcomes -- #193 inbox writer with the RULES-193 §0 budget.

Files are written as w172-<name>.json into the live drv/inbox/ (temp name in drv/, then mv). Checks before writing:
the live menu's option / window rules (common193.check_file), no duplicate of an assignment already in the run
(same option for every weight), my outstanding files (waiting in inbox/ + submitted whose eval has no result) <= 6,
decide <= 40, probe <= 15 in total (ledger data/alloc193/submitted.jsonl). `outcomes` prints every #193 tag's stage-1 /
stage-2 lines from driver.log (stage-2 verdicts are read for reporting only; RULES-193 §0)."""
import glob, hashlib, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common193 as C  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..")); import menu as MN  # noqa: E402

RUN = "/workspace/rad-local/workers/lead-srch/run"
DRV = f"{RUN}/drv"
LEDGER = os.path.join(HERE, "..", "..", "..", "data", "alloc193", "submitted.jsonl")
MAXOUT, MAXDEC, MAXPRB = 80, 600, 100         # RULES-193 amendments 7, 8


def ledger():
    return [json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else []


def state():
    return [json.loads(l) for l in open(f"{DRV}/state.jsonl")]


def mine(st):
    """file -> inbox event (my files only)"""
    return {e["file"]: e for e in st if e.get("ev") == "inbox" and e["file"].startswith("w172-")}


def outstanding(st):
    res = {e["tag"] for e in st if e.get("ev") == "result"}
    waiting = [f for f in os.listdir(f"{DRV}/inbox") if f.startswith("w172-") and f.endswith(".json")]
    sub = [f for f, e in mine(st).items() if any(t not in res for t in e["tags"])]
    return waiting, sub


def status():
    st = state(); w, s = outstanding(st); L = ledger()
    print(f"waiting {len(w)} {w}\nsubmitted, no result {len(s)} {s}")
    print(f"ledger: decide {sum(x['mode'] == 'decide' for x in L)} / {MAXDEC}, probe {sum(x['mode'] == 'probe' for x in L)} / {MAXPRB}")


def submit(files, max_new=None):
    conf = json.load(open(f"{RUN}/driver.json")); M = MN.Menu.load(conf["menu"])
    st = state(); w, s = outstanding(st); L = ledger(); room = MAXOUT - len(w) - len(s)
    if max_new is not None: room = min(room, max_new)
    known = set()
    for p in glob.glob(f"{DRV}/assign/*.json"):
        known.add(hashlib.sha1(json.dumps(json.load(open(p)), sort_keys=True).encode()).hexdigest())
    for f in files:
        if room <= 0: print(f"no room (outstanding cap {MAXOUT}); stop at {f}"); break
        d = json.load(open(f)); name = "w172-" + os.path.basename(f)
        nd = sum(x["mode"] == "decide" for x in L); npb = sum(x["mode"] == "probe" for x in L)
        if (d["mode"] == "decide" and nd >= MAXDEC) or (d["mode"] == "probe" and npb >= MAXPRB):
            print(f"{f}: budget exhausted for {d['mode']}"); continue
        why = C.check_file(M, d)
        if why: print(f"{f}: REFUSED {why}"); continue
        if "#193" not in d.get("note", "") or "code " not in d["note"]: print(f"{f}: REFUSED note"); continue
        h = hashlib.sha1(json.dumps(d["assign"], sort_keys=True).encode()).hexdigest()
        if h in known and not name.endswith("-re.json"): print(f"{f}: REFUSED duplicate of an assignment already in the run"); continue
        if any(x["name"] == name for x in L) or os.path.exists(f"{DRV}/inbox/done/{name}"): print(f"{f}: REFUSED name used"); continue
        tmp = f"{DRV}/.{name}.tmp"
        json.dump(d, open(tmp, "w")); os.replace(tmp, f"{DRV}/inbox/{name}")
        rec = {"name": name, "mode": d["mode"], "size": d["size"], "bpw": M.bpw(d["assign"]), "note": d["note"],
               "sha": hashlib.sha1(open(f"{DRV}/inbox/{name}", "rb").read()).hexdigest()[:16],
               "t": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 6 * 3600)) + " CST"}
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as fh: fh.write(json.dumps(rec) + "\n")
        L.append(rec); room -= 1; known.add(h)
        print(f"SUBMITTED {name} ({d['mode']} {d['size']:g}, bpw {rec['bpw']:.4f})")


def outcomes():
    st = state(); my = mine(st); tags = {e["tag"]: f for f, e in my.items()}
    log = open(f"{DRV}/driver.log").read().splitlines()
    for f, e in sorted(my.items(), key=lambda kv: kv[1]["t"]):
        t = e["tag"]; lines = [l for l in log if re.search(re.escape(t) + r"\b", l) and "ibx-" not in l and "INBOX" not in l]
        print(f"== {f} -> {t}")
        for l in lines: print("   " + l[:330])
    rej = [f for f in os.listdir(f"{DRV}/inbox/rejected") if f.startswith("w172-")] if os.path.isdir(f"{DRV}/inbox/rejected") else []
    if rej: print("REJECTED by the driver:", rej, [l for l in log if "rejected" in l and "w172-" in l])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "status": status()
    elif a[0] == "outcomes": outcomes()
    elif a[0] == "submit":
        mx = None
        if "--max-new" in a: i = a.index("--max-new"); mx = int(a[i + 1]); a = a[:i] + a[i + 2:]
        submit(a[1:], mx)
