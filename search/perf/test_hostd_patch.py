#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_hostd_patch.py PIPELINE_DIR -- offline check of hostd-evalperf.patch (w123, #161): rxrun.hostd.run_eval with
stub pack / mix / score commands on two small goldens in radiance's file layout (tokens.i32, gen.u8, top.bin, rows.f32,
kld.json). The stub scorer writes per-document rows that depend only on (mix, document tokens), as radiance's do
(max-num-seqs 1: every document is prefilled on its own). Checks, against the unpatched behaviour (eval.goldcat off):
  goldcat     one score run for both parts; per-part rows byte-identical; result kld / p99 / top1 / positions / docs equal
  cards       eval.cards: a unit takes the first free card (card 0 held by this test -> card 1)
  ram_mode    eval.ram_mode "shared": the mix runs while ram.lock is held shared elsewhere (a level session); the default
              ("exclusive") waits, as before; a RAM admission above MemAvailable holds the step back
PIPELINE_DIR: a pipeline/ tree with the patch applied (rxrun/goldcat.py present)."""
import fcntl, hashlib, json, os, shutil, sys, tempfile, threading, time

import numpy as np

PIPE = os.path.abspath(sys.argv[1]); sys.path.insert(0, PIPE)
from rxrun import hostd   # noqa: E402

W = tempfile.mkdtemp(prefix="w123-hostd-")
STUB = f"{W}/stub.py"
open(STUB, "w").write(r"""import hashlib, json, os, sys
import numpy as np
op = sys.argv[1]
if op == "pack": open(sys.argv[3], "w").write("pack of " + open(sys.argv[2]).read()[:200])
elif op == "mix": open(sys.argv[2], "w").write(open(sys.argv[3]).read() + open(sys.argv[4]).read())
else:
    mix, g, o, card = sys.argv[2:6]
    open(os.path.dirname(o) + "/cards.log", "a").write(f"{os.path.basename(o)} {card}\n")
    m = json.load(open(g + "/kld.json")); tok = np.fromfile(g + "/tokens.i32", np.int32)
    h = hashlib.sha256(open(mix, "rb").read()).digest()
    R = np.zeros((m["rows"], 4), np.float32); t0 = 0
    for d in m["docs"]:
        n = d["tokens"] - 1 - d["score_from"]; t = tok[t0:t0 + d["tokens"]]; t0 += d["tokens"]
        seed = int(hashlib.sha256(h + t.tobytes()).hexdigest()[:8], 16); rng = np.random.default_rng(seed)
        R[d["row0"]:d["row0"] + n, 0] = rng.gamma(0.5, 0.2, n); R[d["row0"]:d["row0"] + n, 3] = rng.random(n) < 0.9
    R.tofile(o + ".json.part.rows"); os.replace(o + ".json.part.rows", o + ".json.rows")
    sel = np.concatenate([np.arange(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in m["docs"]])
    json.dump({"all": {"positions": int(len(sel)), "top1_agreement": float(R[sel, 3].astype(np.float64).mean()),
                       "kld": {"mean": float(R[sel, 0].mean())}}}, open(o + ".json", "w"))
""")


def golden(path, nd, seed, top=4):
    rng = np.random.default_rng(seed); os.makedirs(path); docs, r, toks = [], 0, []
    for i in range(nd):
        T = int(rng.integers(50, 400)); sf = int(rng.integers(0, 20)) if i % 3 == 0 else 0
        toks.append(rng.integers(0, 1000, T).astype(np.int32)); docs.append({"source": "t", "tokens": T, "score_from": sf, "row0": r, "orig": i})
        r += T - 1 - sf
    np.concatenate(toks).tofile(f"{path}/tokens.i32")
    (rng.random(r) < 0.8).astype(np.uint8).tofile(f"{path}/gen.u8")
    rng.integers(0, 255, r * top * 8).astype(np.uint8).tofile(f"{path}/top.bin")
    rng.random(r * 3).astype(np.float32).tofile(f"{path}/rows.f32")
    json.dump({"format": "radiance-kld-top-1", "model": "t", "top": top, "n_vocab": 1000, "rows": r, "docs": docs}, open(f"{path}/kld.json", "w"))


golden(f"{W}/gx", 9, 1); golden(f"{W}/gc", 5, 2)
LV = "lvA"
os.makedirs(f"{W}/ladder/{LV}"); json.dump({"level": LV}, open(f"{W}/ladder/{LV}/DONE", "w"))
open(f"{W}/ladder/{LV}/final.recipe", "w").write("recipe A\n"); open(f"{W}/base.rad", "w").write("base\n")


class FakeHost:
    def __init__(self, root, E):
        self.root = root; self.name = "t"; self.rc = {"root": f"{W}/ladder"}
        self.hc = {"eval": E}

    def current(self, n): return n == LV


def E_(root, **kw):
    E = {"packs": f"{W}/packs", "mixes": f"{root}/mixes", "scores": f"{root}/scores", "aliases": {"GX": f"{W}/gx", "GC": f"{W}/gc"},
         "card": 7, "card_lock": f"{W}/card7.lock", "ram_lock": f"{W}/ram.lock", "conv_lock": f"{W}/conv.lock",
         "pack_cmd": [sys.executable, STUB, "pack", "{recipe}", "{out}"], "mix_cmd": [sys.executable, STUB, "mix", "{out}", "{base}", "{mixfile}"],
         "score_cmd": [sys.executable, STUB, "score", "{mix}", "{golden}", "{out}", "{card}"]}
    E.update(kw); return E


req = {"tag": "e1", "mix": {"base": f"{W}/base.rad", "trunk": LV}, "slices": [{"name": "s", "parts": {"x": "GX", "c": "GC"}}]}
ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


res = {}
for nm, kw in (("plain", {}), ("goldcat", {"goldcat": True}),
               ("cards", {"goldcat": True, "cards": [{"card": 0, "lock": f"{W}/card0.lock"}, {"card": 1, "lock": f"{W}/card1.lock"}]})):
    root = f"{W}/{nm}"; os.makedirs(root); u = f"{root}/unit"; os.makedirs(u)
    hold = None
    if nm == "cards":
        hold = os.open(f"{W}/card0.lock", os.O_RDWR | os.O_CREAT); fcntl.flock(hold, fcntl.LOCK_EX)
    res[nm] = hostd.run_eval(FakeHost(root, E_(root, **kw)), u, req)
    if hold is not None: os.close(hold)
    check(f"{nm} rc", res[nm]["rc"] == 0, res[nm].get("err", ""))
ref = res["plain"]["slices"]["s"]
for nm in ("goldcat", "cards"):
    r = res[nm]["slices"]["s"]
    same = all(r[k] == ref[k] for k in ("kld", "p99", "top1", "positions")) and all(r["parts"][p] == ref["parts"][p] for p in ("x", "c"))
    check(f"{nm} result == plain", same, f"kld {r['kld']!r} vs {ref['kld']!r}, top1 {r['top1']!r} vs {ref['top1']!r}")
    for p in ("x", "c"):
        a = open(f"{W}/plain/scores/e1-s-{p}.json.rows", "rb").read(); b = open(f"{W}/{nm}/scores/e1-s-{p}.json.rows", "rb").read()
        check(f"{nm} rows {p} byte-identical", a == b, f"{len(a)} bytes")
    runs = open(f"{W}/{nm}/scores/cards.log").read().split()
    check(f"{nm} one score run", len(runs) == 2, f"runs {runs}")
check("cards: card 0 held -> card 1", open(f"{W}/cards/scores/cards.log").read().split()[1] == "1")
check("plain: two runs on eval.card 7", open(f"{W}/plain/scores/cards.log").read().split()[1::2] == ["7", "7"])

# mix lock: ram.lock held shared by "a level session"; the mix must not wait for it (default), must wait with mix_ram_excl
for nm, kw, expect_wait in (("mixsh", {"ram_mode": "shared", "mix_ram_gib": 0.5, "pack_ram_gib": 0.5}, False), ("mixex", {}, True),
                            ("mixadm", {"ram_mode": "shared", "mix_ram_gib": 1e6, "pack_ram_gib": 0.5}, True)):
    root = f"{W}/{nm}"; os.makedirs(f"{root}/unit")
    fd = os.open(f"{W}/ram.lock", os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_SH)
    out = {}
    th = threading.Thread(target=lambda: out.update(hostd.run_eval(FakeHost(root, E_(root, **kw)), f"{root}/unit", req)), daemon=True)
    t0 = time.time(); th.start(); time.sleep(3)
    mixed = os.path.exists(f"{root}/mixes/e1.rad")
    os.close(fd)
    if nm == "mixadm":                       # held back by the admission (not by the lock): lower the bar, then it runs
        check(f"{nm}: RAM admission holds the mix back", not mixed, f"mix done: {mixed}")
        hostd.RamGate.DEFAULT_GIB["mix"] = 0.5; th.join(1); out_ = out
        # the held unit waits inside conv.lock (one pack / mix a host: the right order) and stays there (a daemon
        # thread); a second unit on its own conv.lock with the default mix bar (lowered to 0.5 GiB) runs through
        root2 = f"{W}/mixadm2"; os.makedirs(f"{root2}/unit")
        out_ = hostd.run_eval(FakeHost(root2, E_(root2, ram_mode="shared", pack_ram_gib=0.5, conv_lock=f"{W}/conv2.lock")), f"{root2}/unit", req)
        check("mixadm2: default mix admission (0.5 GiB) runs", out_.get("rc") == 0 and os.path.exists(f"{root2}/mixes/e1.rad"))
        continue
    th.join(60)
    check(f"{nm}: mix {'waits for' if expect_wait else 'runs beside'} a shared ram.lock holder", mixed != expect_wait and out.get("rc") == 0,
          f"mix done while held: {mixed}, rc {out.get('rc')}")
print("ALL PASS" if ok else "FAILED", W)
if ok: shutil.rmtree(W)
sys.exit(0 if ok else 1)
