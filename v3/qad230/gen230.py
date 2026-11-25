#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gen230.py RENDER OUT --ports P[,P..] [--slots 6] [--target 0] -- w230 step 3: generate the assistant turns at cuts230.py's cuts
(RENDER/cuts.jsonl, RENDER/sess/<sidx>.npz) on radiance servers (IQ6R-V4, TP2, i8 KV, prefix cache, DFlash2), the sampling of
Qwen's thinking mode (temperature 1.0, top_p 0.95, top_k 20, min_p 0; presence 0 / repetition 1 are the server defaults).
Derived from gen228.py; the differences:
  * the generation header comes from the cut (`gp` token ids: thinking-on `<|im_start|>assistant\\n<think>\\n` or the thinking-off form),
    and n_predict from the cut's `max_new` (32,768; 16,384 for thinking-off);
  * a session's cuts (cuts of one `group`, in depth order) run on one server; sessions are CLAIMED (mkdir OUT/claims/g<group>), so several
    gen230 processes (one a lane / server pair, started and stopped at different times) share one cut list;
  * --target T (0 = every cut): no new session starts once the finished generations of ALL processes hold T tokens (OUT/gen.tok, one line
    a finished cut); sessions under way finish their cuts. When nothing is left to claim the process writes OUT/closed and waits for its threads;
  * restartable: a cut whose OUT/gen/<cut>.json exists is skipped; chain230.py clears OUT/claims and OUT/running-* before a restart.
Output OUT/gen/<cut>.json {cut, sidx, p, prompt_len, gen, stop_type, timings, wall, port} (.part, renamed); OUT/gen.log: a progress line a minute."""
import argparse, collections, json, os, threading, time, urllib.request

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("render"); ap.add_argument("out")
ap.add_argument("--ports", required=True); ap.add_argument("--slots", type=int, default=6)
ap.add_argument("--target", type=int, default=0)
a = ap.parse_args()
os.makedirs(f"{a.out}/gen", exist_ok=True); os.makedirs(f"{a.out}/claims", exist_ok=True)
LOG = f"{a.out}/gen.log"; TOK = f"{a.out}/gen.tok"


def log(s):
    with open(LOG, "a") as f: f.write(f"{time.strftime('%m-%d %I:%M:%S %p', time.localtime())} {s}\n")


def total_tokens():
    try: return sum(int(l.split()[1]) for l in open(TOK) if l.strip())
    except FileNotFoundError: return 0


cuts = [json.loads(l) for l in open(f"{a.render}/cuts.jsonl")]
by_s = collections.OrderedDict()
for c in cuts: by_s.setdefault(c["group"], []).append(c)
sessions = [(g, sorted(cs, key=lambda c: c["p"])) for g, cs in by_s.items()]
lock = threading.Lock(); nxt = [0]; st = collections.Counter()
if not os.path.exists(TOK):                                       # first start: seed gen.tok from finished cut files (a restart of an old run)
    with open(TOK, "w") as f:
        for fn in os.listdir(f"{a.out}/gen"):
            if fn.endswith(".json"): f.write(f"{fn[:-5]} {len(json.load(open(f'{a.out}/gen/{fn}'))['gen'])}\n")


def post(port, body, timeout=14400):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read())


def claim_next():
    """the next unclaimed session whose cuts are not all done; None = nothing left (or the target is met)"""
    while True:
        with lock:
            if nxt[0] >= len(sessions): return None
            g, cs = sessions[nxt[0]]; nxt[0] += 1
        todo = [c for c in cs if not os.path.exists(f"{a.out}/gen/{c['cut']}.json")]
        if not todo: continue
        if a.target and total_tokens() >= a.target: return None
        try: os.mkdir(f"{a.out}/claims/g{g}")
        except FileExistsError: continue
        return todo


def worker(port):
    last = (None, None)
    while True:
        todo = claim_next()
        if todo is None: return
        with lock: st["sessions"] += 1
        grp = todo[0]["group"]
        for c in todo:
            if last[0] != c["sidx"]: last = (c["sidx"], np.load(f"{a.render}/sess/{c['sidx']}.npz")["tokens"])
            prompt = last[1][:c["p"]].tolist() + c["gp"]
            body = {"prompt": prompt, "n_predict": c["max_new"], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "return_tokens": True, "cache_prompt": True}
            r = None
            for attempt in range(3):
                t0 = time.time()
                try: r = post(port, body); break
                except Exception as e:
                    log(f"port {port} cut {c['cut']} attempt {attempt} failed: {type(e).__name__} {str(e)[:200]}"); r = None; time.sleep(30)
            if r is None:
                with lock: st["failed"] += 1
                continue
            d = {"cut": c["cut"], "sidx": c["sidx"], "p": c["p"], "prompt_len": len(prompt), "gen": r.get("tokens", []), "stop_type": r.get("stop_type"),
                 "timings": r.get("timings", {}), "wall": time.time() - t0, "port": port}
            fn = f"{a.out}/gen/{c['cut']}.json"
            with open(fn + ".part", "w") as f: json.dump(d, f)
            os.replace(fn + ".part", fn)
            with open(TOK, "a") as f: f.write(f"{c['cut']} {len(d['gen'])}\n")
            tm = d["timings"]
            with lock:
                st["cuts"] += 1; st["new_gen"] += len(d["gen"]); st[f"stop_{d['stop_type']}"] += 1
                st["prompt_n"] += tm.get("prompt_n", 0); st["cache_n"] += tm.get("cache_n", 0)
                st["prompt_ms"] += tm.get("prompt_ms", 0); st["pred_ms"] += tm.get("predicted_ms", 0)
                st["draft_n"] += tm.get("draft_n", 0); st["draft_acc"] += tm.get("draft_n_accepted", 0)
        if any(not os.path.exists(f"{a.out}/gen/{c['cut']}.json") for c in todo):      # a failed request: release the claim for a retry
            try: os.rmdir(f"{a.out}/claims/g{grp}")
            except OSError: pass


def monitor(t0):
    while not done.is_set():
        done.wait(60)
        with lock:
            el = time.time() - t0
            log(f"cuts {st['cuts']} sessions {st['sessions']} new {st['new_gen']} ({st['new_gen'] / el:.0f} tok/s this process) all-process total {total_tokens()} "
                f"prompt_n {st['prompt_n']} cache_n {st['cache_n']} per-req prefill {st['prompt_n'] / max(st['prompt_ms'], 1) * 1e3:.0f} tok/s "
                f"decode {st['new_gen'] / max(st['pred_ms'], 1) * 1e3:.1f} tok/s/seq draft acc {st['draft_acc'] / max(st['draft_n'], 1):.2f} "
                f"stops {dict((k, v) for k, v in st.items() if k.startswith('stop_'))} failed {st['failed']}")


done = threading.Event(); t0 = time.time()
log(f"start: {len(cuts)} cuts in {len(sessions)} sessions, {len(os.listdir(a.out + '/gen'))} files, tokens so far {total_tokens()}, target {a.target}, ports {a.ports} x {a.slots} slots")
threading.Thread(target=monitor, args=(t0,), daemon=True).start()
th = [threading.Thread(target=worker, args=(int(p),)) for p in a.ports.split(",") for _ in range(a.slots)]
for t in th: t.start(); time.sleep(2)
for t in th: t.join()
done.set()
if not st["failed"] and (nxt[0] >= len(sessions) or (a.target and total_tokens() >= a.target)): open(f"{a.out}/closed", "w").close()
el = time.time() - t0
log(f"end: {st['cuts']} cuts, {st['new_gen']} new tokens in {el / 3600:.2f} h ({st['new_gen'] / max(el, 1):.0f} tok/s), all-process total {total_tokens()}, failed {st['failed']}")
raise SystemExit(1 if st["failed"] else 0)
