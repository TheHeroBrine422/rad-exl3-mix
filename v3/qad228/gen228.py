#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gen228.py RENDER OUT [--ports 8020,8021] [--slots 6] [--target 2000000] [--max-new 4096] -- w228 step 3: generate the
on-policy assistant turns at cuts228.py's cuts (RENDER/cuts.jsonl, RENDER/sess/<sidx>.npz) on the radiance servers (IQ6R-V4, TP2,
i8 KV, prefix cache on, DFlash2 drafter).

Each server gets --slots worker threads; a thread takes the next session not yet started (cuts.jsonl order) and runs that session's
cuts in depth order on ITS server (so the deeper cut's prompt hits the shallower one's cached prefix). Request: llama-server's
native POST /completion with the prompt as token ids (the session's tokens before the cut + the template's generation prompt
`<|im_start|>assistant\\n<think>\\n`: exactly what the chat template renders for the conversation up to there, thinking on),
the model's recommended thinking-mode sampling (temperature 1.0, top_p 0.95, top_k 20, min_p 0: generation_config.json + README),
n_predict --max-new, return_tokens; it stops at the model's eos (<|im_end|>, included in the returned ids).
No new session starts once the finished generations hold --target tokens (sessions under way finish their cuts).
Output OUT/gen/<cut>.json {cut, sidx, p, prompt_len, gen, stop_type, timings, wall, port} (.part, renamed); a cut whose file exists
is skipped (restartable). OUT/gen.log: a progress line a minute (CT): cuts, generated tokens, decode / prefill tok/s, cache hits.
PRIVATE output (the user's sessions): stays on box2."""
import argparse, collections, json, os, queue, threading, time, urllib.request

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("render"); ap.add_argument("out")
ap.add_argument("--ports", default="8020,8021"); ap.add_argument("--slots", type=int, default=6)
ap.add_argument("--target", type=int, default=2_000_000); ap.add_argument("--max-new", type=int, default=4096)
ap.add_argument("--max-sessions", type=int, default=0, help="test runs: stop after this many sessions")
a = ap.parse_args()
GP = [248045, 74455, 198, 248068, 198]            # <|im_start|> assistant \n <think> \n  (the Qwen3.8 tokenizer's ids, checked 11-24)
os.makedirs(f"{a.out}/gen", exist_ok=True)
LOG = f"{a.out}/gen.log"


def log(s):
    t = time.strftime("%m-%d %I:%M:%S %p", time.localtime())
    with open(LOG, "a") as f: f.write(f"{t} {s}\n")


cuts = [json.loads(l) for l in open(f"{a.render}/cuts.jsonl")]
by_s = collections.OrderedDict()
for c in cuts: by_s.setdefault(c["sidx"], []).append(c)
sq = queue.Queue()
for s, cs in by_s.items(): sq.put((s, sorted(cs, key=lambda c: c["p"])))
lock = threading.Lock()
st = collections.Counter()
for f in os.listdir(f"{a.out}/gen"):
    if f.endswith(".json"):
        d = json.load(open(f"{a.out}/gen/{f}")); st["gen_tokens"] += len(d["gen"]); st["cuts_done_before"] += 1


def post(port, body, timeout=7200):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read())


def worker(port):
    while True:
        with lock:
            if st["gen_tokens"] >= a.target or (a.max_sessions and st["sessions"] >= a.max_sessions): return
            try: s, cs = sq.get_nowait()
            except queue.Empty: return
            todo = [c for c in cs if not os.path.exists(f"{a.out}/gen/{c['cut']}.json")]
            if not todo: continue
            st["sessions"] += 1
        toks = np.load(f"{a.render}/sess/{s}.npz")["tokens"]
        for c in todo:
            prompt = toks[:c["p"]].tolist() + GP
            body = {"prompt": prompt, "n_predict": a.max_new, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "return_tokens": True, "cache_prompt": True}
            for attempt in range(3):
                t0 = time.time()
                try: r = post(port, body); break
                except Exception as e:
                    log(f"port {port} cut {c['cut']} attempt {attempt} failed: {type(e).__name__} {str(e)[:200]}"); r = None; time.sleep(30)
            if r is None:
                with lock: st["failed"] += 1
                continue
            d = {"cut": c["cut"], "sidx": s, "p": c["p"], "prompt_len": len(prompt), "gen": r.get("tokens", []), "stop_type": r.get("stop_type"),
                 "timings": r.get("timings", {}), "wall": time.time() - t0, "port": port}
            fn = f"{a.out}/gen/{c['cut']}.json"
            with open(fn + ".part", "w") as f: json.dump(d, f)
            os.replace(fn + ".part", fn)
            tm = d["timings"]
            with lock:
                st["cuts"] += 1; st["gen_tokens"] += len(d["gen"]); st["new_gen"] += len(d["gen"]); st[f"stop_{d['stop_type']}"] += 1
                st["prompt_n"] += tm.get("prompt_n", 0); st["cache_n"] += tm.get("cache_n", 0)
                st["prompt_ms"] += tm.get("prompt_ms", 0); st["pred_ms"] += tm.get("predicted_ms", 0)
                st["draft_n"] += tm.get("draft_n", 0); st["draft_acc"] += tm.get("draft_n_accepted", 0)


def monitor(t0):
    while not done.is_set():
        done.wait(60)
        with lock:
            el = time.time() - t0
            log(f"cuts {st['cuts']} (+{st['cuts_done_before']} before) sessions {st['sessions']} gen {st['gen_tokens']} "
                f"(new {st['new_gen']}: {st['new_gen'] / el:.0f} tok/s overall) prompt_n {st['prompt_n']} cache_n {st['cache_n']} "
                f"per-req prefill {st['prompt_n'] / max(st['prompt_ms'], 1) * 1e3:.0f} tok/s decode {st['new_gen'] / max(st['pred_ms'], 1) * 1e3:.1f} tok/s/seq "
                f"draft acc {st['draft_acc'] / max(st['draft_n'], 1):.2f} stops {dict((k, v) for k, v in st.items() if k.startswith('stop_'))} failed {st['failed']}")


done = threading.Event()
t0 = time.time()
log(f"start: {len(cuts)} cuts in {len(by_s)} sessions, {st['cuts_done_before']} done before ({st['gen_tokens']} tokens), target {a.target}, "
    f"ports {a.ports} x {a.slots} slots")
m = threading.Thread(target=monitor, args=(t0,), daemon=True); m.start()
th = [threading.Thread(target=worker, args=(int(p),)) for p in a.ports.split(",") for _ in range(a.slots)]
for t in th: t.start(); time.sleep(2)
for t in th: t.join()
done.set()
el = time.time() - t0
log(f"end: {st['cuts']} cuts, {st['new_gen']} new tokens in {el / 3600:.2f} h ({st['new_gen'] / el:.0f} tok/s), total {st['gen_tokens']}, failed {st['failed']}")
