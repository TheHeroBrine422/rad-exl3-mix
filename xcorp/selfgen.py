#!/usr/bin/env python3
"""selfgen.py TRACES.jsonl OUT.jsonl --url URL [options] -- self-generated continuations of agent sessions: at assistant
turn boundaries of real sessions (pisess.py's rendered traces), the context so far (system block + the most recent
turns, <= --ctx tokens) is sent to a running OpenAI-compatible server (radiance) as a raw /v1/completions prompt ending
in "<|im_start|>assistant\\n", and the model's own next turn (reasoning, text, tool calls) is stored. The text is then
scored offline by a bf16 --kld-record pass (teacher logits), or by a recording server build (qc-kld-serve).

Meant to run beside interactive use of the same server: requests carry a low `priority` (radiance serves higher
priority first and preempts lower), and --conc streams stay in flight (batched decode is where the throughput is).
Resumable: a boundary already in OUT is skipped; each result is one appended line. Private input: our machines only.

Options: --conc N (8) --ctx TOKENS (28672) --max-tokens N (4096) --priority P (-10) --per-session N (8 boundaries,
evenly spread) --limit N (boundaries in all) --model NAME --temperature T (server default if absent) --seed S (0).
Docs: docs/design/eval-data.md."""
import argparse, json, os, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("traces"); ap.add_argument("out"); ap.add_argument("--url", required=True)
ap.add_argument("--conc", type=int, default=8); ap.add_argument("--ctx", type=int, default=28672)
ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--priority", type=int, default=-10)
ap.add_argument("--per-session", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=None); ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokenizer", default=os.environ.get("XCORP_TOKENIZER", "/data/models/qwen38-27b-bf16"))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
IM, ASSIST = tok.convert_tokens_to_ids("<|im_start|>"), tok("assistant\n", add_special_tokens=False)["input_ids"]

def boundaries():
    """(key, prompt text) for each chosen assistant-turn start: system block + the latest whole turns that fit"""
    rng = random.Random(a.seed); out = []
    for i, line in enumerate(open(a.traces)):
        d = json.loads(line); ids = tok(d["prompt"], add_special_tokens=False)["input_ids"]
        st = [k for k, x in enumerate(ids) if x == IM]
        sys_end = next((s for s in st if ids[s + 1:s + 2] != tok("system", add_special_tokens=False)["input_ids"][:1]), None)
        if sys_end is None: continue
        asst = [s for s in st if s > sys_end and ids[s + 1:s + 1 + len(ASSIST)] == ASSIST]
        if not asst: continue
        pick = asst if len(asst) <= a.per_session else [asst[round(j * (len(asst) - 1) / (a.per_session - 1))] for j in range(a.per_session)] if a.per_session > 1 else [rng.choice(asst)]
        for s in pick:
            room = a.ctx - sys_end
            starts = [x for x in st if sys_end <= x <= s]              # turn starts before the boundary
            c0 = next((x for x in starts if s - x <= room), s)         # earliest turn start that still fits
            prompt = tok.decode(ids[:sys_end] + ids[c0:s] + [IM] + ASSIST)
            out.append((f"{d.get('session', i)}:{i}:{s}", prompt, s - c0 + sys_end))
    rng.shuffle(out)
    return out[:a.limit] if a.limit else out

done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
todo = [b for b in boundaries() if b[0] not in done]
print(f"selfgen: {len(todo)} boundaries to do ({len(done)} done)", flush=True)
lock = threading.Lock(); stats = {"n": 0, "tok": 0, "t0": time.time()}

def one(b):
    key, prompt, ptok = b
    body = {"prompt": prompt, "max_tokens": a.max_tokens, "priority": a.priority, "stop": ["<|im_end|>"]}
    if a.model: body["model"] = a.model
    if a.temperature is not None: body["temperature"] = a.temperature
    for attempt in range(5):
        try:
            req = urllib.request.Request(a.url.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            c = r["choices"][0]; n = r.get("usage", {}).get("completion_tokens", 0)
            rec = {"key": key, "prompt_tokens": ptok, "completion": c["text"], "finish": c.get("finish_reason"), "tokens": n}
            with lock:
                with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                stats["n"] += 1; stats["tok"] += n
                if stats["n"] % 20 == 0:
                    dt = time.time() - stats["t0"]
                    print(f"selfgen: {stats['n']} done, {stats['tok']} tokens, {stats['tok'] / dt:.1f} tok/s", flush=True)
            return
        except Exception as e:
            print(f"selfgen: {key} attempt {attempt}: {e}", file=sys.stderr, flush=True); time.sleep(30 * (attempt + 1))

with ThreadPoolExecutor(a.conc) as ex: list(ex.map(one, todo))
print(f"SELFGEN DONE {stats['n']} turns, {stats['tok']} tokens", flush=True)
