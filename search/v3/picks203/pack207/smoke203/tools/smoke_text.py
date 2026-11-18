#!/usr/bin/env python3
"""smoke_text.py PORT PROMPTS.jsonl OUT.jsonl MAXTOK -- greedy completions (one at a time, top-20 logprobs) with the
tokens per decode step of each request, from the server's counters (w203): decode_tokens_total / engine_steps_decode_total
deltas across the request. Record form as gen.py's ({"id", "text", "tokens", "top"}) plus "steps", "tok_per_step",
"draft_tokens", "draft_accepted" deltas. Output one line per prompt; a summary line on stdout."""
import json, re, sys, urllib.request

port, pf, out, mt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
BASE = f"http://127.0.0.1:{port}"
KEYS = ["radiance:decode_tokens_total", "radiance:engine_steps_decode_total",
        "radiance:draft_tokens_total", "radiance:draft_accepted_total"]


def counters():
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=60).read().decode()
    vals = {}
    for k in KEYS:
        m = re.findall(r"^" + re.escape(k) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", txt, re.M)
        vals[k] = sum(float(x) for x in m)
    return vals


tot_t = tot_s = 0.0
with open(out, "w") as f:
    for line in open(pf):
        p = json.loads(line)
        c0 = counters()
        body = json.dumps({"model": __import__("os").environ.get("SMOKE_MODEL", "q"), "prompt": p["prompt"], "max_tokens": mt, "temperature": 0, "logprobs": 20,
                           "ignore_eos": True}).encode()
        req = urllib.request.Request(f"{BASE}/v1/completions", data=body, headers={"Content-Type": "application/json"})
        try:
            j = json.load(urllib.request.urlopen(req, timeout=1800))
            c = j["choices"][0]
            lp = c.get("logprobs") or {}
            c1 = counters()
            dt = c1[KEYS[0]] - c0[KEYS[0]]
            ds = c1[KEYS[1]] - c0[KEYS[1]]
            rec = {"id": p["id"], "text": c["text"], "tokens": lp.get("tokens"), "top": lp.get("top_logprobs"),
                   "steps": ds, "tok_per_step": (dt / ds) if ds else None,
                   "draft_tokens": c1[KEYS[2]] - c0[KEYS[2]], "draft_accepted": c1[KEYS[3]] - c0[KEYS[3]]}
            tot_t += dt
            tot_s += ds
        except Exception as e:  # noqa: BLE001 -- the record says what failed
            rec = {"id": p["id"], "error": repr(e)[:300]}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(p["id"], "error" if "error" in rec else
              f"{len(rec['tokens'] or [])} tokens, tok/step {rec['tok_per_step'] and round(rec['tok_per_step'], 3)}, "
              f"draft {rec['draft_accepted']:.0f}/{rec['draft_tokens']:.0f}", flush=True)
print(f"SUMMARY decode tokens {tot_t:.0f} over decode steps {tot_s:.0f}: tok/step {tot_t / tot_s if tot_s else float('nan'):.3f}")
