"""gen92.py TAG PORT OUT CORPUS -- e2e92's requests to a running radiance server: for TAG fs0 three greedy 300-token chat
generations (code, chat, a synthetic tool trace) -> OUT/gen-<i>.json; for every TAG the prefill speed: three ~8k-token
prompts cut from different xcorp documents of CORPUS (golden-search27c-x's text: no prefix-cache reuse), max_tokens 1,
wall per request -> OUT/speed-TAG.json (tok/s of runs 2-3; run 1 warms)."""
import json, sys, time, urllib.request
tag, port, O, corpus = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
U = f"http://127.0.0.1:{port}/v1"


def post(path, body):
    r = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(r, timeout=900) as f: out = json.load(f)
    return out, time.perf_counter() - t


if tag == "fs0":
    tools = [{"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output",
              "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
    convs = [
        [{"role": "user", "content": "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals. "
                                     "Give it a short docstring and two doctest examples."}],
        [{"role": "user", "content": "I'm visiting Kyoto for three days in November. Suggest an itinerary, one sentence per stop."}],
        [{"role": "system", "content": "You are a careful command-line agent. Use run_shell to inspect the machine before answering."},
         {"role": "user", "content": "Which file under /var/log is the largest, and what is in its last line?"},
         {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "run_shell", "arguments": json.dumps({"cmd": "du -a /var/log | sort -n -r | head -3"})}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "48212\t/var/log\n31877\t/var/log/journal\n9120\t/var/log/syslog.1"}],
    ]
    for i, msgs in enumerate(convs):
        body = {"model": "w92", "messages": msgs, "temperature": 0, "max_tokens": 300}
        if i == 2: body["tools"] = tools
        try: out, dt = post("/chat/completions", body)
        except Exception as e:                      # tool calling refused: the same trace as plain text
            if i != 2: raise
            body.pop("tools"); body["messages"] = msgs[:2] + [{"role": "assistant", "content": "run_shell: du -a /var/log | sort -n -r | head -3"},
                                                            {"role": "user", "content": "Output:\n" + msgs[3]["content"]}]
            out, dt = post("/chat/completions", body); out["fallback"] = repr(e)
        out["wall_s"] = dt
        json.dump(out, open(f"{O}/gen-{i}.json", "w"), indent=1)
        m = out["choices"][0]["message"]
        print(f"gen {i}: {out['usage']} {dt:.1f}s; reasoning {len(m.get('reasoning_content') or '')} chars; "
              f"content: {(m.get('content') or '')[:200]!r}; tool_calls: {m.get('tool_calls')}", flush=True)
docs = [json.loads(l)["prompt"] for l in open(corpus)]
docs = sorted(docs, key=len, reverse=True)
runs = []
for i in range(3):
    text = "".join(docs[i::3])[:30000]                 # ~8k tokens of different documents a run
    out, dt = post("/completions", {"model": "w92", "prompt": text, "max_tokens": 1, "temperature": 0})
    n = out["usage"]["prompt_tokens"]; runs.append({"prompt_tokens": n, "wall_s": round(dt, 3), "tok_s": round(n / dt, 1)})
    print(f"speed {tag} run {i}: {n} tokens {dt:.2f}s {n / dt:.0f} tok/s", flush=True)
json.dump({"tag": tag, "runs": runs, "tok_s_runs23": round(sum(r["tok_s"] for r in runs[1:]) / 2, 1)}, open(f"{O}/speed-{tag}.json", "w"))
