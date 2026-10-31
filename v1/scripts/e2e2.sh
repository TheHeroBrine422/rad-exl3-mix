#!/bin/bash
# e2e2.sh MODEL.rad TAG -- serving smoke test of the installed build at TP1 (bf16 KV): greedy completions,
# chat completions (incl. a tool call), 4 concurrent requests, an ~8k-token prefill and a 256-token decode.
# Texts and timings to /root/e2e2-TAG.json; server log /root/e2e2-TAG-srv.log. Waits for card 0.
exec 9>/root/card0.lock; flock 9
M=$1 T=$2
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance ROCR_VISIBLE_DEVICES=0
echo "start $(date +%T)"
radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --host-headroom-mib 512 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --port 4681 > /root/e2e2-$T-srv.log 2>&1 &
SRV=$!
for i in $(seq 1 150); do grep -q "listening on" /root/e2e2-$T-srv.log && break; kill -0 $SRV 2>/dev/null || break; sleep 2; done
python3 - $T <<"PY"
import json, sys, time, urllib.request, concurrent.futures as cf
U = "http://127.0.0.1:4681/v1/"
def post(ep, body):
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(U + ep, json.dumps(body).encode(),
                  {"Content-Type": "application/json"}), timeout=1800))
    return time.time() - t, r
def comp(p, n, **kw): return post("completions", dict(model="m", prompt=p, max_tokens=n, temperature=0, **kw))
def chat(msgs, n, **kw): return post("chat/completions", dict(model="m", messages=msgs, max_tokens=n, temperature=0, **kw))
out = {"tag": sys.argv[1]}
def step(name, f):
    try: out[name] = f()
    except Exception as e: out[name] = {"fail": repr(e)}
    print(name, "done", flush=True)
step("warm", lambda: comp("warm up", 4)[1]["choices"][0]["text"])
step("completions", lambda: [p + comp(p, 64)[1]["choices"][0]["text"] for p in [
    "The history of the Roman Empire began", "def quicksort(arr):\n",
    "Translate to French: The cat sat on the mat because it was warm.\nFrench:"]])
step("chat", lambda: [chat([{"role": "user", "content": q}], 400)[1]["choices"][0]["message"] for q in [
    "Explain the theory of general relativity in simple terms, in three sentences.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "What is 17 * 23? Show your working briefly."]])
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
step("tool_call", lambda: chat([{"role": "user", "content": "What's the weather in Paris right now?"}], 400,
                               tools=tools)[1]["choices"][0])
def conc():
    ps = ["Once upon a time in a small village", "The three laws of thermodynamics are", "import numpy as np\n",
          "The capital of Japan is"]
    with cf.ThreadPoolExecutor(4) as ex:
        seq = [comp(p, 48)[1]["choices"][0]["text"] for p in ps]
        par = list(ex.map(lambda p: comp(p, 48)[1]["choices"][0]["text"], ps))
    return {"same_as_sequential": [a == b for a, b in zip(seq, par)], "texts": par}
step("concurrent", conc)
def speed():
    doc = json.loads(open("/root/corpus8k.jsonl").readline())["prompt"]
    t1, r1 = comp(doc, 1)
    t2, _ = comp("Write a long story about a dragon.", 257, ignore_eos=True)
    t3, _ = comp("Write a long story about a dragon!", 1)
    return {"prompt_tok": r1["usage"]["prompt_tokens"], "prefill_tps": round(r1["usage"]["prompt_tokens"] / t1, 1),
            "decode_tps": round(256 / (t2 - t3), 1)}
step("speed", speed)
json.dump(out, open("/root/e2e2-%s.json" % sys.argv[1], "w"), indent=1, ensure_ascii=False)
PY
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "server errors: $(grep -c '^E ' /root/e2e2-$T-srv.log); embed: $(grep -o 'embed_tq -> [^ ]*' /root/e2e2-$T-srv.log | sort -u | tr '\n' ' ')"
echo "end $(date +%T)"
