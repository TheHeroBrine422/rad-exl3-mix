import json, sys, time, urllib.request
out = sys.argv[1]
prompts = ["The history of the Roman Empire began", "def quicksort(arr):\n", "Explain the theory of general relativity in simple terms.", "Translate to French: The cat sat on the mat because it was warm."]
res = []
for p in prompts:
    body = json.dumps({"model":"m","prompt":p,"max_tokens":32,"temperature":0}).encode()
    t=time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:4681/v1/completions", body, {"Content-Type":"application/json"})))
    c = r["choices"][0]
    lp = c.get("logprobs") or {}
    res.append({"text": c["text"], "tokens": lp.get("tokens"), "lps": lp.get("token_logprobs"), "dt": time.time()-t})
    print(repr(c["text"][:120]), round(time.time()-t,2))
json.dump(res, open(out,"w"))
