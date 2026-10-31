#!/bin/bash
# e2e.sh MODEL.rad TAG -- serve MODEL (bf16 KV): the embedding kernel the engine chose, greedy text on 4 prompts,
# an 8192-token prefill (one step) and a 256-token decode; then the 100k KLD. One JSON line to /root/e2e.jsonl,
# the texts in /root/e2e-TAG.txt.
exec 9>/root/card0.lock; flock 9   # card 0 (the fast link) runs one model at a time
M=$1 T=$2
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance ROCR_VISIBLE_DEVICES=0
radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --host-headroom-mib 512 --max-num-seqs 1 \
  --max-num-batched-tokens 8192 --port 4681 > /root/e2e-$T-srv.log 2>&1 &
SRV=$!
for i in $(seq 1 150); do grep -q "listening on" /root/e2e-$T-srv.log && break; kill -0 $SRV 2>/dev/null || break; sleep 2; done
python3 - $T <<"PY" > /root/e2e-$T.json
import json, sys, time, urllib.request
def req(prompt, n):
    body = json.dumps({"model":"m","prompt":prompt,"max_tokens":n,"temperature":0,"ignore_eos":True}).encode()
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:4681/v1/completions", body, {"Content-Type":"application/json"}), timeout=1800)); return time.time()-t, r
out = {"tag": sys.argv[1]}
try:
    req("warm up the kernels", 4)
    texts = []
    for p in ["The history of the Roman Empire began", "def quicksort(arr):\n", "Explain the theory of general relativity in simple terms.", "Translate to French: The cat sat on the mat because it was warm."]:
        texts.append(p + " ||| " + req(p, 48)[1]["choices"][0]["text"])
    open("/root/e2e-%s.txt" % sys.argv[1], "w").write("\n=====\n".join(texts))
    doc = json.loads(open("/root/corpus8k.jsonl").readline())["prompt"]
    t1, r1 = req(doc, 1)
    t2, _ = req("Write a long story about a dragon.", 257); t3, _ = req("Write a long story about a dragon!", 1)
    out.update({"prompt_tok": r1["usage"]["prompt_tokens"], "prefill_tps": round(r1["usage"]["prompt_tokens"]/t1, 1), "decode_tps": round(256/(t2-t3), 1)})
except Exception as e:
    out["serve_fail"] = repr(e)
print(json.dumps(out))
PY
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
python3 - $T $M <<"PY" >> /root/e2e.jsonl
import json, re, subprocess, sys
t, m = sys.argv[1], sys.argv[2]
d = json.loads(open("/root/e2e-%s.json" % t).read() or "{}") or {"tag": t, "serve_fail": "no output"}
log = open("/root/e2e-%s-srv.log" % t).read()
d["embed_kernels"] = sorted(set(re.findall(r"(embed_tq -> \S+|embed_lookup -> \S+)", log)))
d["errors"] = len(re.findall(r"^E ", log, re.M))
w = subprocess.run(["bash", "-c", "cd /root; CARD=0 RADIANCE_KLD_MAXTOK=4096 ./gold.sh %s golden100k corpus100k.jsonl /root/e2e-k-%s bf16 8192 --max-num-batched-tokens 4096" % (m, t)], capture_output=True, text=True).stdout.strip()
d["kld_run"] = w
try:
    r = json.load(open("/root/e2e-k-%s.json" % t))
    d.update({"kld": r["kld"]["mean"], "p99": r["kld"]["p99"], "p99_9": r["kld"]["p99_9"], "top1": r["top1_agreement"], "ppl_ratio": r["ppl"]["ratio"]})
except Exception as e:
    d["kld_fail"] = repr(e)
subprocess.run(["rm", "-rf", "/root/e2e-k-%s" % t])
print(json.dumps(d))
PY
