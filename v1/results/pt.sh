#!/bin/bash
# pt.sh BPW : 8k doc -- server prefill (one step) + decode, then KLD record; one JSON line to /root/pt8k.jsonl
B=$1; M=/root/models/q38-27b-exl3-${B}bpw.rad
export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance
pkill -x radiance; while pgrep -x radiance >/dev/null; do sleep 1; done
radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --host-headroom-mib 512 --max-num-seqs 1 \
  --max-num-batched-tokens ${MNBT:-8192} --port 4681 > /root/pt-$B-srv.log 2>&1 &
for i in $(seq 1 150); do grep -q "listening on" /root/pt-$B-srv.log && break; pgrep -x radiance >/dev/null || break; sleep 2; done
python3 - $B <<"PY" > /root/pt-$B.json
import json, sys, time, urllib.request
def req(prompt, n):
    body = json.dumps({"model":"m","prompt":prompt,"max_tokens":n,"temperature":0,"ignore_eos":True}).encode()
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:4681/v1/completions", body, {"Content-Type":"application/json"}), timeout=1800)); return time.time()-t, r["usage"]
req("warm up the kernels", 4)
doc = json.loads(open("/root/corpus8k.jsonl").readline())["prompt"]
t1, u1 = req(doc, 1)                       # prefill (+1 sampled token)
t2, u2 = req("Write a long story about a dragon.", 257); t3, _ = req("Write a long story about a dragon!", 1)
print(json.dumps({"bpw": sys.argv[1], "prompt_tok": u1["prompt_tokens"], "prefill_s": round(t1,3),
  "prefill_tps": round(u1["prompt_tokens"]/t1,1), "decode_tps": round(256/(t2-t3),1)}))
PY
pkill -x radiance; while pgrep -x radiance >/dev/null; do sleep 1; done
radiance --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --host-headroom-mib 512 --max-num-seqs 1 \
  --kld-record /root/rec8k --kld-corpus /root/corpus8k.jsonl > /root/pt-$B-rec.log 2>&1
python3 - $B <<"PY"
import json, subprocess, sys
srv = json.load(open("/root/pt-%s.json" % sys.argv[1])) if open("/root/pt-%s.json" % sys.argv[1]).read().strip() else {"bpw": sys.argv[1], "srv_fail": 1}
r = json.loads(subprocess.run(["python3","-I","/root/scorerec.py","/root/golden8k","/root/rec8k"],capture_output=True,text=True).stdout or "{}")
if r: srv.update({"kld8k": r["kld"]["mean"], "top1": r["top1_agreement"], "ppl_ratio": r["ppl"]["ratio"]})
else: srv["rec_fail"] = 1
srv["cache_hit"] = open("/root/pt-%s-srv.log" % sys.argv[1]).read().count("(cache 0)")
print(json.dumps(srv))
PY
rm -rf /root/rec8k
