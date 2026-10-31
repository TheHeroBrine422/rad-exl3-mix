import json, time, urllib.request
def req(n, p="Write a long story about a dragon."):
    body = json.dumps({"model":"m","prompt":p,"max_tokens":n,"temperature":0,"ignore_eos":True}).encode()
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:4681/v1/completions", body, {"Content-Type":"application/json"}))); return time.time()-t, r["usage"]
t1,u1=req(1); t2,u2=req(257)
print("ttft %.3fs  decode %.1f tok/s (%d tokens)"%(t1,(u2["completion_tokens"]-1)/(t2-t1),u2["completion_tokens"]))
