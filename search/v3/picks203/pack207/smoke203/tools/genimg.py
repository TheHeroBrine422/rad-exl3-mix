#!/usr/bin/env python3
"""genimg.py PORT PROMPTS.jsonl IMGDIR OUT.jsonl [MAXTOK] -- image chat prompts (one image + one question each) to a
running server, greedy, top-20 logprobs, one at a time: one line per prompt {"id", "text", "tokens", "top"} in gen.py's
form, for dcmp.py (two engines on the same image prompts). Thinking off, as ask.sh. (w124 #164 scope 3)"""
import base64, json, mimetypes, sys, urllib.request

port, pf, imgdir, out = sys.argv[1:5]
mt = int(sys.argv[5]) if len(sys.argv) > 5 else 96
U = f"http://127.0.0.1:{port}/v1/chat/completions"
with open(out, "w") as f:
    for p in map(json.loads, open(pf)):
        path = f"{imgdir}/{p['image']}"
        mime = mimetypes.guess_type(path)[0] or "image/png"
        url = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
        body = {"model": __import__("os").environ.get("SMOKE_MODEL", "q"), "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": p["text"]}]}],
                "max_tokens": mt, "temperature": 0, "logprobs": True, "top_logprobs": 20, "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            j = json.load(urllib.request.urlopen(req, timeout=1200))
            c = j["choices"][0]
            lc = (c.get("logprobs") or {}).get("content") or []
            rec = {"id": p["id"], "text": c["message"]["content"], "tokens": [e["token"] for e in lc],
                   "top": [{t["token"]: t["logprob"] for t in e.get("top_logprobs", [])} for e in lc],
                   "prompt_tokens": j.get("usage", {}).get("prompt_tokens")}
        except Exception as e:   # noqa: BLE001 -- the record says what failed
            rec = {"id": p["id"], "error": repr(e)[:300]}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(p["id"], "error" if "error" in rec else f"{len(rec['tokens'])} tokens, prompt {rec['prompt_tokens']}")
