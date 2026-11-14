#!/usr/bin/env python3
"""gen182.py PORT ITEMS.jsonl IMGDIR OUT.jsonl [MAXTOK] [CONC] -- the model-written answers of the #182 set (w156): each item
as one user turn (its images first, in order, then the question) to a running radiance server's chat endpoint, greedy,
thinking off, at most MAXTOK (512) tokens. One line per item {id, text, finish, completion_tokens, prompt_tokens};
resumable (ids in OUT are skipped). The answers are only a token stream: the bf16 reference scores them."""
import base64, json, mimetypes, os, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, items, imgdir, out = sys.argv[1:5]
mt = int(sys.argv[5]) if len(sys.argv) > 5 else 512
conc = int(sys.argv[6]) if len(sys.argv) > 6 else 8
U = f"http://127.0.0.1:{port}/v1/chat/completions"
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
todo = [it for it in map(json.loads, open(items)) if it["id"] not in done]
lock = threading.Lock()
def one(it):
    content = []
    for n in it["images"]:
        mime = mimetypes.guess_type(n)[0] or "image/png"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(open(f"{imgdir}/{n}", "rb").read()).decode()}})
    content.append({"type": "text", "text": it["question"]})
    body = {"model": "q", "messages": [{"role": "user", "content": content}], "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(U, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        j = json.load(urllib.request.urlopen(req, timeout=1800)); c = j["choices"][0]
        rec = {"id": it["id"], "text": c["message"]["content"], "finish": c.get("finish_reason"),
               "completion_tokens": j["usage"]["completion_tokens"], "prompt_tokens": j["usage"]["prompt_tokens"]}
    except Exception as e:   # noqa: BLE001 -- recorded, retried on a rerun
        print(it["id"], "error", repr(e)[:300], flush=True); return
    with lock:
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(it["id"], rec["completion_tokens"], rec["prompt_tokens"], rec["finish"], flush=True)
with ThreadPoolExecutor(conc) as ex: list(ex.map(one, todo))
