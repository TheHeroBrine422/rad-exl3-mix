#!/usr/bin/env python3
"""pq.py PORT PROMPT N -- N prefill-only chat requests (max_tokens 1, greedy, a nonce line each: no prefix-cache hit),
one at a time; prints prompt tokens and wall seconds per request. #175 (w137), for --profile-ops servers."""
import json, sys, time, uuid, urllib.request
port, text, n = int(sys.argv[1]), open(sys.argv[2]).read(), int(sys.argv[3])
for i in range(n):
    body = {"model": "q", "messages": [{"role": "user", "content": f"Request {uuid.uuid4().hex}.\n" + text}],
            "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); j = json.load(urllib.request.urlopen(req, timeout=1200)); dt = time.perf_counter() - t0
    print(f"req {i}: prompt {j['usage']['prompt_tokens']} tokens, {dt:.3f} s", flush=True)
