#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fake_radiance230.py --config YAML -- the no-GPU stand-in for radiance in chaintest230.py: a config with `kld-record` writes a fake golden
(kld layout, random rows) for its kld-corpus and exits (rc 1 when tp 2 and max-model-len > FAKE_TP2_MAX, like a deep chunk that does not
fit two cards); any other config serves /health and a mock /completion on its port until SIGTERM."""
import http.server, json, os, re, sys, threading

import numpy as np

cfg = open(sys.argv[sys.argv.index("--config") + 1]).read()
g = lambda k: (re.search(rf"^\s*{k}:\s*(\S+)", cfg, re.M) or [None, None])[1]
if g("kld-record"):
    if g("tp") == "2" and int(g("max-model-len")) > int(os.environ.get("FAKE_TP2_MAX", 10**9)): sys.exit("fake: does not fit")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(os.environ["W230_MODEL"]); out = g("kld-record"); os.makedirs(out); K = 512; rng = np.random.default_rng(1)
    toks, docs, rows = [], [], 0
    for l in open(g("kld-corpus")):
        d = json.loads(l); ids = tk(d["prompt"], add_special_tokens=False)["input_ids"]; n = len(ids) - 1 - d["score_from"]
        docs.append({"source": d["source"], "tokens": len(ids), "score_from": d["score_from"], "row0": rows, "orig": len(docs)}); toks += ids; rows += n
    np.asarray(toks, np.int32).tofile(f"{out}/tokens.i32"); rng.random((rows, 3), dtype=np.float32).tofile(f"{out}/rows.f32")
    p = np.sort(rng.dirichlet(np.ones(K), rows), axis=1)[:, ::-1] * 0.9
    t = np.zeros(rows * K, np.dtype([("id", "<i4"), ("lp", "<f4")])); t["id"] = rng.integers(0, 248000, rows * K); t["lp"] = np.log(p).reshape(-1); t.tofile(f"{out}/top.bin")
    json.dump({"format": "radiance-kld-top-1", "model": "m", "corpus": g("kld-corpus"), "corpus_fnv1a64": "0", "created": "t", "n_vocab": 248320, "top": K, "rows": rows, "docs": docs}, open(f"{out}/kld.json", "w"))
    sys.exit(0)
from transformers import AutoTokenizer
GEN = AutoTokenizer.from_pretrained(os.environ["W230_MODEL"]).encode("ok.\n</think>\n\nDone.<|im_end|>", add_special_tokens=False)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self): self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        r = json.dumps({"tokens": GEN, "stop_type": "eos", "timings": {"prompt_n": len(b["prompt"]), "cache_n": 0, "prompt_ms": 1, "predicted_ms": 1}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(r))); self.end_headers(); self.wfile.write(r)
    def log_message(self, *a): pass


http.server.ThreadingHTTPServer(("127.0.0.1", int(g("port"))), H).serve_forever()
