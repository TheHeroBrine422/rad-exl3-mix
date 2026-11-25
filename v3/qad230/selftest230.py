#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""selftest230.py CUTS_DIR TOKENIZER WORKDIR [--set t2-judge] [--n 40] -- no-GPU test of the w230 data path on the first N cuts of one cuts230.py set:
a mock /completion server (returns a fixed token list), gen230.py (claims, gen.tok, closed), docs230.py (2 chunks), fake chunk goldens in
radiance's kld layout, merge230.py, check230.py (the stored tokens / rows / gen mask against the corpus). Prints one line a step; exit 1 on a failure."""
import http.server, json, os, shutil, subprocess, sys, threading

import numpy as np
from transformers import AutoTokenizer

cuts_dir, tkp, work = sys.argv[1:4]
name = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "t2-judge"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
tk = AutoTokenizer.from_pretrained(tkp)
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
R = f"{work}/render"; os.makedirs(R)
cs = [json.loads(l) for l in open(f"{cuts_dir}/{name}/cuts.jsonl")][:N]
open(f"{R}/cuts.jsonl", "w").writelines(json.dumps(c) + "\n" for c in cs); os.symlink(os.path.abspath(f"{cuts_dir}/{name}/sess"), f"{R}/sess")
GEN = tk.encode("Let me think.\n</think>\n\nHello, world! Ünïcode ✓<|im_end|>", add_special_tokens=False)


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert b["temperature"] == 1.0 and b["top_k"] == 20 and b["top_p"] == 0.95 and b["min_p"] == 0.0 and b["n_predict"] in (32768, 16384)
        r = json.dumps({"tokens": GEN, "stop_type": "eos", "timings": {"prompt_n": len(b["prompt"]), "cache_n": 0, "prompt_ms": 1, "predicted_ms": 1}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(r))); self.end_headers(); self.wfile.write(r)
    def log_message(self, *a): pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]


def run(*cmd):
    r = subprocess.run([PY, "-I", *cmd], capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(f"FAILED: {cmd[0]}")
    return r.stdout


G = f"{work}/gen"
run(f"{HERE}/gen230.py", R, G, "--ports", str(port), "--slots", "3")
assert os.path.exists(f"{G}/closed") and len(os.listdir(f"{G}/gen")) == len(cs), "gen230: closed / cut files"
assert sum(int(l.split()[1]) for l in open(f"{G}/gen.tok")) == len(cs) * len(GEN)
print(f"gen230 ok: {len(cs)} cuts, {len(cs) * len(GEN)} tokens; claims {len(os.listdir(G + '/claims'))}")
run(f"{HERE}/gen230.py", R, G, "--ports", str(port), "--slots", "3")        # restart: nothing to do
assert len(open(f"{G}/gen.tok").read().split("\n")) == len(cs) + 1
D = f"{work}/docs"
out = run(f"{HERE}/docs230.py", name, R, G, D, "--chunks", "2", "--tok", tkp)
ch = json.load(open(f"{D}/{name}.chunks.json")); meta = [json.loads(l) for l in open(f"{D}/{name}.jsonl.meta.jsonl")]
print(f"docs230 ok: {len(meta)} docs in {len(ch)} chunks {[c['docs'] for c in ch]}, retok_same {sum(m['retok_same'] for m in meta)}, prompt_retok_same {sum(m['prompt_retok_same'] for m in meta)}")
assert all(m["gen_rows"] == len(GEN) or not m["retok_same"] for m in meta)
rng = np.random.default_rng(0); K = 512
for c in ch:                                                          # fake chunk goldens (radiance's layout)
    g = f"{work}/rec/{name}-c{c['k']}"; os.makedirs(g)
    lines = [json.loads(l) for l in open(f"{D}/{c['file']}")]
    toks, docs, rows = [], [], 0
    for l in lines:
        ids = tk(l["prompt"], add_special_tokens=False)["input_ids"]; n = len(ids) - 1 - l["score_from"]
        docs.append({"source": l["source"], "tokens": len(ids), "score_from": l["score_from"], "row0": rows, "orig": len(docs)}); toks += ids; rows += n
    np.asarray(toks, np.int32).tofile(f"{g}/tokens.i32")
    rng.random((rows, 3), dtype=np.float32).tofile(f"{g}/rows.f32")
    p = np.sort(rng.dirichlet(np.ones(K), rows), axis=1)[:, ::-1] * 0.9
    t = np.zeros(rows * K, np.dtype([("id", "<i4"), ("lp", "<f4")])); t["id"] = rng.integers(0, 248000, rows * K); t["lp"] = np.log(p).reshape(-1)
    t.tofile(f"{g}/top.bin")
    json.dump({"format": "radiance-kld-top-1", "model": "m", "corpus": f"{D}/{c['file']}", "corpus_fnv1a64": "0", "created": "t", "n_vocab": 248320, "top": K, "rows": rows, "docs": docs}, open(f"{g}/kld.json", "w"))
    open(f"{g}/.done", "w").close()
F = f"{work}/final"
print(run(f"{HERE}/merge230.py", name, D, f"{work}/rec", F, "--corpus-out", f"{work}/final.jsonl", "--mask-out", f"{work}/final.u8").strip())
print(run(f"{HERE}/check230.py", F, f"{work}/final.jsonl", tkp).strip())
k = json.load(open(f"{F}/kld.json")); assert k["corpus"] == f"{work}/final.jsonl" and [d["orig"] for d in k["docs"]] == list(range(len(meta)))
assert open(f"{work}/final.jsonl").read() == open(f"{D}/{name}.jsonl").read()
print("selftest230 OK")
