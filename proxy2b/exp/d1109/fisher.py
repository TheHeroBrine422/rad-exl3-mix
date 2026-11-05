#!/usr/bin/env python3
"""fisher.py MODEL OUT NWIN [--seqlen 2048] [--seed 0] [--no-genmask] [--fp32] -- REAL-Q stage 0 (arXiv 2609.00049 §4.2,
App. D.1): the aggregated Fisher at every transformer block's output, F_l = mean_t g_t g_tᵀ (hidden x hidden, fp32),
g_t = dL/dy_t at block l's output (the residual stream after layer l), L = sum of the CE of labels SAMPLED from the
bf16 model's own distribution (true Fisher), one end-to-end backward per window.

Windows: the first NWIN of rqdata.windows (qad pool, collect.py's shuffle). Gen mask (default on, as c23 / YAQA_GENMASK):
only the scored positions' CE terms are in L (every position still runs and its label is still drawn); the mean runs
over every position of the windows used (a position's g carries its effect on the later scored ones). Windows without a
scored position are skipped (logged). --fp32: model and activations in fp32 (the bf16 run is the default; the d1109
smoke test compares the two).

OUT/fisher.pt: {"F": [layers, d, d] fp32 means, "n_tok", "windows", "args"}; OUT/state.pt every 64 windows (a rerun
continues from it; same args); fisher.pt is written .part then renamed, so its existence means done."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import rqdata

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("nwin", type=int)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0, help="window shuffle seed (rqdata) and label RNG")
ap.add_argument("--no-genmask", action="store_true")
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--chunk", type=int, default=512, help="positions per head chunk (logits never whole)")
ap.add_argument("--max-sec", type=float, default=0, help="stop cleanly (state saved) after this many seconds")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FIN = f"{a.out}/fisher.pt"
if os.path.exists(FIN): print("done already", FIN); sys.exit(0)
if os.environ.get("VRAM_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
from common import load_model

toks, gpos, wins = rqdata.windows(a.seqlen, a.seed)
assert a.nwin <= len(wins), (a.nwin, len(wins))
wins = wins[:a.nwin]
model, dec, head = load_model(a.model, dtype=torch.float32 if a.fp32 else torch.bfloat16)
model.requires_grad_(False)
layers = dec.layers; nl = len(layers); d = model.config.get_text_config().hidden_size
dev = torch.device("cuda")
args = {k: v for k, v in vars(a).items() if k not in ("out", "max_sec")}
ST = f"{a.out}/state.pt"
if os.path.exists(ST):
    s = torch.load(ST)
    assert s["args"] == args, f"{ST}: made with {s['args']}, this run {args}"
    Fsum, n_tok, done, skipped = s["Fsum"].to(dev), s["n_tok"], s["done"], s["skipped"]
    print(f"resume at window {done}", flush=True)
else:
    Fsum, n_tok, done, skipped = torch.zeros(nl, d, d, device=dev), 0, 0, []
gen = torch.Generator(device=dev)

G = {}
def hook(l):
    def fh(mod, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        if y.requires_grad:
            y.register_hook(lambda g, l=l: G.__setitem__(l, g.detach()))
    return fh
for l, lay in enumerate(layers): lay.register_forward_hook(hook(l))

def save_state():
    torch.save({"Fsum": Fsum.cpu(), "n_tok": n_tok, "done": done, "skipped": skipped, "args": args}, ST + ".part")
    os.replace(ST + ".part", ST)

t0 = time.time(); tw = time.time()
while done < len(wins):
    w0 = wins[done]
    ids = torch.from_numpy(toks[w0:w0 + a.seqlen].astype(np.int64))[None].to(dev)
    m = torch.ones(a.seqlen, device=dev) if a.no_genmask else torch.from_numpy(gpos[w0:w0 + a.seqlen]).to(dev)
    if m.sum() == 0:
        skipped.append(done); done += 1; continue
    gen.manual_seed(a.seed * 1_000_003 + done)
    emb = dec.embed_tokens(ids).detach().requires_grad_(True)
    h = dec(inputs_embeds=emb).last_hidden_state[0]                     # final-norm output, [T, d]
    hd = h.detach().requires_grad_(True)
    for c in range(0, a.seqlen, a.chunk):                                # CE of sampled labels, chunked; grad into hd
        lg = head(hd[c:c + a.chunk]).float()
        with torch.no_grad():
            y = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen)[:, 0]
        ce = (torch.nn.functional.cross_entropy(lg, y, reduction="none") * m[c:c + a.chunk]).sum()
        ce.backward(); del lg, ce
    G.clear()
    h.backward(hd.grad)
    assert len(G) == nl, (len(G), nl)
    for l in range(nl):
        g = G[l][0].float()
        Fsum[l].addmm_(g.T, g)
    n_tok += a.seqlen; done += 1; G.clear(); del h, hd, emb
    if done % 64 == 0 or done == len(wins):
        save_state()
        print(f"window {done}/{len(wins)} {time.time() - t0:.0f}s ({(time.time() - tw) / 64:.2f} s/win) "
              f"max mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True); tw = time.time()
    if a.max_sec and time.time() - t0 > a.max_sec and done < len(wins):
        save_state(); print(f"STOP at window {done} (max-sec)", flush=True); sys.exit(3)
F = (Fsum / n_tok).cpu()
torch.save({"F": F, "n_tok": n_tok, "windows": wins, "skipped": skipped, "args": args}, FIN + ".part")
os.replace(FIN + ".part", FIN)
tr = F.diagonal(dim1=1, dim2=2).sum(1)
print(json.dumps({"FISHER DONE": FIN, "windows": len(wins), "skipped": len(skipped), "n_tok": n_tok, "sec": round(time.time() - t0),
                  "trace_first_last": [float(tr[0]), float(tr[-1])], "max_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
