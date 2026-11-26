import json, numpy as np
Q = "/data/q27/queue"; g = "golden-jlong"
m = json.load(open(f"{Q}/slices/{g}/kld.json")); gen = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
def per(nm):
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); out = {}
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); x = K[sl][gen[sl]]
        out.setdefault(d["source"], []).append(x)
    return out
base = per("JL-I25-step-10M-ckpt-1e+07")
for s in base:
    if s.startswith("agent-trace"): print(s, "docs", len(base[s]), "rows", sum(len(x) for x in base[s]), "10M KL %.5f" % np.concatenate(base[s]).mean())
for a in ["C228-ctl", "C229-xgc", "C228-opmix", "C230-p701515", "C230-p502525", "C230-p403030"]:
    p = per(f"JL-{a}-ckpt-1e+07"); line = []
    for s in sorted(base):
        if not s.startswith("agent-trace"): continue
        b = np.concatenate(base[s]); x = np.concatenate(p[s]); n = len(b); blk = 2048; nb = n // blk
        rb = np.random.default_rng(0); bs = []
        for _ in range(1000):
            ii = rb.integers(0, nb, nb); sel = np.concatenate([np.arange(i*blk, (i+1)*blk) for i in ii]); bs.append(x[sel].sum() / b[sel].sum())
        line.append(f"{s.split('/')[1]} {x.sum()/b.sum():.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    bo = np.concatenate([np.concatenate(base[s]) for s in base if not s.startswith("agent")]); xo = np.concatenate([np.concatenate(p[s]) for s in base if not s.startswith("agent")])
    print(f"{a:11s} " + " | ".join(line) + f" | other jlong docs {xo.sum()/bo.sum():.3f}")
