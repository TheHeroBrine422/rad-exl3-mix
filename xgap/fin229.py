import json, numpy as np, collections, sys, os
Q = "/data/q27/queue"
jgm = [json.loads(l) for l in open(f"{Q}/slices/golden-jgap/corpus.meta.jsonl")]
kg = lambda i: jgm[i]["cell"] + (" (WC)" if jgm[i]["cell"] == "long-chat" and "wildchat" in jgm[i]["source"].lower() else "")
def per(g, nm, key):
    m = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(list); n = collections.defaultdict(list)
    for i, d in enumerate(m["docs"]):
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); x = K[sl][ge[sl]]; s[key(i)].append(x.sum()); n[key(i)].append(max(len(x), 1))
    return {c: (np.array(s[c]), np.array(n[c])) for c in s}
rng = np.random.default_rng(0)
def cmp(b, x):
    (sb, nb), (sx, _) = b, x; bs = []
    for _ in range(2000):
        j = rng.integers(0, len(sb), len(sb)); bs.append(sx[j].sum() / sb[j].sum())
    med = np.median((sx / nb) / np.maximum(sb / nb, 1e-12))
    return f"{sx.sum()/sb.sum():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}] med {med:.3f}"
def cat(D, cells):
    return (np.concatenate([D[c][0] for c in cells]), np.concatenate([D[c][1] for c in cells]))
arms = sys.argv[1:]
for J, g, key in [("JC", "golden-jc228", lambda i: "jc228"), ("JG", "golden-jgap", kg)]:
    for base in ["I25-step-10M-ckpt-1e+07", "C228-ctl-ckpt-1e+07"]:
        B = per(g, f"{J}-{base}", key)
        for a in arms:
            nm = f"{J}-{a}-ckpt-1e+07"
            if not os.path.exists(f"{Q}/scores/{nm}.json"): continue
            X = per(g, nm, key); out = []
            for c in sorted(B): out.append(f"{c}: {cmp(B[c], X[c])}")
            if J == "JG":
                nw = [c for c in B if c != "long-chat (WC)"]
                out.append(f"ALL w/o WildChat: {cmp(cat(B, nw), cat(X, nw))}"); out.append(f"ALL: {cmp(cat(B, list(B)), cat(X, list(B)))}")
            print(f"\n{J} {a} vs {base.split('-ckpt')[0]}:\n  " + "\n  ".join(out))
