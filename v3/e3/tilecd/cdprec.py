#!/usr/bin/env python3
"""cdprec.py SETTLE_DIR LAYER K SEED OUT.json -- why does the GPU's cd3 / LDLQ differ from settle.py's on 27B layer-3
attn_qg (w50, README.md)? settle.py's own proxy (P2 transforms, its g_scale, block LDLQ in float64), then the CD
sweeps of trellis2.ldlq_cd run twice from the same LDLQ codes: in float64 (settle.py's) and with H, E, G, the targets
and the acceptance test in float32 (as tilecd.py on the card). E3_F16=1 in the environment makes the Viterbi use fp16
costs (trellis.py's emulation of the GPU search) for LDLQ and CD alike. CDPREC_REV=1 reverses the input-channel
order before LDLQ and CD (exllamav3's ldlq() quantizes the last 16-column block first; trellis.ldlq_trellis the first)
and back for scoring (the metric is invariant under the permutation). Prints rel (after the per-output-channel
refit, settle.py's) and the ratio per stage."""
import json, os, sys, time
D, layer, K, seed, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
out = os.path.abspath(out); sys.path[:0] = [D, os.path.join(D, "..")]; os.chdir(D)   # chdir: settle.py's cache path
import numpy as np, torch
import settle, trellis, trellis2
torch.set_grad_enabled(False)

p = settle.P2(layer, "attn_qg", 128, seed, seed)
cb = np.ascontiguousarray(trellis2.codebook("mul1_i8"), dtype=np.float32)
s_exl, _ = settle.search_scale(p.T, cb, K)
Hq, dq = p.metric(0.005)
W = p.Wt * s_exl
REV = os.environ.get("CDPREC_REV") == "1"
if REV:
    ix = torch.arange(W.shape[1] - 1, -1, -1); W = W[:, ix].contiguous(); Hq = Hq[ix][:, ix].contiguous()
t0 = time.time(); Q0 = trellis.ldlq_trellis(W, Hq, cb, K, dq); tl = time.time() - t0
rel = lambda Q: settle.rels(p, (Q.flip(1) if REV else Q) / s_exl, 0.005)[0][0]
res = {"layer": layer, "K": K, "seed": seed, "f16": os.environ.get("E3_F16") == "1", "rev": REV, "s_exl": s_exl, "sec_ldlq": round(tl, 1),
       "rel_ldlq": rel(Q0)}


def cd(dt, sweeps=3):
    """trellis2.ldlq_cd's sweeps, in dtype dt, from Q0"""
    Q = Q0.clone().to(dt); R, Kd = W.shape
    Hd = Hq.clone(); dead = Hd.diagonal() == 0
    Hd[dead, dead] = 1; Wz = W.clone(); Wz[:, dead] = 0
    Hd.diagonal().add_(dq * Hd.diagonal().mean())
    Hd = Hd.to(dt); Wz = Wz.to(dt)
    E = Wz - Q; G = E @ Hd; r = {}
    for sw in range(1, sweeps + 1):
        acc = 0
        for a in range(0, Kd, 16):
            Haa = Hd[a:a + 16, a:a + 16]
            T = Q[:, a:a + 16] + torch.linalg.solve(Haa, G[:, a:a + 16].T).T
            qn = torch.from_numpy(trellis.vit(T.double().reshape(R // 16, 256).numpy(), cb, K).astype(np.float64)).reshape(R, 16).to(dt)
            Dl = Q[:, a:a + 16] - qn
            dt_ = (2 * (Dl * G[:, a:a + 16]).sum(1) + ((Dl @ Haa) * Dl).sum(1)).reshape(R // 16, 16).sum(1)
            keep = (dt_ < 0).repeat_interleave(16); acc += int((dt_ < 0).sum())
            if keep.any():
                Dk = Dl * keep[:, None]
                Q[:, a:a + 16] = torch.where(keep[:, None], qn, Q[:, a:a + 16])
                E[:, a:a + 16] += Dk; G += Dk @ Hd[a:a + 16, :]
        r[f"cd{sw}"] = {"rel": rel(Q.double()), "accepted": acc / (R // 16 * Kd // 16)}
    return r


for name, dt in [x for x in (("f64", torch.float64), ("f32", torch.float32)) if x[0] in os.environ.get("CDPREC_DTYPES", "f64,f32")]:
    t0 = time.time(); r = cd(dt)
    res[name] = {"stages": r, "ratio_cd3": r["cd3"]["rel"] / res["rel_ldlq"], "sec": round(time.time() - t0, 1)}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("f64", "f32")}))
