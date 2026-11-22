import sys, json, numpy as np, torch
sys.path.insert(0, "/data/w218/code/pipeline")
from rxq import deq
torch.set_num_threads(8); torch.manual_seed(0)
R = "/data/w218/runs/r0"; meta = json.load(open(f"{R}/g0/grad.json"))["tensors"]
lev = deq.Level("/data/w218/lv/2.50/trunk"); H = deq.hadamard128("cpu").float()
def hk(A): k, n = A.shape; return (H @ A.reshape(k // 128, 128, n)).reshape(k, n)
def hn(A): k, n = A.shape; return (A.reshape(k, n // 128, 128) @ H).reshape(k, n)
def ld(c, k): m = meta[k]; return torch.from_numpy(np.fromfile(f"{R}/g{c}/{k}.g", np.float32).reshape(m["k"], m["n"]))
def cos(a, b): return float((a * b).sum() / (a.norm() * b.norm()))
def sg(G, HW_k, HW_n): return (hk(G) * HW_k).sum(1), (HW_n * hn(G)).sum(0)
pick = [k for k in meta if any(s in k for s in ("40.mlp.down", "40.mlp.gate", "55.mlp.gate", "55.mlp.up", "23.self_attn.q", "40.self_attn.o", "8.linear_attn.in_proj_z", "8.mlp.down", "23.mlp.up"))]
print("cos between chunk 0 and chunk 1 scale gradients (log-mult): raw | denoised s(P_r G) with P_r from chunk 2 (r=4, 16) | null = same with a random rank-r subspace")
for key in pick:
    t = lev.raw(key); K = t["trellis"].shape[2] / 16
    W = deq.decode_hat(t["trellis"], K, lev.table).float(); HWk, HWn = hk(W), hn(W)
    G0, G1, G2 = ld(0, key), ld(1, key), ld(2, key)
    r0 = sg(G0, HWk, HWn); r1 = sg(G1, HWk, HWn)
    row = {"in": [cos(r0[0], r1[0])], "out": [cos(r0[1], r1[1])]}
    for r in (4, 16):
        U, S, V = torch.svd_lowrank(G2, q=r, niter=4)
        Ur, _ = torch.linalg.qr(torch.randn(U.shape)); Vr, _ = torch.linalg.qr(torch.randn(V.shape))
        for (A, B) in ((U, V), (Ur, Vr)):
            P = lambda g: A @ (A.T @ g @ B) @ B.T
            d0 = sg(P(G0), HWk, HWn); d1 = sg(P(G1), HWk, HWn)
            row["in"].append(cos(d0[0], d1[0])); row["out"].append(cos(d0[1], d1[1]))
    short = key.replace("model.language_model.layers.", "")
    for side in ("in", "out"):
        v = row[side]
        print(f"{short:30s} {side:3s} raw {v[0]:6.3f} | r4 {v[1]:6.3f} (null {v[2]:6.3f}) | r16 {v[3]:6.3f} (null {v[4]:6.3f})", flush=True)
