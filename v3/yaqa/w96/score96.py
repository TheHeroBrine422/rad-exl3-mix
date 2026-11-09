#!/usr/bin/env python3
"""score96.py DUMPDIR QDIR PERM.npy CKPT OUTDIR [ROWBLOCK] -- headsim94's exact head-only KL, on the CPU, decomposed per
128-row head block and donor so that any plan is scored afterwards without matmuls (plans96.py).

For every gen row r of the dump (headx94 layout: 336.x bf16, gen.u8) and donor K (QDIR/K<k>/wq/lm_head.safetensors,
f16 rows in sensitivity order), with p the bf16 head's softmax (perm order):
  d[r, k, b] = Σ_{v∈b} p_v (z_ref,v − z_k,v)        δ[r, k, b] = Σ_{v∈b} p_v expm1(z_k,v − z_ref,v)
  mx[r, k, b] = max_{v∈b} z_k,v                     zt[r, k] = z_k at the reference argmax
so for a plan a (block → donor) KL_r = Σ_b d[r, a_b, b] + log1p(Σ_b δ[r, a_b, b]) and top-1 holds iff zt[r, a_t] ≥
max_b mx[r, a_b, b] (t = the argmax's block). Same arithmetic as headsim94: reference = bf16 inputs × bf16 head, donors =
f16 inputs × f16 rows, products exact in f32, f32 accumulation. Also per row: lse, argmax (perm index), and the exact
Fisher Σ_r p(1−p) and Σ_r p per perm row (census). Row blocks are written as OUTDIR/blk-<i>.npz (restartable)."""
import glob, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

DUMP, QD, PERMF, CK, OUT = sys.argv[1:6]; RB = int(sys.argv[6]) if len(sys.argv) > 6 else 512
torch.set_num_threads(int(os.environ.get("T", 24)))
H, bf, N = open(f"{DUMP}/336.meta").read().split(); H, N = int(H), int(N)
gen = np.fromfile(f"{DUMP}/gen.u8", np.uint8).astype(bool); assert len(gen) == N
rows = np.nonzero(gen)[0]
X = np.memmap(f"{DUMP}/336.x", np.int16, "r", shape=(N, H))
KS = [float(x) for x in os.environ.get("KS", "3,4,5,6,7").split(",")]
os.makedirs(OUT, exist_ok=True)
json.dump({"dump": DUMP, "qdir": QD, "Ks": KS, "rows": rows.tolist(), "rb": RB}, open(f"{OUT}/meta.json", "w"))
nblk = (len(rows) + RB - 1) // RB
todo = [i for i in range(nblk) if not os.path.exists(f"{OUT}/blk-{i}.npz")]
if not todo: print("all blocks done"); sys.exit(0)
perm = torch.from_numpy(np.load(PERMF))
t0 = time.time()
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: Wref = f.get_tensor("lm_head.weight")[perm].float()
V = Wref.shape[0]; NB = V // 128
W = []
for k in KS:
    with safe_open(f"{QD}/K{k:g}/wq/lm_head.safetensors", "pt") as f: W.append(f.get_tensor("weight").float())
print(f"{len(rows)} gen rows, {nblk} blocks ({len(todo)} to do), donors {KS}, loaded in {time.time() - t0:.0f}s",
      file=sys.stderr, flush=True)
with torch.no_grad():
    for i in todo:
        r = rows[i * RB:(i + 1) * RB]; n = len(r)
        x = torch.from_numpy(np.ascontiguousarray(X[r])).view(torch.bfloat16)
        zr = x.float() @ Wref.T
        ls = torch.logsumexp(zr, -1); p = torch.exp(zr - ls[:, None]); tp = zr.argmax(-1)
        xh = x.to(torch.float16).float()
        d = np.zeros((n, len(KS), NB), np.float32); dl = np.zeros_like(d); mx = np.zeros_like(d)
        zt = np.zeros((n, len(KS)), np.float32)
        for j, w in enumerate(W):
            zk = xh @ w.T
            zt[:, j] = zk.gather(1, tp[:, None])[:, 0].numpy()
            mx[:, j] = zk.view(n, NB, 128).amax(-1).numpy()
            zk -= zr                                                   # zk := z_k − z_ref
            d[:, j] = (-(p * zk)).view(n, NB, 128).sum(-1).numpy()
            dl[:, j] = (p * torch.expm1(zk)).view(n, NB, 128).sum(-1).numpy()
            del zk
        fis = (p * (1 - p)).double().sum(0).numpy(); ps = p.double().sum(0).numpy()
        np.savez(f"{OUT}/blk-{i}.tmp.npz", d=d, dl=dl, mx=mx, zt=zt, lse=ls.numpy(), top=tp.numpy(), fisher=fis, psum=ps,
                 n1e4=(p > 1e-4).sum(0).numpy().astype(np.int32))
        os.replace(f"{OUT}/blk-{i}.tmp.npz", f"{OUT}/blk-{i}.npz")
        print(f"block {i + 1}/{nblk} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
print("done", file=sys.stderr)
