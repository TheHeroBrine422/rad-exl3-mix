#!/usr/bin/env python3
"""gate87.py FORK_QUANTIZE_PY OUT.json [real] -- w87 gates G1 / G3 (../w87.md).

G1: settle87.ldlq2 + settle87.block_ldl against the fork's own ldlq_2hess + block_ldl (source extracted from the fork's
exllamav3/modules/quant/exl3_lib/quantize.py by ast; CUDA calls stubbed; the only text change: ldlq_2hess's work
buffer F in the weight's dtype instead of fp32, so both sides run fp64). Tile quantizer: a deterministic smooth stand-in
(q = x - 0.3 tanh(3x)), elementwise, so the tensor-core permutation is irrelevant. The fork runs on (k, n) = X^T.
Cases: random SPD H_in / H_out (two sizes); with "real", one real block (11:attn_k s0, H_A at sigma 0.005, the bn1920
H_out; needs the w77 environment). Pass: max |dQ| / max |Q| <= 1e-9 and the L factors agree to 1e-9.
G3: every (layer, tensor, sample) of the plan has an H_out with N entries, all finite and > 0 (with "real")."""
import ast, contextlib, json, math, os, sys, types
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _h)
torch.set_grad_enabled(False)


def fork_fns(path):
    src = open(path).read(); tree = ast.parse(src)
    want = {"block_ldl", "ldlq_2hess"}; code = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in want:
            s = ast.get_source_segment(src, n)
            if n.name == "ldlq_2hess":
                old = "F = torch.zeros((size_k, size_n), dtype = torch.float, device = device)"
                assert old in s, "ldlq_2hess's F line changed"
                s = s.replace(old, "F = torch.zeros((size_k, size_n), dtype = weight.dtype, device = device)")
            code.append(s)
    assert len(code) == 2, "fork functions not found"
    ns = {"torch": torch, "math": math, "ProgressBar": object, "save_failed_cholesky": lambda *a: None,
          "free_mem": lambda: None, "get_quant_stream": lambda d: None,
          "tensor_core_perm": lambda d: torch.arange(256), "tensor_core_perm_i": lambda d: torch.arange(256),
          "quantize_tiles_multigpu": lambda t, qa: (stand_in(t), torch.zeros(t.shape, dtype=torch.short))}
    exec("\n\n".join(code), ns)
    return ns["block_ldl"], ns["ldlq_2hess"]


def stand_in(t): return t - 0.3 * torch.tanh(3 * t)


torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()


def spd(n, g, cond=1e3):
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    Qm, _ = torch.linalg.qr(A); ev = torch.logspace(0, math.log10(cond), n, dtype=torch.float64)
    ev = ev[torch.randperm(n, generator=g)]
    return (Qm * ev) @ Qm.T


def case(name, X, Hin, Hout, fbl, f2h):
    import settle87
    Lin_m, Lout_m = settle87.block_ldl(Hin.clone()), settle87.block_ldl(Hout.clone())
    Lin_f = fbl(Hin.clone(), 16, {"sigma_reg": 0.025}, False)[0]
    Lout_f = fbl(Hout.clone(), 16, {"sigma_reg": 0.025}, False)[0]
    dL = max(float((Lin_m - Lin_f).abs().max()), float((Lout_m - Lout_f).abs().max()))
    Qm = settle87.ldlq2(X.clone(), Lin_m, Lout_m, lambda t: stand_in(torch.from_numpy(t)).numpy())
    Lk = Lin_f.clone(); Lk.diagonal().fill_(0.0); Ln = Lout_f.clone(); Ln.diagonal().fill_(0.0)   # as the fork passes them
    Qf, _ = f2h(X.T.contiguous(), Lk, Ln, {"devices": ["cpu"]}, None)
    dQ = float((Qm - Qf.T).abs().max()) / float(Qf.abs().max())
    # the objective the loop minimises vs plain one-sided and no feedback (sanity: two-sided feedback helps on it)
    E = X - Qm; obj = float(torch.einsum("ij,jk,lk,il->", E, Hin, E, Hout))
    Q1 = settle87.ldlq2(X.clone(), Lin_m, torch.eye(X.shape[0], dtype=X.dtype), lambda t: stand_in(torch.from_numpy(t)).numpy())
    E1 = X - Q1; obj1 = float(torch.einsum("ij,jk,lk,il->", E1, Hin, E1, Hout))
    o = {"case": name, "shape": list(X.shape), "max_dL": dL, "max_rel_dQ": dQ, "obj_two_sided": obj, "obj_one_sided_loop": obj1,
         "PASS": dQ <= 1e-9 and dL <= 1e-9}
    print(json.dumps(o), flush=True); return o


def main():
    fork, out = sys.argv[1], sys.argv[2]; real = "real" in sys.argv[3:]
    fbl, f2h = fork_fns(fork)
    g = torch.Generator().manual_seed(87); res = []
    for R, Kd in ((128, 256), (128, 512)):
        X = torch.randn(R, Kd, generator=g, dtype=torch.float64) * 1.5
        res.append(case(f"random {R}x{Kd}", X, spd(Kd, g), spd(R, g, 30.0), fbl, f2h))
    g3 = None
    if real:
        import settle87, settle77, e3
        p = settle87.prep_nohess(11, "attn_k", 0); p.H = settle77.load_h(os.environ["W77_HA"], 11, "input")
        Hq, _ = p.metric(0.005)
        hd = settle87.settle.had128(); Lout, hw, info = settle87.hout_metric(11, "attn_k", p, 0.01)
        # rebuild the rotated H_out from the same pieces (block_ldl consumed it): P diag(h r^2) P / mean + 0.01
        key = info["hout_key"]; from safetensors import safe_open
        with safe_open(os.path.join(os.environ["W87_HOUT"], key + ".safetensors"), "pt") as f: h = f.get_tensor("hout_diag").double()
        r = settle87.row_rms(11, key); idx = (torch.tensor(p.extra["blocks"])[:, None] * 128 + torch.arange(128)).reshape(-1)
        Ho = (hd * (h[idx] * r[idx] ** 2)[None, :]) @ hd / float((h * r ** 2).mean()); Ho.diagonal().add_(0.01)
        Xr = p.Wt.double() * 1.3                       # a scale of the order the quantizer uses
        res.append(case("real 11:attn_k s0 (H_A d0.005, bn1920 H_out)", Xr[:, :1024].contiguous(), Hq[:1024, :1024].contiguous(), Ho, fbl, f2h))
        # G3: H_out shapes for the plan's tensors
        sys.path.insert(0, _h); import plan87
        bad = []
        for t, l, s in plan87.pairs():
            names = e3.ROUTE[t][0]; name = names[s % len(names)] if len(names) > 1 else names[0]
            key = f"{e3.P}{l}.{name}"
            with safe_open(os.path.join(os.environ["W87_HOUT"], key + ".safetensors"), "pt") as f: h = f.get_tensor("hout_diag")
            idxj = json.load(open(f"{e3.CK}/model.safetensors.index.json"))["weight_map"]
            with safe_open(f"{e3.CK}/{idxj[key + '.weight']}", "pt") as f: N = f.get_slice(key + ".weight").get_shape()[0]
            if h.numel() != N or not bool(torch.isfinite(h).all()) or float(h.min()) <= 0: bad.append(key)
        g3 = {"tensors": len(plan87.pairs()), "bad": bad, "PASS": not bad}
        print(json.dumps({"G3": g3}), flush=True)
    o = {"G1": res, "G1_PASS": all(x["PASS"] for x in res), "G3": g3}
    with open(out + ".part", "w") as f: json.dump(o, f, indent=1)
    os.replace(out + ".part", out)


if __name__ == "__main__":
    main()
