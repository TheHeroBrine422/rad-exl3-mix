# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Post-quantization writer passes, frozen codes (DESIGN.md S4-S5b). Order on a level / mix (settings post.*):

  swj  (S4)   gate / up output scales through the SwiGLU (rxq/swj.py, rxq/swjfit.py); every level (#104)
  asym (S5)   the 128 writers requantized against W F with metric G from sibling moments (#87 / #105); PER MIX only
              (#111): after the knapsack, on the mix's own siblings; off until #105 + the user's rollout pick
  c9   (S5b)  C9 alpha on the writers' rows (#112, adopted 11-07), folded into svh AFTER everything else, on the codes
              the mix ships. Plain writers: computed inside each writer's quant job (rxq.job, c9_tensor: seconds a
              writer), assembled to LEVEL/c9, folded with swj's scales in one apply (apply LEVEL SWJ,LEVEL/c9 DST);
              asym writers (per mix): c9_scales --dn on the final writers

c9_scales(): port of rad-exl3-mix v3/e3/settle/w75/c9e27.py (the 27B implementation, w75 #112-02; the 0.8B original is
v3/e3/settle/w64/c9w.py), the same arithmetic in the same order (tests/check_deq_c9.py: bit-identical scales):
  Q   the writer as the level holds it (rxq.deq: the plugin's exl3i arithmetic, f16 suh / svh)
  t   W (bf16 checkpoint), M = the calib H_in of the writer's group (raw sum / rows, f64); asym writers (dn = a
      Denoiser output dir): t = W F = (Ft W^T)^T and M = G (DN/hin, the same layout)  [the dn form follows c9w.py's
      definition in c9e27's arithmetic; no research tool runs that combination at 27B]
  e_n = (q_n - t_n) M (q_n - t_n)^T / t_n M t_n^T, clipped to [0, e_clip];  c_n = (1 - e_n)^(-alpha / 2)   (f64)
OUT/scales/<key>.safetensors: "c" (out,) f32, "r" (in,) ones (swjapply's layout: rxq.swj.apply_level folds them into
svh, codes untouched); OUT/c9.jsonl per writer; OUT/.done last. Families with alpha 0 get no file. Restartable.

plan_asym(): the per-mix asym pass as steps (a stub until #105: rxq has the pieces -- sibling moments (rxq.sibmom),
the denoiser (rxq.asym), writer requant jobs (rxq quant --hin --wpre), splice (rxq.pairs) -- not the mix driver).

python -m rxq.post c9 LEVEL OUT [--alpha attn_o=-1,gdn_out=3] [--model CKPT] [--hess DIR] [--dn DIR] [--config F]
python -m rxq.post apply LEVEL SCALES[,SCALES..] DST   (= rxq.swj apply: svh *= c, suh *= r; several disjoint dirs merged)
python -m rxq.post plan -K K [--config F]          the post passes a level of K gets under the settings"""
import argparse, json, os, struct, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import qwen35

WRITERS = {"attn_o": "self_attn.o_proj", "gdn_out": "linear_attn.out_proj", "down": "mlp.down_proj"}
C9FAM = {"attn_output": "attn_o", "ssm_out": "gdn_out", "ffn_down": "down"}        # rxq.qwen35 family -> C9 family


def writer_keys(alpha, n_layers=64):
    """[(key, family, Hessian group)] in c9e27's order (per layer: attn_o / gdn_out, then down)"""
    Lp = qwen35.L; keys = []
    for l in range(n_layers):
        full = qwen35.is_full_attention(l)
        if alpha.get("attn_o", 0) and full: keys.append((f"{Lp}.{l}.self_attn.o_proj", "attn_o", f"{Lp}.{l}.self_attn.o_proj.input"))
        if alpha.get("gdn_out", 0) and not full: keys.append((f"{Lp}.{l}.linear_attn.out_proj", "gdn_out", f"{Lp}.{l}.linear_attn.out_proj.input"))
        if alpha.get("down", 0): keys.append((f"{Lp}.{l}.mlp.down_proj", "down", f"{Lp}.{l}.mlp.down_proj.input"))
    return keys


def hessian(p, dev):
    """c9e27.py's: the challenge layout -> (H + triu(H, 1)^T) / rows, f64"""
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k, rows = int(meta["k"]), int(meta["rows"])
    with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper")
    H = torch.zeros(k, k, dtype=torch.float64, device=dev); iu = torch.triu_indices(k, k, device=dev)
    H[iu[0], iu[1]] = up.to(dev).double(); del up
    return (H + H.triu(1).T) / rows


def c9_rows(Q, W, M, al, e_clip=0.5):
    """C9 of one writer (f64, (out, in) Q / W, M the group metric): -> (c (out,) f64, c9.jsonl's statistics). c9e27.py's
    arithmetic, in its order"""
    E = Q - W
    e = ((E @ M) * E).sum(1) / ((W @ M) * W).sum(1).clamp(min=1e-300)
    ncl = int((e > e_clip).sum()); nneg = int((e < 0).sum()); e = e.clamp(min=0.0, max=e_clip)
    c = (1 - e).pow(-al / 2)
    q = torch.quantile(c.float().cpu(), torch.tensor([0.01, 0.5, 0.99])).tolist()
    return c, {"rows": int(c.numel()), "e_mean": float(e.mean()), "e_max": float(e.max()), "clipped": ncl, "neg": nneg,
               "c_mean": float(c.mean()), "c_sd": float(c.std()), "c_min": float(c.min()), "c_max": float(c.max()),
               "c_q01_q50_q99": q, "n_gt15": int((c > 1.5).sum()), "n_gt2": int((c > 2).sum())}


def c9_file(c, n_in):
    """the scales file's tensors (swjapply's layout): svh *= c, suh *= r = 1"""
    return {"c": c.float().cpu().contiguous(), "r": torch.ones(n_in, dtype=torch.float32)}


def c9_family(key):
    """the C9 family name of a writer key (attn_o / gdn_out / down), None for the other linears"""
    return C9FAM.get(qwen35.family(key))


def c9_tensor(tensors, key, codebook, w, hess_path, al, e_clip, dev, mad=None):
    """C9 inside the per-tensor quant job (S2, rxq.job): the writer's job outputs (tensors: {<key>.trellis / .suh / .svh},
    as the job saves them), its bf16 weight w (out, in) and its group's calib H_in file -> (c, statistics). The same
    decode (rxq.deq.weight with the codebook's table, f16 suh / svh) and c9_rows as c9_scales on the assembled level, so
    the scales equal the level pass's on the same device (tests/check_deq_c9.py "injob")."""
    from . import deq
    dev = torch.device(dev)
    tr = tensors[f"{key}.trellis"]
    Q = deq.weight(tr.to(dev), tensors[f"{key}.suh"].to(dev), tensors[f"{key}.svh"].to(dev), tr.shape[2] / 16,
                   torch.from_numpy(deq.lut(codebook, mad)).to(dev)).T.double()
    W = w.to(dev).double()
    assert Q.shape == W.shape, (key, Q.shape, W.shape)
    M = hessian(hess_path, dev)
    c, rec = c9_rows(Q, W, M, al, e_clip)
    del Q, W, M
    return c, rec


def c9_scales(level, out, alpha, model, hess, dn=None, dev="cpu", e_clip=0.5, only=None, n_layers=64):
    from . import deq
    assert set(alpha) <= set(WRITERS), alpha
    if os.path.exists(f"{out}/.done"): print(f"{out} done"); return 0
    os.makedirs(f"{out}/scales", exist_ok=True)
    keys = writer_keys(alpha, n_layers)
    if only: keys = [k for k in keys if k[0] in only]
    lv = deq.Level(level, dev=dev)
    ix = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    done = {json.loads(l)["key"] for l in open(f"{out}/c9.jsonl")} if os.path.exists(f"{out}/c9.jsonl") else set()
    with open(f"{out}/c9.jsonl", "a") as log:
        for key, fam, grp in keys:
            if key in done and os.path.exists(f"{out}/scales/{key}.safetensors"): continue
            al = alpha[fam]
            Q = lv.get(key).double()                                                   # (out, in)
            with safe_open(f"{model}/{ix[key + '.weight']}", "pt", device="cpu") as f: W = f.get_tensor(key + ".weight").to(dev).double()
            assert Q.shape == W.shape, (key, Q.shape, W.shape)
            if dn:
                M = hessian(f"{dn}/hin/{grp}.safetensors", dev)
                with safe_open(f"{dn}/wpre/{grp}.safetensors", "pt", device="cpu") as h: Ft = h.get_tensor("Ft").to(dev).double()
                W = (Ft @ W.T).T; del Ft
            else:
                M = hessian(f"{hess}/{grp}.safetensors", dev)
            c, rec = c9_rows(Q, W, M, al, e_clip)
            save_file(c9_file(c, W.shape[1]), f"{out}/scales/{key}.safetensors.part")
            os.replace(f"{out}/scales/{key}.safetensors.part", f"{out}/scales/{key}.safetensors")
            rec = {"key": key, "fam": fam, "layer": int(key.split(".")[3]), "alpha": al, **rec, "target": "WF/G" if dn else "W/H"}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"{key}: e mean {rec['e_mean']:.4f} max {rec['e_max']:.3f}, c mean {rec['c_mean']:.4f} max {rec['c_max']:.3f}", flush=True)
            del Q, W, M
            if torch.device(dev).type == "cuda": torch.cuda.empty_cache()
    open(f"{out}/.done", "w").close()
    print(f"C9 DONE {len(keys)} writers -> {out}")
    return len(keys)


def merge_scales(dirs, dst):
    """one scales dir from several (e.g. a level's swj and C9 scales: disjoint tensors, so one apply = applying them in
    turn); a single dir is returned as is. The merged dir (DST.scales, symlinks) refuses a tensor in two dirs."""
    if len(dirs) == 1: return dirs[0]
    m = f"{dst.rstrip('/')}.scales"; seen = {}
    for d in dirs:
        if not os.path.exists(f"{d}/.done"): raise SystemExit(f"{d}: not finished (.done missing)")
        for f in os.listdir(f"{d}/scales"):
            if f.endswith(".safetensors"):
                if f in seen: raise SystemExit(f"{f[:-12]}: scales in both {seen[f]} and {d}")
                seen[f] = d
    os.makedirs(f"{m}/scales", exist_ok=True)
    for f, d in sorted(seen.items()):
        if not os.path.lexists(f"{m}/scales/{f}"): os.symlink(os.path.abspath(f"{d}/scales/{f}"), f"{m}/scales/{f}")
    return m


def plan_level(cfg, K):
    """the post passes a level of K gets, in order (per-mix passes listed with their scope)"""
    p = cfg["post"]; out = []
    if p["swj"]["enable"] and float(K) <= p["swj"]["k_max"]: out.append(("swj", "level", {"grid": p["swj"]["grid"], "fused": p["swj"]["fused"]}))
    if p["asym"]["enable"]: out.append(("asym", p["asym"]["scope"], {"lam": p["asym"]["lam"], "rho": p["asym"]["rho"]}))
    if p["c9"]["enable"]:                       # plain writers: computed in the quant jobs, folded with swj's apply
        out.append(("c9", "mix" if p["asym"]["enable"] else "job", {"alpha": {k: v for k, v in p["c9"]["alpha"].items() if v}}))
    return out


def plan_asym(cfg, mix):
    """the per-mix asym writer pass (#111) as steps; raises while post.asym is off"""
    a = cfg["post"]["asym"]
    if not a["enable"]: raise SystemExit("post.asym is off (#105 open; #111: per-mix only): nothing to plan")
    return [("siblings", f"the mix's siblings as one level ({mix}: each tensor from its picked level; rxq.pairs splice per level)"),
            ("moments", f"rxq.sibmom over that level, {a['moment_tokens']} qad tokens (a level pass, ~2.5-3 card-h at 27B)"),
            ("denoise", f"rxq.asym denoise --lam {a['lam']} --rho {a['rho']} (per writer group, CPU / GPU)"),
            ("requant", "the 128 writers at their mix K: rxq quant --hin DN/hin --wpre DN/wpre (S2 jobs)"),
            ("splice", "the writers into the mix's level copies (rxq.pairs splice), then c9 if on, then pack")]


def main(argv=None):
    from . import config as C, swj
    ap = argparse.ArgumentParser(prog="rxq.post")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("c9"); c.add_argument("level"); c.add_argument("out"); c.add_argument("--alpha")
    c.add_argument("--model", required=True); c.add_argument("--hess", required=True); c.add_argument("--dn")
    c.add_argument("--config"); c.add_argument("--layers", type=int, default=64)
    p = sub.add_parser("apply"); p.add_argument("level"); p.add_argument("scales"); p.add_argument("dst")
    q = sub.add_parser("plan"); q.add_argument("-K", type=float, required=True); q.add_argument("--config")
    a = ap.parse_args(argv)
    if a.cmd == "apply": swj.apply_level(a.level, merge_scales(a.scales.split(","), a.dst), a.dst); return
    cfg = C.load(a.config)
    if a.cmd == "plan":
        for step in plan_level(cfg, a.K): print(json.dumps(step))
        return
    al = ({k: float(v) for k, v in (x.split("=") for x in a.alpha.split(","))} if a.alpha else
          {k: float(v) for k, v in cfg["post"]["c9"]["alpha"].items()})
    only = os.environ.get("RXQ_C9_ONLY")                                           # tests only (c9e27's C9E27_ONLY)
    c9_scales(a.level, a.out, al, a.model, a.hess, a.dn, os.environ.get("DEV", "cpu"), cfg["post"]["c9"]["e_clip"],
              only.split(",") if only else None, a.layers)


if __name__ == "__main__":
    main()
