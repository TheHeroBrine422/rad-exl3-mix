#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""soup198.py (w186, #198, RULES198): average the trained log multipliers of independent scale-QAD runs.

  soup198.py build OUT {mean,consensus} CKPT [CKPT ...] [--start CKPT] [--force]
  soup198.py unit OUT CKPT [--start CKPT]     a soup of one run = that run (state bitwise; scale files compared)
  soup198.py selftest                         synthetic checkpoints: formulas, file format, refusal paths

A checkpoint is qad196.py's ckpt-<T>/ (qad196 write_ckpt): params.pt = {"state": {"<linear>:r" | "<linear>:c": float32
LOG multiplier, ...}, "meta": {...}}; scales/<linear>.safetensors = {"r", "c"} = exp(state) as float32 (ones where the
linear has no such parameter); .done. The served path (rxq.swj.apply_level) reads only scales/, so a soup is served by
pointing eval196.py --ckpt at OUT.

mean      every scale parameter = mean over the members of its log multiplier (the state value). This is RULES198's soup.
consensus C2: d_i = state_i - start; where all members have the same sign of d_i: start + mean(d_i); else start (0 moved).
          No line search. (start = the untrained student's state; log 1 = 0 everywhere, checked with --start.)
shrink    SH (RULES198 amendment 1, eval only): per log multiplier, m = mean over the N members and v = unbiased variance
          across them; value = m * m^2 / (m^2 + v / N), 0 where m = 0. No fitted parameter. Needs N >= 2. With N identical
          members v = 0 and the value is m exactly, so it equals the plain mean soup bitwise (unitshrink checks it).
Everything that is not a scale parameter must be identical across members (checked) and is copied from CKPT[0].
Member scale files are checked against exp(state) on load (max relative difference is recorded in soup.json).
exp is taken on DEVICE (default: the GPU when torch sees one, as the trainer's write_ckpt does): on CPU the float32 exp
differs from the GPU's in ~4 % of values by 1 ulp, which moves the served float16 rounding (w186, 11-16: served KL +0.04 %
x181g / -0.17 % fe38 for a soup of one). On the GPU a soup of one reproduces the member's scale files bitwise.
Writes OUT.part/ and renames it to OUT once complete (.done is the last file written inside).
"""
import argparse, hashlib, json, os, shutil, sys, tempfile

import numpy as np
import torch
from safetensors.torch import load_file, save_file

SUFFIX = (":r", ":c")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ckpt(d, dev="cpu"):
    """state (float32 CPU), meta, scale keys, the scale files, and the max relative gap between the files and exp(state)"""
    z = torch.load(os.path.join(d, "params.pt"), map_location="cpu", weights_only=False)
    st = {k: v.detach().to("cpu", torch.float32).clone() for k, v in z["state"].items()}
    sd = os.path.join(d, "scales")
    keys = sorted(f[: -len(".safetensors")] for f in os.listdir(sd) if f.endswith(".safetensors"))
    files, gap = {}, 0.0
    for k in keys:
        t = load_file(os.path.join(sd, k + ".safetensors"))
        r, c = t["r"].to(torch.float32), t["c"].to(torch.float32)
        for nm, f_ in (("r", r), ("c", c)):
            p = k + ":" + nm
            if p in st:
                e = torch.exp(st[p].to(dev)).cpu()
                gap = max(gap, float(((f_.double() - e.double()).abs() / e.double().abs()).max()))
            else:
                assert bool((f_ == 1).all()), f"{d}: {k}:{nm} has no state parameter but its scale file is not all ones"
        files[k] = (r, c)
    orphan = sorted({n[: -2] for n in st if n.endswith(SUFFIX)} - set(keys))
    assert not orphan, f"{d}: state parameters without a scale file: {orphan[:3]}"
    assert gap <= 1e-5, f"{d}: scale files differ from exp(state) by {gap:.3g} (relative)"
    return {"dir": d, "state": st, "meta": z["meta"], "keys": keys, "files": files, "gap": gap,
            "sha": sha256(os.path.join(d, "params.pt"))}


def check_same_student(M):
    ref = M[0]
    names = sorted(ref["state"])
    for m in M[1:]:
        assert sorted(m["state"]) == names, f"{m['dir']}: different parameter names"
        assert m["keys"] == ref["keys"], f"{m['dir']}: different scale keys"
        for n in names:
            assert m["state"][n].shape == ref["state"][n].shape, f"{m['dir']}: shape of {n}"
            assert m["state"][n].dtype == ref["state"][n].dtype, f"{m['dir']}: dtype of {n}"
    for n in names:
        if n.endswith(SUFFIX): continue
        for m in M[1:]:
            assert torch.equal(m["state"][n], ref["state"][n]), f"{m['dir']}: non-scale parameter {n} differs from {ref['dir']}"
    return names


def build(out, mode, dirs, start_dir=None, force=False, dev="cpu"):
    assert mode in ("mean", "consensus", "shrink"), mode
    if os.path.exists(os.path.join(out, ".done")) and not force:
        print(f"soup198: {out} exists (.done): skipped", file=sys.stderr); return None
    M = [load_ckpt(d, dev) for d in dirs]
    if mode == "shrink": assert len(M) >= 2, "shrink needs at least two members (unbiased variance)"
    names = check_same_student(M)
    ref = M[0]
    scale = [n for n in names if n.endswith(SUFFIX)]
    if start_dir:
        S = load_ckpt(start_dir, dev)
        assert sorted(S["state"]) == names and S["keys"] == ref["keys"], "start checkpoint: different student"
        for n in names:
            assert S["state"][n].shape == ref["state"][n].shape, f"start checkpoint: shape of {n}"
        start = {n: S["state"][n].double() for n in scale}
        start_info = {"dir": start_dir, "sha256": S["sha"], "max_abs": max(float(v.abs().max()) for v in start.values()) if scale else 0.0}
    else:
        start = {n: torch.zeros(ref["state"][n].shape, dtype=torch.float64) for n in scale}
        start_info = {"dir": None, "assumed": "log 1 everywhere (0)"}
    st = {}
    agree_frac = {}
    shrink_ratio = []
    for n in names:
        if not n.endswith(SUFFIX):
            st[n] = ref["state"][n].clone(); continue
        X = torch.stack([m["state"][n].double() for m in M])
        if mode == "mean":
            Y = X.mean(0)
        elif mode == "shrink":
            N = X.shape[0]
            mu = X.mean(0); var = X.var(0, unbiased=True)
            den = mu * mu + var / N
            pos = den > 0
            ratio = torch.where(pos, mu * mu / torch.where(pos, den, torch.ones_like(den)), torch.zeros_like(mu))
            Y = mu * ratio
            nz = mu != 0
            if bool(nz.any()): shrink_ratio.append(ratio[nz].flatten())
        else:
            D = X - start[n].unsqueeze(0)
            sg = torch.sign(D)
            agree = (sg == sg[:1]).all(0)
            Y = start[n] + torch.where(agree, D.mean(0), torch.zeros_like(D[0]))
            agree_frac[n] = float(agree.double().mean())
        st[n] = Y.to(torch.float32).contiguous()
    scales = {}
    for k in ref["keys"]:
        r = torch.exp(st[k + ":r"].to(dev)).cpu().to(torch.float32) if (k + ":r") in st else ref["files"][k][0].clone()
        c = torch.exp(st[k + ":c"].to(dev)).cpu().to(torch.float32) if (k + ":c") in st else ref["files"][k][1].clone()
        scales[k] = (r.contiguous(), c.contiguous())
    # member spread and the soup's distance to each member (log units), for the manifest
    spread = {}
    if len(M) > 1:
        sds = [float(torch.stack([m["state"][n] for m in M]).std(0, unbiased=False).mean()) for n in scale]
        spread = {"mean_over_params_of_std_across_members": float(np.mean(sds))}
    dist = [{"dir": m["dir"], "mean_abs_log_diff": float(np.mean([float((st[n] - m["state"][n]).abs().mean()) for n in scale])),
             "max_abs_log_diff": float(max(float((st[n] - m["state"][n]).abs().max()) for n in scale))} for m in M]
    sign_pairs = []
    if mode == "consensus" or len(M) > 1:
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                ai = torch.cat([(M[i]["state"][n] - start[n].float()).flatten() for n in scale])
                aj = torch.cat([(M[j]["state"][n] - start[n].float()).flatten() for n in scale])
                nz = (ai != 0) & (aj != 0)
                sign_pairs.append({"a": M[i]["dir"], "b": M[j]["dir"], "same_sign_frac_of_both_nonzero": float((torch.sign(ai[nz]) == torch.sign(aj[nz])).float().mean()) if bool(nz.any()) else None,
                                   "nonzero_frac": float(nz.float().mean())})
    manifest = {
        "tag": os.path.basename(os.path.abspath(out)), "mode": mode, "exp_device": dev,
        "members": [{"dir": m["dir"], "params_sha256": m["sha"], "scale_file_gap_rel": m["gap"],
                     "steps": m["meta"].get("steps"), "rows": m["meta"].get("rows"), "checksum": m["meta"].get("checksum"),
                     "args_seed": (m["meta"].get("args") or {}).get("seed"), "args_mix50": (m["meta"].get("args") or {}).get("mix50"),
                     "args_a8": (m["meta"].get("args") or {}).get("a8")} for m in M],
        "start": start_info, "n_scale_params": len(scale), "n_scale_keys": len(ref["keys"]),
        "n_nonscale_params": len(names) - len(scale), "member_spread": spread, "soup_vs_member": dist,
        "sign_pairs": sign_pairs,
        "consensus_agree_frac_mean": (float(np.mean(list(agree_frac.values()))) if agree_frac else None),
        "soup_abs_log_mean": float(np.mean([float(st[n].abs().mean()) for n in scale])) if scale else None,
    }
    if mode == "shrink":
        rr = torch.cat(shrink_ratio).double()
        manifest["shrink"] = {"formula": "m * m^2 / (m^2 + v/N), v unbiased, N = members", "n": len(M),
                              "ratio_mean": float(rr.mean()), "ratio_median": float(rr.median()), "ratio_min": float(rr.min()),
                              "frac_ratio_below_0.5": float((rr < 0.5).double().mean()), "nonzero_params": int(rr.numel())}
    meta = dict(ref["meta"]); meta["soup"] = {"mode": mode, "members": [m["dir"] for m in M], "start": start_info, "exp_device": dev}
    tmp = out + ".part"
    if os.path.exists(tmp): shutil.rmtree(tmp)
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(os.path.join(tmp, "scales"))
    for k, (r, c) in scales.items():
        save_file({"r": r, "c": c}, os.path.join(tmp, "scales", k + ".safetensors"))
    torch.save({"state": st, "meta": meta}, os.path.join(tmp, "params.pt.part")); os.replace(os.path.join(tmp, "params.pt.part"), os.path.join(tmp, "params.pt"))
    json.dump({"mon_kl": None, "soup": True}, open(os.path.join(tmp, "mon.json"), "w"))
    json.dump(manifest, open(os.path.join(tmp, "soup.json"), "w"), indent=1)
    open(os.path.join(tmp, ".done"), "w").close()
    os.replace(tmp, out)
    print(f"soup198: {out} <- {mode} of {len(M)} member(s); {len(scale)} scale params; "
          f"gap {max(m['gap'] for m in M):.2g}", file=sys.stderr)
    return manifest


def unit(out, member, start_dir=None, dev="cpu"):
    """a soup of one run = that run: state bitwise equal; scale files compared with the member's own files (bitwise on
    the GPU: the same exp as the trainer's write)"""
    build(out, "mean", [member], start_dir=start_dir, force=True, dev=dev)
    A, B = load_ckpt(member, dev), load_ckpt(out, dev)
    names = sorted(A["state"])
    state_eq = all(torch.equal(A["state"][n], B["state"][n]) for n in names)
    n_eq = n_tot = 0; maxrel = 0.0
    for k in A["keys"]:
        for x, y in zip(A["files"][k], B["files"][k]):
            n_tot += x.numel(); n_eq += int((x == y).sum())
            maxrel = max(maxrel, float(((x.double() - y.double()).abs() / x.double().abs().clamp_min(1e-30)).max()))
    rep = {"member": member, "soup": out, "exp_device": dev, "state_bitwise_equal": bool(state_eq), "state_params": len(names),
           "scale_values": n_tot, "scale_values_bitwise_equal": n_eq, "scale_max_rel_diff": maxrel}
    assert state_eq, "unit check failed: state differs"
    assert maxrel <= 2.5e-7, f"unit check failed: scale files differ by {maxrel:.3g}"
    rep["exact"] = bool(state_eq and n_eq == n_tot)
    print(json.dumps(rep))
    return rep


def unitshrink(out_dir, member, n, start_dir=None, dev="cpu"):
    """N identical members: the shrink soup equals the plain mean soup exactly (state and scale files, bitwise)"""
    tm, ts = out_dir + "-mean", out_dir + "-shrink"
    build(tm, "mean", [member] * n, start_dir=start_dir, force=True, dev=dev)
    build(ts, "shrink", [member] * n, start_dir=start_dir, force=True, dev=dev)
    A, B = load_ckpt(tm, dev), load_ckpt(ts, dev)
    names = sorted(A["state"])
    state_eq = all(torch.equal(A["state"][n_], B["state"][n_]) for n_ in names)
    n_eq = n_tot = 0
    for k in A["keys"]:
        for x, y in zip(A["files"][k], B["files"][k]):
            n_tot += x.numel(); n_eq += int((x == y).sum())
    rep = {"member": member, "n": n, "exp_device": dev, "state_bitwise_equal": bool(state_eq), "state_params": len(names),
           "scale_values": n_tot, "scale_values_bitwise_equal": n_eq, "exact": bool(state_eq and n_eq == n_tot)}
    shutil.rmtree(tm); shutil.rmtree(ts)
    print(json.dumps(rep))
    assert rep["exact"], "unitshrink failed: shrink of identical members differs from the mean soup"
    return rep


def selftest():
    rng = np.random.default_rng(0)
    T = tempfile.mkdtemp(prefix="soup198-selftest-")
    def mk(name, logs, extra=None, zero_start=True):
        d = os.path.join(T, name); os.makedirs(os.path.join(d, "scales"))
        st = {}
        for k, (r, c) in logs.items():
            if r is not None: st[k + ":r"] = torch.tensor(r, dtype=torch.float32)
            if c is not None: st[k + ":c"] = torch.tensor(c, dtype=torch.float32)
        for n, v in (extra or {}).items(): st[n] = torch.tensor(v, dtype=torch.float32)
        torch.save({"state": st, "meta": {"steps": 3, "rows": 10, "args": {"seed": 0}}}, os.path.join(d, "params.pt"))
        for k, (r, c) in logs.items():
            rr = torch.exp(torch.tensor(r, dtype=torch.float32)) if r is not None else torch.ones(2)
            cc = torch.exp(torch.tensor(c, dtype=torch.float32)) if c is not None else torch.ones(3)
            save_file({"r": rr, "c": cc}, os.path.join(d, "scales", k + ".safetensors"))
        open(os.path.join(d, ".done"), "w").close()
        return d
    def arr(n): return rng.normal(0, 0.01, n).astype(np.float32)
    lin = {"l.a": (arr(2), arr(3)), "l.b": (None, arr(3))}            # l.b has no r (identity)
    start_dir = mk("start", {"l.a": (np.zeros(2), np.zeros(3)), "l.b": (None, np.zeros(3))}, {"norm.x": np.zeros(4)})
    m1 = mk("m1", lin, {"norm.x": np.zeros(4)})
    lin2 = {"l.a": (arr(2), arr(3)), "l.b": (None, arr(3))}
    m2 = mk("m2", lin2, {"norm.x": np.zeros(4)})
    m3 = mk("m3", {"l.a": (arr(2), arr(3)), "l.b": (None, arr(3))}, {"norm.x": np.zeros(4)})
    # 1: mean of logs = arithmetic mean of the state values; scale files = exp of it; identity stays 1
    out = os.path.join(T, "mean2")
    build(out, "mean", [m1, m2])
    S = torch.load(os.path.join(out, "params.pt"), weights_only=False)["state"]
    B = load_ckpt(out)
    for n in ("l.a:r", "l.a:c", "l.b:c"):
        a = torch.load(os.path.join(m1, "params.pt"), weights_only=False)["state"][n]
        b = torch.load(os.path.join(m2, "params.pt"), weights_only=False)["state"][n]
        assert torch.allclose(S[n], (a + b) / 2, atol=1e-7, rtol=0), n
    assert torch.equal(B["files"]["l.b"][0], torch.ones(2)), "identity r must stay 1"
    assert torch.equal(S["norm.x"], torch.zeros(4)), "non-scale parameter copied from the student"
    # 2: consensus over a pair with a known sign split: same sign -> mean, opposite -> start (0); start = 0 here
    ca = mk("ca", {"l.a": ([0.01, 0.02, -0.03, 0.0], [0.5, -0.25, 0.1]), "l.b": (None, [0.1, 0.1, -0.1])}, {"norm.x": np.zeros(4)})
    cb = mk("cb", {"l.a": ([-0.02, 0.04, 0.05, 0.0], [0.5, -0.5, 0.2]), "l.b": (None, [0.3, -0.1, -0.1])}, {"norm.x": np.zeros(4)})
    out = os.path.join(T, "cons2")
    start4 = mk("start4", {"l.a": ([0.0] * 4, [0.0] * 3), "l.b": (None, [0.0] * 3)}, {"norm.x": np.zeros(4)})
    build(out, "consensus", [ca, cb], start_dir=start4)
    S = torch.load(os.path.join(out, "params.pt"), weights_only=False)["state"]
    assert torch.allclose(S["l.a:r"], torch.tensor([0.0, 0.03, 0.0, 0.0]), atol=1e-7, rtol=0), S["l.a:r"]
    assert torch.allclose(S["l.a:c"], torch.tensor([0.5, -0.375, 0.15]), atol=1e-7, rtol=0), S["l.a:c"]
    assert torch.allclose(S["l.b:c"], torch.tensor([0.2, 0.0, -0.1]), atol=1e-7, rtol=0), S["l.b:c"]
    mm = build(os.path.join(T, "mean-ab"), "mean", [ca, cb])
    assert mm["consensus_agree_frac_mean"] is None and mm["n_scale_params"] == 3
    # 3: a soup of one is the member, exactly (state) and within float32 rounding (scale files)
    rep = unit(os.path.join(T, "one"), m3, start_dir=start_dir)
    assert rep["state_bitwise_equal"]
    # 3b: shrink: hand-computed value (m = 0.02, v = 0.0002 unbiased, N = 2 -> ratio 0.8 -> 0.016); zero mean -> 0;
    #     identical members -> the mean exactly; one member -> refused
    sa = mk("sh_a", {"l.a": ([0.01, 0.01, 0.0, 0.0], [0.5, -0.25, 0.1]), "l.b": (None, [0.1, 0.1, -0.1])}, {"norm.x": np.zeros(4)})
    sb = mk("sh_b", {"l.a": ([0.03, -0.01, 0.0, 0.0], [0.5, -0.25, 0.3]), "l.b": (None, [0.3, -0.1, -0.1])}, {"norm.x": np.zeros(4)})
    shd = os.path.join(T, "shrink2")
    msh = build(shd, "shrink", [sa, sb], start_dir=start4)
    S = torch.load(os.path.join(shd, "params.pt"), weights_only=False)["state"]
    assert abs(float(S["l.a:r"][0]) - 0.016) < 1e-7, S["l.a:r"]          # m = 0.02, v = 0.0002, ratio = 0.8
    assert float(S["l.a:r"][2]) == 0.0 and float(S["l.a:r"][3]) == 0.0    # zero mean -> 0
    assert abs(float(S["l.a:c"][0]) - 0.5) < 1e-7                         # identical values -> exact
    assert msh["shrink"]["n"] == 2 and msh["mode"] == "shrink"
    unitshrink(os.path.join(T, "ushr2"), sa, 2, start_dir=start4)
    unitshrink(os.path.join(T, "ushr4"), m1, 4, start_dir=start_dir)
    try:
        build(os.path.join(T, "shrink1"), "shrink", [m1]); raise SystemExit("selftest: shrink of one member not refused")
    except AssertionError as e:
        print("refused (shrink one member):", str(e)[:70])
    # 4: refusals: different student (a shape), a non-scale parameter that differs, a scale file that is not exp(state)
    bad = mk("bad_shape", {"l.a": (arr(5), arr(3)), "l.b": (None, arr(3))}, {"norm.x": np.zeros(4)})
    for args in ([m1, bad], ):
        try:
            build(os.path.join(T, "bad1"), "mean", list(args)); raise SystemExit("selftest: shape mismatch not refused")
        except AssertionError as e:
            print("refused (shape):", str(e)[:90])
    nsd = mk("nonscale", lin, {"norm.x": np.ones(4)})
    try:
        build(os.path.join(T, "bad2"), "mean", [m1, nsd]); raise SystemExit("selftest: non-scale mismatch not refused")
    except AssertionError as e:
        print("refused (non-scale):", str(e)[:90])
    corrupt = mk("corrupt", lin, {"norm.x": np.zeros(4)})
    save_file({"r": torch.ones(2) * 2, "c": torch.exp(torch.tensor(lin["l.a"][1]))}, os.path.join(corrupt, "scales", "l.a.safetensors"))
    try:
        build(os.path.join(T, "bad3"), "mean", [m1, corrupt]); raise SystemExit("selftest: corrupt scale file not refused")
    except AssertionError as e:
        print("refused (scale file):", str(e)[:90])
    # 5: a finished soup is not rebuilt
    assert build(out, "consensus", [m1, m2], start_dir=start_dir) is None
    shutil.rmtree(T)
    print("soup198 selftest: ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("out"); b.add_argument("mode", choices=["mean", "consensus", "shrink"])
    b.add_argument("ckpts", nargs="+"); b.add_argument("--start"); b.add_argument("--force", action="store_true")
    b.add_argument("--device", default="auto", help="where exp runs: cpu, cuda:N (ROCm too), or auto (cuda:0 if visible)")
    u = sub.add_parser("unit"); u.add_argument("out"); u.add_argument("ckpt"); u.add_argument("--start")
    u.add_argument("--device", default="auto")
    us = sub.add_parser("unitshrink"); us.add_argument("out"); us.add_argument("ckpt"); us.add_argument("n", type=int)
    us.add_argument("--start"); us.add_argument("--device", default="cpu")
    sub.add_parser("selftest")
    a = ap.parse_args()
    dev = ("cuda:0" if torch.cuda.is_available() else "cpu") if a.cmd != "selftest" and getattr(a, "device", "auto") == "auto" \
        else getattr(a, "device", "cpu")
    if a.cmd == "build": build(os.path.abspath(a.out), a.mode, [os.path.abspath(x) for x in a.ckpts],
                               os.path.abspath(a.start) if a.start else None, a.force, dev=dev)
    elif a.cmd == "unit": unit(os.path.abspath(a.out), os.path.abspath(a.ckpt), os.path.abspath(a.start) if a.start else None, dev=dev)
    elif a.cmd == "unitshrink": unitshrink(os.path.abspath(a.out), os.path.abspath(a.ckpt), a.n,
                                           os.path.abspath(a.start) if a.start else None, dev=a.device)
    else: selftest()


if __name__ == "__main__":
    main()
