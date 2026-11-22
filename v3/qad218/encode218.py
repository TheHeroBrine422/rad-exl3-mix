#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""encode218.py OUT --grad GDIR[,GDIR...] [--base BDIR] --etas 0,0.01,... [--tool EXE] -- #218's discrete step: for each eta, the probe
tensors' target W_t = W_hat - eta / rms(G) * G (W_hat: the current codes' values, BDIR/<key>.q or the released level's; G: the
gradient, the rows-weighted mean of the GDIRs' (probe218.py grad chunks at the same codes; merged once into OUT/grad); rms over all probe tensors together, so eta is the step's rms in codebook units and the tensors keep their relative
gradient scale), re-encoded at the tensor's own K and its released suh / svh by the native quantizer's reencode_fixed
(exl3v3_tool reencode: LDLQ in the stored codes' metric Hb S H S Hb with the calib Hessian, sigma_reg 0.025, the fp16 tile search).

Writes OUT/eta-<e>/<key>.q (f32 [k, n]), <key>.rec (u32 record, exl3i plane 1), stats.json (per tensor: proxy, tiles whose record
words / values changed vs the current codes, the record's decode == .q). --precond hinv: G -> H'^-1 G per output column (H' the
damped rotated metric), rescaled the same way. Restartable per eta (stats.json last)."""
import argparse, json, os, subprocess, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from rxq import deq  # noqa: E402

LP = "model.language_model.layers"


def hname(key):
    i = key.split(".")[3]; sub = key.split(".", 4)[4]
    if sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"):
        return f"{LP}.{i}.input"
    if sub in ("mlp.gate_proj", "mlp.up_proj"): return f"{LP}.{i}.post_attention.input"
    return f"{key}.input"                                   # o_proj, linear_attn.out_proj, mlp.down_proj


def rec_to_trellis(rec, k, n, K):
    m = int(round(16 * K))
    return np.ascontiguousarray(rec.view(np.int16).reshape(n // 16, k // 16, m).transpose(1, 0, 2))


def tiles_changed(a, b, k, n):
    """tiles of a [k, n] pair (16 x 16) that differ anywhere"""
    d = (a != b).reshape(k // 16, 16, n // 16, 16).any(axis=(1, 3))
    return int(d.sum())


def merge(gdirs, out):
    """the rows-weighted mean of several chunks' gradients (each the mean-KL gradient over its own rows) -> out (restartable)"""
    if os.path.exists(os.path.join(out, "grad.json")): return out
    os.makedirs(out, exist_ok=True)
    zs = [json.load(open(os.path.join(d, "grad.json"))) for d in gdirs]
    assert len({z["codes"] for z in zs}) == 1, "chunks at different codes"
    N = sum(z["rows"] for z in zs); meta = {}
    for k, v in zs[0]["tensors"].items():
        g = np.zeros((v["k"], v["n"]), np.float64)
        for d, z in zip(gdirs, zs): g += (z["rows"] / N) * np.fromfile(os.path.join(d, k + ".g"), np.float32).reshape(v["k"], v["n"])
        g = g.astype(np.float32); g.tofile(os.path.join(out, k + ".g"))
        meta[k] = {**v, "g_rms": float(np.sqrt((g.astype(np.float64) ** 2).mean()))}
    json.dump({"chunks": gdirs, "rows": N, "codes": zs[0]["codes"], "tensors": meta}, open(os.path.join(out, "grad.json.part"), "w"))
    os.replace(os.path.join(out, "grad.json.part"), os.path.join(out, "grad.json"))
    print(f"merged {len(gdirs)} chunks, {N} rows -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--grad", required=True); ap.add_argument("--base", default="")
    ap.add_argument("--etas", required=True); ap.add_argument("--lv", default="/data/w218/lv/2.50")
    ap.add_argument("--tool", default="/data/w218/bin/exl3v3_tool"); ap.add_argument("--hess", default="/data/challenge/hessians/calib")
    ap.add_argument("--precond", choices=("none",), default="none"); ap.add_argument("--keep-targets", action="store_true")
    a = ap.parse_args()
    gdirs = a.grad.split(",")
    if len(gdirs) > 1:
        a.grad = merge(gdirs, os.path.join(a.out, "grad"))
    gm = json.load(open(os.path.join(a.grad, "grad.json")))["tensors"]
    keys = sorted(gm, key=lambda k: (hname(k), k))
    lev = deq.Level(f"{a.lv}/trunk", "cpu")
    table = torch.from_numpy(deq.lut(lev.codebook))
    g2 = sum(gm[k]["g_rms"] ** 2 * gm[k]["k"] * gm[k]["n"] for k in keys); nv = sum(gm[k]["k"] * gm[k]["n"] for k in keys)
    g_rms = float(np.sqrt(g2 / nv))
    print(f"{len(keys)} tensors, G rms {g_rms:.4e}", flush=True)

    def current(k):
        """(W_hat f32 [k, n], trellis int16 of the current codes)"""
        kk, n, K = gm[k]["k"], gm[k]["n"], gm[k]["K"]
        if a.base and os.path.exists(os.path.join(a.base, k + ".q")):
            w = np.fromfile(os.path.join(a.base, k + ".q"), np.float32).reshape(kk, n)
            tr = rec_to_trellis(np.fromfile(os.path.join(a.base, k + ".rec"), np.uint32), kk, n, K)
            return w, tr
        raw = lev.raw(k)
        tr = raw["trellis"]
        return deq.decode_hat(tr, K, table).numpy(), tr.numpy()

    for e in [float(x) for x in a.etas.split(",")]:
        od = os.path.join(a.out, f"eta-{e:g}")
        if os.path.exists(os.path.join(od, "stats.json")): print(f"eta {e:g}: done before", flush=True); continue
        os.makedirs(od, exist_ok=True)
        t0 = time.time(); lines = []
        for k in keys:
            kk, n = gm[k]["k"], gm[k]["n"]
            w, _ = current(k)
            g = np.fromfile(os.path.join(a.grad, k + ".g"), np.float32).reshape(kk, n)
            t = (w - (e / g_rms) * g).astype(np.float32) if e else w
            t.tofile(os.path.join(od, k + ".target"))
            np.asarray(lev.raw(k)["suh"].float().numpy(), np.float64).tofile(os.path.join(od, k + ".suh"))
            lines.append(f"{hname(k)} {gm[k]['K']:g} {kk} {n} {od}/{k}.target {od}/{k}.suh {od}/{k}")
        open(os.path.join(od, "list.txt"), "w").write("\n".join(lines) + "\n")
        t1 = time.time()
        r = subprocess.run([a.tool, "reencode", "--list", os.path.join(od, "list.txt"), "--hess", a.hess],
                           capture_output=True, text=True, env={**os.environ, "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", "0")})
        if r.returncode:
            print(r.stdout, r.stderr, flush=True); sys.exit(f"reencode failed (rc {r.returncode})")
        tool = {json.loads(l)["out"].rsplit("/", 1)[1]: json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")}
        t2 = time.time()
        st = {"eta": e, "g_rms": g_rms, "base": a.base or "released", "grad": a.grad, "sec_targets": t1 - t0, "sec_encode": t2 - t1, "tensors": {}}
        for k in keys:
            kk, n, K = gm[k]["k"], gm[k]["n"], gm[k]["K"]
            w, tr0 = current(k)
            q = np.fromfile(os.path.join(od, k + ".q"), np.float32).reshape(kk, n)
            tr = rec_to_trellis(np.fromfile(os.path.join(od, k + ".rec"), np.uint32), kk, n, K)
            dec = deq.decode_hat(torch.from_numpy(tr), K, table).numpy()
            tw = int((tr != tr0).any(axis=2).sum())
            st["tensors"][k] = {"K": K, "k": kk, "n": n, "proxy": tool[k]["proxy"], "sec": tool[k]["sec_quant"],
                                "tiles": kk * n // 256, "tiles_words_changed": tw, "tiles_values_changed": tiles_changed(q, w, kk, n),
                                "decode_equal": bool(np.array_equal(dec, q)), "rms_dq": float(np.sqrt(((q - w).astype(np.float64) ** 2).mean()))}
            if not a.keep_targets: os.remove(os.path.join(od, k + ".target"))
        tt = sum(v["tiles"] for v in st["tensors"].values())
        st["tiles"] = tt; st["tiles_values_changed"] = sum(v["tiles_values_changed"] for v in st["tensors"].values())
        st["tiles_words_changed"] = sum(v["tiles_words_changed"] for v in st["tensors"].values())
        st["decode_all_equal"] = all(v["decode_equal"] for v in st["tensors"].values())
        json.dump(st, open(os.path.join(od, "stats.json.part"), "w"), indent=1); os.replace(os.path.join(od, "stats.json.part"), os.path.join(od, "stats.json"))
        print(f"eta {e:g}: encode {t2 - t1:.0f} s, tiles values changed {st['tiles_values_changed']} / {tt} "
              f"({100 * st['tiles_values_changed'] / tt:.2f} %), words {st['tiles_words_changed']}, decode == q: {st['decode_all_equal']}", flush=True)


if __name__ == "__main__":
    main()
