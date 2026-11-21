#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rad2level214.py RAD OUT --model BF16_DIR [--check N] -- a released EXL3-V3 container (exl3i, mad) as the QAD student's inputs (#214):

  OUT/trunk/   an ourq-layout level (q-LNN.safetensors per layer + model.safetensors.index.json + stamp.json + config.json):
               every trunk linear's trellis (int16 [k/16, n/16, 16 K]), suh, svh (f16) and a .mul1 marker, under the checkpoint's
               names (model.language_model.layers.N.<module>); ffn_gate_up is split into mlp.gate_proj / mlp.up_proj
  OUT/head/    the same for `lm_head` (the container's output.weight; rows stored in R16 order)
  OUT/R16.npy  stored row s -> vocabulary row (exl3i_r16_row), qad's --head-perm
  OUT/ab.safetensors   the GDN a / b weights (the container's ssm_ab, K4 mad, stacked) decoded to bf16 as blk.N.ssm_a / ssm_b.weight
                       (qad200's --ab; the served model reads these, not the checkpoint's)
  OUT/map214.json      container weight -> {level keys, rate, rows, cols}; the scale planes' layout (used by swap214.py)

The container's record (plane `record`, u32 [rows/16, cols/16 * m/2], exl3i_encoding.h) holds weight rows 16 t .. 16 t + 15 in
tile row t, its cols/16 tiles in k order, 8 K little-endian u32 words each (exllamav3's tile stream); the level's trellis is the
same words as int16 [k tile, n tile, 16 K]: trellis[kt, nt] = record[nt, kt]. Scales: suh [k] (ffn_gate_up: suh = gate's, suh.1 =
up's), svh [rows] (gate's rows then up's).

--check N: decode N sampled trunk tensors (and the ab / head) on the CPU and print the relative error vs the bf16 checkpoint's
weight (a wrong mapping reads ~1.0; K ~2.5 mad reads ~0.2-0.4)."""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from rad214 import Rad  # noqa: E402

FAM = {"attn_qg": ["self_attn.q_proj"], "attn_k": ["self_attn.k_proj"], "attn_v": ["self_attn.v_proj"],
       "attn_output": ["self_attn.o_proj"], "ssm_qkv": ["linear_attn.in_proj_qkv"], "ssm_z": ["linear_attn.in_proj_z"],
       "ssm_out": ["linear_attn.out_proj"], "ffn_gate_up": ["mlp.gate_proj", "mlp.up_proj"], "ffn_down": ["mlp.down_proj"]}
LP = "model.language_model.layers"


def parse_transform(t):
    """'mad/h128/k2.625[/r16]' -> (codebook, rate m = 16 K, r16)"""
    parts = t.split("/")
    assert parts[1] == "h128" and parts[2][0] == "k", t
    K = float(parts[2][1:])
    return parts[0], int(round(K * 16)), len(parts) > 3 and parts[3] == "r16"


def r16_perm(rows):
    s = np.arange(rows)
    return ((s >> 4) & 7) * (rows // 8) + (s >> 7) * 16 + (s & 15)


def trellis_of(rad, name, nt0=0, nt1=None):
    """int16 [k/16, (nt1 - nt0), 16 K] for tile rows [nt0, nt1) of the record"""
    e = rad.entries[name]
    cb, m, _ = parse_transform(e.transform)
    rows, cols = e.shape
    rec = np.frombuffer(rad.read(name, "record"), dtype="<u4").reshape(rows // 16, (cols // 16) * (m // 2))
    rec = rec[nt0:nt1]
    t = np.ascontiguousarray(rec).view(np.int16).reshape(rec.shape[0], cols // 16, m)
    return np.ascontiguousarray(t.transpose(1, 0, 2))


def f16_plane(rad, name, role):
    return np.frombuffer(rad.read(name, role), dtype="<f2").copy()


def tensors_of(rad, w):
    """container weight w -> {level key: {trellis, suh, svh}} (numpy)"""
    e = rad.entries[w]
    _, blk, fam, _ = w.split(".")
    rows, cols = e.shape
    if fam == "ffn_gate_up":
        h = rows // 2
        return {f"{LP}.{blk}.mlp.gate_proj": {"trellis": trellis_of(rad, w, 0, h // 16), "suh": f16_plane(rad, w, "suh"), "svh": f16_plane(rad, w, "svh")[:h]},
                f"{LP}.{blk}.mlp.up_proj": {"trellis": trellis_of(rad, w, h // 16, None), "suh": f16_plane(rad, w, "suh.1"), "svh": f16_plane(rad, w, "svh")[h:]}}
    return {f"{LP}.{blk}.{FAM[fam][0]}": {"trellis": trellis_of(rad, w), "suh": f16_plane(rad, w, "suh"), "svh": f16_plane(rad, w, "svh")}}


def trunk_weights(rad):
    return sorted((n for n, e in rad.entries.items() if n.startswith("blk.") and e.scheme == "exl3i" and n.split(".")[2] in FAM),
                  key=lambda n: (int(n.split(".")[1]), n))


def write_level(out, per_key, model_dir, cb="mad"):
    """per_key: {level key: {trellis, suh, svh}} -> an ourq-layout level dir"""
    import shutil, torch
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    groups = {}
    for k in per_key:
        g = f"q-L{int(k.split('.layers.')[1].split('.')[0]):02d}" if ".layers." in k else "q-head"
        groups.setdefault(g, []).append(k)
    wmap = {}
    for g, ks in sorted(groups.items()):
        t = {}
        for k in ks:
            d = per_key[k]
            t[f"{k}.trellis"] = torch.from_numpy(d["trellis"]); t[f"{k}.suh"] = torch.from_numpy(d["suh"]); t[f"{k}.svh"] = torch.from_numpy(d["svh"])
            t[f"{k}.mul1"] = torch.zeros(1, dtype=torch.int32)
        for n in t: wmap[n] = f"{g}.safetensors"
        save_file(t, f"{out}/{g}.safetensors.part", metadata={"rad2level214": "1"}); os.replace(f"{out}/{g}.safetensors.part", f"{out}/{g}.safetensors")
    json.dump({"metadata": {}, "weight_map": wmap}, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
    shutil.copy(f"{model_dir}/config.json", f"{out}/config.json")
    json.dump([{"codebook": cb, "K": "mixed", "args": {"K": "mixed", "codebook": cb}}], open(f"{out}/stamp.json", "w"), indent=1)
    open(f"{out}/.done", "w").close()


def decode_ab(rad, blk, dev="cpu"):
    """the ssm_ab record of layer blk decoded to (a [48, k], b [48, k]) f32 (rows 0-47 = a, 48-95 = b: verified by --check)"""
    import torch
    from rxq import deq
    w = f"blk.{blk}.ssm_ab.weight"
    e = rad.entries[w]; cb, m, _ = parse_transform(e.transform)
    tr = torch.from_numpy(trellis_of(rad, w).astype(np.int16))
    W = deq.weight(tr.to(dev), torch.from_numpy(f16_plane(rad, w, "suh")).to(dev), torch.from_numpy(f16_plane(rad, w, "svh")).to(dev),
                   m / 16, torch.from_numpy(deq.lut(cb)).to(dev)).T          # [rows, k]
    return W[:48], W[48:96]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rad"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--check", type=int, default=0); ap.add_argument("--skip-write", action="store_true")
    a = ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from rxq import deq
    rad = Rad(a.rad)
    mp = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
    LPm = "model.language_model." if any(k.startswith("model.language_model.") for k in mp) else "model."
    assert LPm == "model.language_model.", LPm
    ws = trunk_weights(rad)
    assert len(ws) == 336, len(ws)          # 400 level keys: ffn_gate_up is two
    mapping = {}
    if not a.skip_write and not os.path.exists(f"{a.out}/trunk/.done"):
        per = {}
        for i, w in enumerate(ws):
            t = tensors_of(rad, w)
            per.update(t)
            e = rad.entries[w]; cb, m, _ = parse_transform(e.transform)
            mapping[w] = {"keys": list(t), "rate": m / 16, "rows": e.shape[0], "cols": e.shape[1], "stacked": w.endswith("ffn_gate_up.weight")}
            if (i + 1) % 80 == 0: print(f"read {i + 1}/{len(ws)}", flush=True)
        assert {rad.entries[w].transform.split("/")[0] for w in ws} == {"mad"}
        write_level(f"{a.out}/trunk", per, a.model); del per
        ho = rad.entries["output.weight"]; cb, m, r16 = parse_transform(ho.transform); assert r16 and cb == "mad"
        write_level(f"{a.out}/head", {"lm_head": {"trellis": trellis_of(rad, "output.weight"), "suh": f16_plane(rad, "output.weight", "suh"),
                                                 "svh": f16_plane(rad, "output.weight", "svh")}}, a.model)
        np.save(f"{a.out}/R16.npy", r16_perm(ho.shape[0]).astype(np.int64))
        ab = {}
        for blk in range(64):
            if f"blk.{blk}.ssm_ab.weight" in rad.entries:
                A, B = decode_ab(rad, blk)
                ab[f"blk.{blk}.ssm_a.weight"] = A.to(torch.bfloat16).contiguous(); ab[f"blk.{blk}.ssm_b.weight"] = B.to(torch.bfloat16).contiguous()
        save_file(ab, f"{a.out}/ab.safetensors.part"); os.replace(f"{a.out}/ab.safetensors.part", f"{a.out}/ab.safetensors")
        json.dump({"rad": os.path.abspath(a.rad), "weights": mapping, "head": {"rate": m / 16, "rows": ho.shape[0], "cols": ho.shape[1]}},
                  open(f"{a.out}/map214.json", "w"), indent=1)
        print(f"WROTE {len(ws)} trunk weights, head K{m / 16}, {len(ab) // 2} ab layers -> {a.out}")
    if a.check:
        rng = np.random.default_rng(0)
        cks = [ws[i] for i in rng.choice(len(ws), a.check, replace=False)]
        lev = deq.Level(f"{a.out}/trunk", "cpu")
        wf = {}
        def ckpt(k):
            f = mp[k + ".weight"]
            if f not in wf: wf[f] = safe_open(f"{a.model}/{f}", "pt", device="cpu")
            return wf[f].get_tensor(k + ".weight").float()
        for w in cks:
            for k in tensors_of(rad, w):
                W, Wb = lev.get(k), ckpt(k)
                print(f"check {k}: K {lev.K(k)} shape {tuple(W.shape)} rel err {float((W - Wb).norm() / Wb.norm()):.4f}", flush=True)
        for blk in (0, 1, 2):
            A, B = decode_ab(rad, blk)
            a_, b_ = ckpt(f"{LPm}layers.{blk}.linear_attn.in_proj_a"), ckpt(f"{LPm}layers.{blk}.linear_attn.in_proj_b")
            for nm, X, Y in (("a", A, a_), ("b", B, b_), ("a<-b", A, b_), ("b<-a", B, a_)):
                print(f"check ab layer {blk} {nm}: rel err {float((X - Y).norm() / Y.norm()):.4f}", flush=True)
        print("CHECK DONE")


if __name__ == "__main__":
    main()
