# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""#117 residual block order (quant.block_order), route B: the residual readers' LDLQ visits their 40 input 128-blocks
in a stored order; nothing else changes. Port of rad-exl3-mix v3/e3/settle/w84/blkorder84.py (w84), extended to the
two-sided LDLQ (H_out at every level since 11-08: dense and w81's block-diagonal path) and to tile CD.

Why route B ships (w84): quantizing the residual-permuted model (route A: permuted checkpoint and Hessians) equals
quantizing the natural model with LDLQ visiting the readers' input blocks in the permuted order, then storing in natural
order, provided route A draws the natural per-position signs permuted along and maps exllamav3's g_scale tile sample
through the permutation (w84 gate G2 on 27B tensors: trellis bit-identical, LDLQ inputs equal). Route B needs neither
trap: the signs, the g_scale sample, the writers, the embedding, the head, the norms, MTP, DFlash taps and the vision
merger are the natural model's, and the stored codes are in natural order, so no consumer of the residual needs a
reverse permutation and no permuted checkpoint / Hessian is a job input. The order itself (40 ints) is in the settings
file, so it is part of the settings stamp.

install(Q, stored_order): stored_order[p] = the natural block at stored position p (w82 / w84 order82.json
"stored_order"); exllamav3's LDLQ runs last-to-first, so natural block stored_order[39] is visited first. Applies to a
tensor quantized against a residual-input Hessian (layers.N.input, layers.N.post_attention.input; k = 5120) with
scaled_hessian (the ladder's path; anything else is refused):
  scaled_ldl(H, su)            -> scaled_ldl(H[idx][:, idx], su[idx])      (the permuted metric's L)
  ldlq(W, L) / ldlq_2hess(W, L, L_out)   -> on W[idx]; weight_q[inv], encoded tile rows [tinv]   (L_out: output side)
Tile CD's wrappers sit inside (installed first), so CD sweeps the permuted rows with the permuted metric, as route A.
The identity order (list(range(40))) runs the wrappers and must give the plain codes bit for bit (tests/check_blkorder.py)."""
import re
import torch

D, B = 5120, 128
NB = D // B
_ST = {"on": False, "n": 0}


def reader(hname):
    return bool(re.search(r"\.layers\.\d+\.input$", hname or "") or (hname or "").endswith(".post_attention.input"))


def check_order(so):
    so = [int(x) for x in so]
    if sorted(so) != list(range(NB)): raise ValueError(f"block order: not a permutation of 0..{NB - 1}: {so}")
    return so


def install(Q, stored_order):
    so = check_order(stored_order)
    if getattr(Q, "_rxq_blkorder", None) is not None:
        assert Q._rxq_blkorder == so, "block order already installed with another order"
        return
    idx = torch.cat([torch.arange(b * B, (b + 1) * B) for b in so])
    inv = torch.argsort(idx)
    tidx = torch.cat([torch.arange(b * 8, (b + 1) * 8) for b in so])
    tinv = torch.argsort(tidx)
    dv = {}

    def on(t, x):                                    # index tensor on x's device
        k = (t, str(x.device))
        if k not in dv: dv[k] = {"idx": idx, "inv": inv, "tidx": tidx, "tinv": tinv}[t].to(x.device)
        return dv[k]

    q_exl3, sldl, ldlq, ldlq2 = Q.quantize_exl3, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess

    def quantize_exl3(weight, H_data, quant_args, *a, **kw):
        act = reader(H_data.get("first_key")) and weight.shape[0] == D
        if act: assert quant_args.get("scaled_hessian"), "block order: only the scaled_hessian path"
        _ST["on"] = act; _ST["used"] = {"sldl": 0, "ldlq": 0}
        try:
            r = q_exl3(weight, H_data, quant_args, *a, **kw)
        finally:
            _ST["on"] = False
        if act:
            assert _ST["used"] == {"sldl": 1, "ldlq": 1}, f"block order: wrappers ran {_ST['used']} (expected 1 / 1)"
            _ST["n"] += 1
            quant_args["block_order"] = True
        return r

    def scaled_ldl(H_orig, su, quant_args):
        if not _ST["on"]: return sldl(H_orig, su, quant_args)
        _ST["used"]["sldl"] += 1
        i = on("idx", H_orig)
        return sldl(H_orig.index_select(0, i).index_select(1, i), su.view(-1).index_select(0, on("idx", su)).view(su.shape),
                    quant_args)

    def back(wq, enc):
        return wq.index_select(0, on("inv", wq)).contiguous(), enc.index_select(0, on("tinv", enc)).contiguous()

    def ldlq_(weight, L, quant_args, pb=None):
        if not _ST["on"]: return ldlq(weight, L, quant_args, pb)
        _ST["used"]["ldlq"] += 1
        return back(*ldlq(weight.index_select(0, on("idx", weight)).contiguous(), L, quant_args, pb))

    def ldlq_2hess(weight, L_in, L_out, quant_args, pb=None):
        if not _ST["on"]: return ldlq2(weight, L_in, L_out, quant_args, pb)
        _ST["used"]["ldlq"] += 1
        return back(*ldlq2(weight.index_select(0, on("idx", weight)).contiguous(), L_in, L_out, quant_args, pb))

    Q.quantize_exl3, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess = quantize_exl3, scaled_ldl, ldlq_, ldlq_2hess
    Q._rxq_blkorder = so
    print(f"block order: installed (LDLQ first visits natural block {so[-1]}, last {so[0]}"
          f"{', identity' if so == list(range(NB)) else ''})", flush=True)


def count():
    return _ST["n"]
