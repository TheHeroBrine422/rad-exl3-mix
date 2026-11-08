"""blkorder84.py -- #117 route B (../w84.md): exllamav3's LDLQ visits the residual readers' 128-blocks in the order of a
stored permutation, everything else exactly as the natural run. Equivalent to quantizing the residual-permuted model
(perm117.py) with the natural signs permuted along, then storing the result back in natural order.

install(Q, order): Q = exllamav3's quantize module (after rxq / ourq set it up). order = path of w82's order82.json
("stored_order"[p] = the natural block at stored position p; LDLQ runs last-to-first, so stored_order[39] goes first),
or "ident" (identity order: the wrappers still run, output must equal the plain run bit for bit -- the noop gate).
Applies to a tensor quantized against a residual-input Hessian (layers.N.input, layers.N.post_attention.input;
k == 5120) with scaled_hessian (the only path the ladder uses; anything else is refused):
  scaled_ldl(H, su)  -> L of the PERMUTED metric:  scaled_ldl(H[idx][:, idx], su[idx])
  ldlq(W, L)         -> ldlq(W[idx], L), outputs moved back: weight_q[inv], encoded tile rows [tinv]
Not applied: writers, lm_head, anything with H_out (ldlq_2hess: refused while active)."""
import json, re
import torch

D, B = 5120, 128
_ST = {"on": False, "n": 0, "enabled": True}     # enabled: signchk84 switches route B off for its route-A runs


def _hname_reader(h):
    return bool(re.search(r"\.layers\.\d+\.input$", h or "") or (h or "").endswith(".post_attention.input"))


def install(Q, order):
    if getattr(Q, "_blkorder84", None) is not None:
        assert Q._blkorder84 == order, "blkorder84 already installed with another order"
        return
    so = list(range(D // B)) if order == "ident" else json.load(open(order))["stored_order"]
    assert sorted(so) == list(range(D // B))
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
        act = _ST["enabled"] and _hname_reader(H_data.get("first_key")) and weight.shape[0] == D
        if act:
            assert quant_args.get("scaled_hessian"), "blkorder84: only the scaled_hessian path"
            assert quant_args.get("H_out") is None, "blkorder84: not with H_out (ldlq_2hess)"
        _ST["on"] = act; _ST["used"] = {"sldl": 0, "ldlq": 0}
        try:
            r = q_exl3(weight, H_data, quant_args, *a, **kw)
        finally:
            _ST["on"] = False
        if act:
            assert _ST["used"] == {"sldl": 1, "ldlq": 1}, f"blkorder84: wrappers ran {_ST['used']} (expected 1 / 1)"
            _ST["n"] += 1
        return r

    def scaled_ldl(H_orig, su, quant_args):
        if not _ST["on"]: return sldl(H_orig, su, quant_args)
        _ST["used"]["sldl"] += 1
        i = on("idx", H_orig)
        Hp = H_orig.index_select(0, i).index_select(1, i)
        sp = su.view(-1).index_select(0, on("idx", su)).view(su.shape)
        return sldl(Hp, sp, quant_args)

    def ldlq_(weight, L, quant_args, pb=None):
        if not _ST["on"]: return ldlq(weight, L, quant_args, pb)
        _ST["used"]["ldlq"] += 1
        wq, enc = ldlq(weight.index_select(0, on("idx", weight)).contiguous(), L, quant_args, pb)
        return wq.index_select(0, on("inv", wq)).contiguous(), enc.index_select(0, on("tinv", enc)).contiguous()

    def ldlq_2hess(*a, **kw):
        assert not _ST["on"], "blkorder84: ldlq_2hess on a reordered tensor"
        return ldlq2(*a, **kw)

    Q.quantize_exl3, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess = quantize_exl3, scaled_ldl, ldlq_, ldlq_2hess
    Q._blkorder84 = order
    print(f"blkorder84: installed, order {'identity' if order == 'ident' else order} "
          f"(LDLQ first visits natural block {so[-1]}, last {so[0]})", flush=True)


def count():
    return _ST["n"]
