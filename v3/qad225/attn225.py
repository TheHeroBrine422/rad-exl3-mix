#!/usr/bin/env python3
"""w225: the segment attention at depth (a 1,024-token query segment after P carried tokens, 24 q heads / 4 KV heads x 256, bf16)
on one card: today's seg_attention (repeat_interleave + causal_lower_right) vs sdpa's enable_gqa, forward + backward (grad to q and
the segment's own k / v; the carried prefix detached), peak memory and time; and the two forms' outputs / grads compared at a small P.
python3 attn225.py --depths 8192,65536,131072,262144 [--forms rep,gqa]"""
import argparse, time
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

ap = argparse.ArgumentParser(); ap.add_argument("--depths", default="8192,65536"); ap.add_argument("--forms", default="rep,gqa")
ap.add_argument("--n", type=int, default=1024); a = ap.parse_args()
dev = torch.device("cuda:0"); BF = torch.bfloat16; H, KV, D = 24, 4, 256


def attn(form, q, k, v):
    L, S = q.shape[2], k.shape[2]
    if form == "rep":
        k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(L, S) if S != L else None, is_causal=S == L)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(L, S) if S != L else None, is_causal=S == L, enable_gqa=True)


def run(form, P, n, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    pk = torch.randn(1, KV, P, D, device=dev, dtype=BF, generator=g); pv = torch.randn(1, KV, P, D, device=dev, dtype=BF, generator=g)
    q = torch.randn(1, H, n, D, device=dev, dtype=BF, generator=g).requires_grad_()
    k = torch.randn(1, KV, n, D, device=dev, dtype=BF, generator=g).requires_grad_(); v = torch.randn(1, KV, n, D, device=dev, dtype=BF, generator=g).requires_grad_()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated(); t = time.time()
    o = attn(form, q, torch.cat([pk, k], 2), torch.cat([pv, v], 2))
    go = torch.randn_like(o)
    o.backward(go); torch.cuda.synchronize()
    return o.detach(), (q.grad, k.grad, v.grad), time.time() - t, (torch.cuda.max_memory_allocated() - base) / 2**30


for P in [int(x) for x in a.depths.split(",")]:
    res = {}
    for form in a.forms.split(","):
        try:
            run(form, P, a.n)                                  # warm
            o, gr, dt, pk = run(form, P, a.n)
            res[form] = (o, gr); print(f"P {P:7d} {form}: {dt * 1e3:8.1f} ms fwd+bwd, transient peak {pk:6.2f} GiB (inputs held: {2 * KV * P * D * 2 / 2**30:.2f} GiB)", flush=True)
        except torch.OutOfMemoryError as e:
            print(f"P {P:7d} {form}: OOM"); torch.cuda.empty_cache()
    if len(res) == 2:
        (o1, g1), (o2, g2) = res.values()
        d = [float((x.float() - y.float()).abs().max()) for x, y in zip((o1,) + g1, (o2,) + g2)]
        print(f"P {P:7d} rep vs gqa: max |diff| out {d[0]:.3g} dq {d[1]:.3g} dk {d[2]:.3g} dv {d[3]:.3g}; bitwise {[torch.equal(x, y) for x, y in zip((o1,) + g1, (o2,) + g2)]}", flush=True)
    del res; torch.cuda.empty_cache()
